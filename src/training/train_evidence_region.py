from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from src.datasets.clearml_data import DEFAULT_XAIGD_DATASET_ID
from src.datasets.xaigd import artifact_pos_weights, split_labeled_train
from src.evaluation.metrics import (
    find_best_localization_threshold,
    find_best_multilabel_thresholds,
    localization_metrics,
    multilabel_metrics,
)
from src.models.deit_forensics_spatial import DeiTForensicsSpatialDecoder
from src.training.spatial_distribution_loss import BCEWithRegionDistributionLoss
from src.training.train_detector import build_warmup_cosine_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.clearml_tracking import create_tracker
from src.utils.results import save_csv, save_json
from src.utils.seed import seed_everything


class SpatialBCEDiceLoss(nn.Module):
    """Weighted BCE with spatial Dice supervision."""

    def __init__(
        self,
        num_classes: int,
        positive_weight: float = 8.0,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.register_buffer(
            "positive_weights",
            torch.full((1, num_classes, 1, 1), positive_weight)
        )
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.positive_weights,
        )

        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(-2, -1))
        total = (
            probs.sum(dim=(-2, -1))
            + targets.sum(dim=(-2, -1))
        )

        dice = 1.0 - (
            (2.0 * intersection + 1e-6)
            / (total + 1e-6)
        )

        present = targets.sum(dim=(-2, -1)) > 0

        dice_loss = (
            dice[present].mean()
            if present.any()
            else logits.new_zeros(())
        )

        return bce + self.dice_weight * dice_loss


@torch.no_grad()
def collect_evidence_predictions(model, loader, device, artifact_loss_fn, patch_loss_fn, artifact_weight, localization_weight):
    model.eval()
    total, total_artifact, total_patch = 0.0, 0.0, 0.0
    artifact_true, artifact_prob, patch_true, patch_prob = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        artifact_target = batch["artifact_target"].to(device, non_blocking=True)
        patch_target = batch["patch_target"].to(device, non_blocking=True)

        out = model(images)
        artifact_loss = artifact_loss_fn(out["artifact_logits"], artifact_target)
        patch_loss = patch_loss_fn(out["patch_logits"], patch_target)
        loss = artifact_weight * artifact_loss + localization_weight * patch_loss

        n = images.size(0)
        total += loss.item() * n
        total_artifact += artifact_loss.item() * n
        total_patch += patch_loss.item() * n
        artifact_true.append(artifact_target.cpu().numpy())
        artifact_prob.append(torch.sigmoid(out["artifact_logits"]).cpu().numpy())
        patch_true.append(patch_target.cpu().numpy())
        patch_prob.append(torch.sigmoid(out["patch_logits"]).cpu().numpy())

    n = max(1, len(loader.dataset))
    return {
        "loss": total / n,
        "artifact_loss": total_artifact / n,
        "patch_loss": total_patch / n,
        "artifact_true": np.concatenate(artifact_true),
        "artifact_prob": np.concatenate(artifact_prob),
        "patch_true": np.concatenate(patch_true),
        "patch_prob": np.concatenate(patch_prob),
    }


def train(args):
    seed_everything(args.seed)
    run_name = args.run_name or f"deit_evidence_lr{args.lr:g}_seed{args.seed}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker = create_tracker(
        enabled=args.clearml,
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or f"Stage2 {run_name}",
        task_type="training",
        output_uri=args.clearml_output_uri,
        tags=["stage2", "spatial-decoder", "region-distribution"],
    )
    tracker.connect_parameters(vars(args), name="Stage2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tracker.report_text(f"Training device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    detector_ckpt = load_checkpoint(args.detector_checkpoint, map_location="cpu")
    if detector_ckpt.get("model_type") != "deit_tiny":
        raise ValueError("Evidence training requires a Stage-1 DeiT-Tiny checkpoint.")

    model = DeiTForensicsSpatialDecoder(pretrained=False)
    model.load_stage1_detector(detector_ckpt["model_state"])
    model.freeze_authenticity_head()
    model.configure_backbone_finetuning(args.last_n_blocks)
    model.to(device)

    train_ds, val_ds, data_info = split_labeled_train(
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_size=args.image_size,
        patch_grid=args.patch_grid,
        max_samples=args.max_samples,
        dataset_id=args.dataset_id,
        local_dir=args.dataset_local_dir,
        cache_dir=args.hf_cache_dir,
    )

    print("Dataset root:", data_info["dataset_root"])
    print("Split method:", data_info["split_method"])
    print("Train positive counts:", data_info["train_positive_counts"])
    print("Validation positive counts:", data_info["val_positive_counts"])
    tracker.report_single_value("evidence_train_samples", len(train_ds))
    tracker.report_single_value("evidence_internal_val_samples", len(val_ds))
    tracker.report_text(f"Dataset ID: {args.dataset_id}")
    tracker.report_text(f"Split method: {data_info['split_method']}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print("Trainable parameters:", f"{n_trainable:,}")
    tracker.report_single_value("trainable_parameters", n_trainable)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, args.warmup_ratio)

    pos_weight = None
    positive_counts = None
    if not args.disable_artifact_pos_weight:
        pos_weight, positive_counts = artifact_pos_weights(train_ds, max_weight=args.max_pos_weight)
        print("Artifact positive counts:", positive_counts)
        print("Artifact pos_weight:", pos_weight.tolist())
        pos_weight = pos_weight.to(device)
        tracker.report_text(f"Artifact positive counts: {positive_counts}")
        tracker.report_text(f"Artifact pos_weight: {pos_weight.detach().cpu().tolist()}")

    artifact_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    patch_loss_fn = BCEWithRegionDistributionLoss(
        distribution_weight=args.distribution_weight
    ).to(device)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")
    best_spatial_ap = float("-inf")
    best_spatial_epoch = 0
    spatial_checkpoint = Path(args.output).with_name(
        Path(args.output).stem
        + "_best_spatial_ap"
        + Path(args.output).suffix
    )
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_total = running_artifact = running_patch = 0.0
            progress = tqdm(train_loader, desc=f"Evidence epoch {epoch}/{args.epochs}")

            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                artifact_target = batch["artifact_target"].to(device, non_blocking=True)
                patch_target = batch["patch_target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    out = model(images)
                    artifact_loss = artifact_loss_fn(out["artifact_logits"], artifact_target)
                    patch_loss = patch_loss_fn(out["patch_logits"], patch_target)
                    loss = args.artifact_weight * artifact_loss + args.localization_weight * patch_loss

                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                n = images.size(0)
                running_total += loss.item() * n
                running_artifact += artifact_loss.item() * n
                running_patch += patch_loss.item() * n
                progress.set_postfix(total=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            n_train = max(1, len(train_loader.dataset))
            train_loss = running_total / n_train
            train_artifact_loss = running_artifact / n_train
            train_patch_loss = running_patch / n_train

            val = collect_evidence_predictions(
                model, val_loader, device, artifact_loss_fn, patch_loss_fn,
                args.artifact_weight, args.localization_weight
            )
            artifact_metrics = multilabel_metrics(val["artifact_true"], val["artifact_prob"], threshold=0.5)
            loc_metrics = localization_metrics(val["patch_true"], val["patch_prob"], threshold=0.5)
            binary_patches = (
                np.asarray(val["patch_true"]) > 0
            ).astype(np.uint8)

            spatial_prob = np.asarray(val["patch_prob"])

            spatial_ap = float(
                average_precision_score(
                    binary_patches.ravel(),
                    spatial_prob.ravel(),
                )
            )

            pointing = float(
                loc_metrics["pointing_accuracy"]
            )

            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_artifact_loss": train_artifact_loss,
                "train_localization_loss": train_patch_loss,
                "val_loss": val["loss"],
                "val_artifact_loss": val["artifact_loss"],
                "val_localization_loss": val["patch_loss"],
                "val_artifact_macro_f1_at_0.5": artifact_metrics["macro_f1"],
                "val_artifact_micro_f1_at_0.5": artifact_metrics["micro_f1"],
                "val_localization_iou_at_0.5": loc_metrics["mean_iou"],
                "val_localization_pixel_f1_at_0.5": loc_metrics["mean_pixel_f1"],
                "val_global_spatial_ap": spatial_ap,
                "val_pointing_accuracy": pointing,
                "learning_rate": lr,
            }
            history.append(row)
            save_csv(run_dir / "history.csv", history)

            print(
                f"epoch={epoch} train={train_loss:.4f} val={val['loss']:.4f} "
                f"artifact_macro_f1={artifact_metrics['macro_f1']:.4f} "
                f"loc_iou={loc_metrics['mean_iou']:.4f} "
                f"spatial_ap={spatial_ap:.4f} "
                f"pointing={pointing:.4f}"
            )

            tracker.report_scalar("Evidence Loss", "train_total", train_loss, epoch)
            tracker.report_scalar("Evidence Loss", "validation_total", val["loss"], epoch)
            tracker.report_scalar("Validation", "artifact_macro_f1_at_0.5", artifact_metrics["macro_f1"], epoch)
            tracker.report_scalar("Validation", "artifact_micro_f1_at_0.5", artifact_metrics["micro_f1"], epoch)
            tracker.report_scalar("Validation", "localization_mean_iou_at_0.5", loc_metrics["mean_iou"], epoch)
            tracker.report_scalar("Validation", "localization_pixel_f1_at_0.5", loc_metrics["mean_pixel_f1"], epoch)
            tracker.report_scalar("Optimization", "learning_rate", lr, epoch)
            tracker.report_scalar(
                "Validation", "global_spatial_ap",
                spatial_ap, epoch
            )
            tracker.report_scalar(
                "Validation", "pointing_accuracy",
                pointing, epoch
            )

            if val["loss"] < best_val_loss - args.min_delta:
                best_val_loss = val["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    args.output,
                    model,
                    model_type="deit_forensics_spatial",
                    stage1_checkpoint=args.detector_checkpoint,
                    dataset_id=args.dataset_id,
                    last_n_blocks=args.last_n_blocks,
                    val_loss=val["loss"],
                    best_epoch=epoch,
                    seed=args.seed,
                    image_size=args.image_size,
                    patch_grid=args.patch_grid,
                    decision_threshold=float(detector_ckpt.get("decision_threshold", 0.5)),
                    stage1_holdout_generators=detector_ckpt.get("holdout_generators", []),
                )
                print("Saved best checkpoint:", args.output)
            else:
                epochs_without_improvement += 1

            if spatial_ap > best_spatial_ap + 1e-6:
                best_spatial_ap = spatial_ap
                best_spatial_epoch = epoch

                save_checkpoint(
                    spatial_checkpoint,
                    model,
                    model_type="deit_forensics_spatial",
                    stage1_checkpoint=args.detector_checkpoint,
                    dataset_id=args.dataset_id,
                    last_n_blocks=args.last_n_blocks,
                    val_loss=val["loss"],
                    val_spatial_ap=spatial_ap,
                    val_pointing_accuracy=pointing,
                    best_epoch=epoch,
                    seed=args.seed,
                    image_size=args.image_size,
                    patch_grid=args.patch_grid,
                    decision_threshold=float(
                        detector_ckpt.get(
                            "decision_threshold", 0.5
                        )
                    ),
                    stage1_holdout_generators=detector_ckpt.get(
                        "holdout_generators", []
                    ),
                )

                print(
                    "Saved best-spatial-AP checkpoint:",
                    spatial_checkpoint
                )

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

        best_ckpt = load_checkpoint(args.output, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state"], strict=True)
        model.to(device)
        val = collect_evidence_predictions(
            model, val_loader, device, artifact_loss_fn, patch_loss_fn,
            args.artifact_weight, args.localization_weight
        )
        artifact_thresholds, tuned_artifact_metrics = find_best_multilabel_thresholds(
            val["artifact_true"], val["artifact_prob"]
        )
        loc_threshold, tuned_loc_metrics = find_best_localization_threshold(
            val["patch_true"], val["patch_prob"]
        )

        # Re-save best weights with validation-only thresholds for later held-out test evaluation.
        save_checkpoint(
            args.output,
            model,
            model_type="deit_forensics_spatial",
            stage1_checkpoint=args.detector_checkpoint,
            dataset_id=args.dataset_id,
            last_n_blocks=args.last_n_blocks,
            val_loss=best_val_loss,
            best_epoch=best_epoch,
            seed=args.seed,
            image_size=args.image_size,
            patch_grid=args.patch_grid,
            decision_threshold=float(detector_ckpt.get("decision_threshold", 0.5)),
            stage1_holdout_generators=detector_ckpt.get("holdout_generators", []),
            artifact_thresholds=artifact_thresholds.tolist(),
            localization_threshold=float(loc_threshold),
            tuned_internal_val_artifact_metrics=tuned_artifact_metrics,
            tuned_internal_val_localization_metrics=tuned_loc_metrics,
        )

        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        )
        summary = {
            "run_name": run_name,
            "dataset_id": args.dataset_id,
            "dataset_root": data_info["dataset_root"],
            "split_method": data_info["split_method"],
            "train_samples": len(train_ds),
            "internal_val_samples": len(val_ds),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_spatial_ap": best_spatial_ap,
            "best_spatial_epoch": best_spatial_epoch,
            "best_spatial_checkpoint": str(
                spatial_checkpoint.resolve()
            ),
            "artifact_thresholds": artifact_thresholds.tolist(),
            "localization_threshold": float(loc_threshold),
            "tuned_artifact_metrics": tuned_artifact_metrics,
            "tuned_localization_metrics": tuned_loc_metrics,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "checkpoint": str(Path(args.output).resolve()),
        }
        summary_path = save_json(run_dir / "summary.json", summary)

        tracker.report_single_value("best_evidence_val_loss", best_val_loss)
        tracker.report_single_value("best_epoch", best_epoch)
        tracker.report_single_value(
            "best_spatial_ap", best_spatial_ap
        )
        tracker.report_single_value(
            "best_spatial_epoch", best_spatial_epoch
        )
        tracker.report_single_value("tuned_artifact_macro_f1", tuned_artifact_metrics["macro_f1"])
        tracker.report_single_value("tuned_localization_mean_iou", tuned_loc_metrics["mean_iou"])
        tracker.report_single_value("peak_gpu_memory_gb", peak_gpu_memory_gb)
        tracker.upload_model(
            args.output,
            model_name="deit-forensics-best"
        )
        tracker.upload_model(
            spatial_checkpoint,
            model_name="deit-forensics-spatial-ap-best"
        )
        tracker.upload_artifact("stage2_history", run_dir / "history.csv")
        tracker.upload_artifact("stage2_summary", summary_path)

        print("Best evidence validation loss:", best_val_loss)
        print("Artifact thresholds:", artifact_thresholds.tolist())
        print("Localization threshold:", float(loc_threshold))
        print("Summary:", summary_path)
    finally:
        tracker.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector-checkpoint", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_XAIGD_DATASET_ID)
    p.add_argument("--dataset-local-dir", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-grid", type=int, default=14)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--last-n-blocks", type=int, default=2)
    p.add_argument("--artifact-weight", type=float, default=1.0)
    p.add_argument("--localization-weight", type=float, default=0.5)
    p.add_argument("--distribution-weight", type=float, default=0.1)
    p.add_argument("--spatial-pos-weight", type=float, default=2.0)
    p.add_argument("--spatial-dice-weight", type=float, default=0.1)
    p.add_argument("--disable-artifact-pos-weight", action="store_true")
    p.add_argument("--max-pos-weight", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", default="checkpoints/deit_forensics.pt")
    p.add_argument("--results-dir", default="outputs/stage2")
    p.add_argument("--run-name", default=None)
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--clearml-project", default="Evidence-Based AI Image Forensics")
    p.add_argument("--clearml-task-name", default=None)
    p.add_argument("--clearml-output-uri", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
