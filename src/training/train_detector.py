from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.clearml_data import DEFAULT_TINY_DATASET_ID
from src.datasets.tiny_genimage import load_stage1_train_val
from src.evaluation.metrics import binary_metrics, find_best_binary_threshold
from src.models.stage1_classifiers import SUPPORTED_STAGE1_MODELS, build_stage1_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.clearml_tracking import create_tracker
from src.utils.results import save_csv, save_json
from src.utils.seed import seed_everything


@torch.no_grad()
def predict_binary(model, loader, device, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    y_true, y_prob, generators = [], [], []
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        if "generator" in batch:
            generators.extend(batch["generator"].cpu().tolist())

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "generators": generators,
        "loss": total_loss / max(1, len(loader.dataset)),
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    pred = predict_binary(model, loader, device)
    metrics = binary_metrics(pred["y_true"], pred["y_prob"], threshold=threshold)
    metrics["loss"] = pred["loss"]
    return metrics


@torch.no_grad()
def benchmark_inference(model, device, image_size=224, warmup=20, runs=100):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / runs


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(args):
    seed_everything(args.seed)
    run_name = args.run_name or f"{args.model}_lr{args.lr:g}_seed{args.seed}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or f"checkpoints/{args.model}_detector.pt"

    tracker = create_tracker(
        enabled=args.clearml,
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or f"Stage1 {run_name}",
        task_type="training",
        output_uri=args.clearml_output_uri,
        tags=["stage1", "authenticity", args.model],
    )

    params = vars(args).copy()
    params["run_name"] = run_name
    tracker.connect_parameters(params, name="Stage1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tracker.report_text(f"Training device: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_ds, val_ds, data_info = load_stage1_train_val(
        holdout_generators=args.holdout_generators,
        val_fraction=args.val_fraction,
        seed=args.seed,
        image_size=args.image_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        dataset_id=args.dataset_id,
        local_dir=args.dataset_local_dir,
        cache_dir=args.hf_cache_dir,
    )

    print("Dataset root:", data_info["dataset_root"])
    print("Generator mapping:", data_info["generator_mapping"])
    print("Held-out IDs:", data_info["holdout_ids"])
    print("Train samples:", len(train_ds), "Internal validation samples:", len(val_ds))

    tracker.report_single_value("train_samples", len(train_ds))
    tracker.report_single_value("internal_val_samples", len(val_ds))
    tracker.report_text(f"Dataset ID: {args.dataset_id}")
    tracker.report_text(f"Generator mapping: {data_info['generator_mapping']}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = build_stage1_model(args.model, pretrained=not args.scratch).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tracker.report_single_value("total_parameters", total_params)
    tracker.report_single_value("trainable_parameters", trainable_params)
    print(f"Parameters: {total_params:,} total / {trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio
    )

    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_auroc = -math.inf
    best_epoch = 0
    best_threshold = 0.5
    best_metrics = None
    epochs_without_improvement = 0
    history = []

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"{args.model} epoch {epoch}/{args.epochs}")

            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * images.size(0)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            train_loss = running_loss / max(1, len(train_loader.dataset))
            pred = predict_binary(model, val_loader, device, criterion=criterion)
            threshold, tuned_metrics = find_best_binary_threshold(pred["y_true"], pred["y_prob"])
            default_metrics = binary_metrics(pred["y_true"], pred["y_prob"], threshold=0.5)
            tuned_metrics["loss"] = pred["loss"]
            default_metrics["loss"] = pred["loss"]
            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": pred["loss"],
                "learning_rate": lr,
                "tuned_threshold": threshold,
                "accuracy": tuned_metrics["accuracy"],
                "balanced_accuracy": tuned_metrics["balanced_accuracy"],
                "precision": tuned_metrics["precision"],
                "recall": tuned_metrics["recall"],
                "specificity": tuned_metrics["specificity"],
                "f1": tuned_metrics["f1"],
                "auroc": tuned_metrics["auroc"],
                "average_precision": tuned_metrics["average_precision"],
                "mcc": tuned_metrics["mcc"],
                "default_f1_at_0.5": default_metrics["f1"],
            }
            history.append(row)

            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={pred['loss']:.4f} "
                f"AUROC={tuned_metrics['auroc']:.4f} AP={tuned_metrics['average_precision']:.4f} "
                f"F1={tuned_metrics['f1']:.4f} threshold={threshold:.4f}"
            )

            tracker.report_scalar("Loss", "train", train_loss, epoch)
            tracker.report_scalar("Loss", "validation", pred["loss"], epoch)
            tracker.report_scalar("Optimization", "learning_rate", lr, epoch)
            tracker.report_scalar("Validation", "threshold", threshold, epoch)
            for key in [
                "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                "f1", "auroc", "average_precision", "mcc"
            ]:
                tracker.report_scalar("Validation", key, tuned_metrics[key], epoch)
            tracker.report_scalar("Validation", "f1_at_0.5", default_metrics["f1"], epoch)

            improved = (
                not math.isnan(tuned_metrics["auroc"])
                and tuned_metrics["auroc"] > best_auroc + args.min_delta
            )
            if improved:
                best_auroc = tuned_metrics["auroc"]
                best_epoch = epoch
                best_threshold = threshold
                best_metrics = tuned_metrics
                epochs_without_improvement = 0
                save_checkpoint(
                    output_path,
                    model,
                    model_type=args.model,
                    holdout_generators=args.holdout_generators,
                    dataset_id=args.dataset_id,
                    generator_mapping=data_info["generator_mapping"],
                    val_metrics=tuned_metrics,
                    val_metrics_at_0_5=default_metrics,
                    decision_threshold=threshold,
                    best_epoch=epoch,
                    seed=args.seed,
                    image_size=args.image_size,
                    pretrained=not args.scratch,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                )
                print("Saved best checkpoint:", output_path)
                tracker.report_single_value("best_internal_val_auroc", best_auroc)
            else:
                epochs_without_improvement += 1

            save_csv(run_dir / "history.csv", history)

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience}).")
                tracker.report_text(f"Early stopping at epoch {epoch}")
                break

        if best_epoch == 0:
            raise RuntimeError("No valid checkpoint was produced.")

        best_ckpt = load_checkpoint(output_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state"], strict=True)
        model.to(device)

        latency_ms = benchmark_inference(model, device, image_size=args.image_size)
        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        )

        summary = {
            "run_name": run_name,
            "model": args.model,
            "dataset_id": args.dataset_id,
            "dataset_root": data_info["dataset_root"],
            "holdout_generators": args.holdout_generators,
            "generator_mapping": data_info["generator_mapping"],
            "train_samples": len(train_ds),
            "internal_val_samples": len(val_ds),
            "best_epoch": best_epoch,
            "best_internal_val_auroc": best_auroc,
            "best_threshold": best_threshold,
            "best_internal_val_metrics": best_metrics,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "single_image_inference_ms": latency_ms,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "checkpoint": str(Path(output_path).resolve()),
        }
        summary_path = save_json(run_dir / "summary.json", summary)

        tracker.report_single_value("best_epoch", best_epoch)
        tracker.report_single_value("best_threshold", best_threshold)
        tracker.report_single_value("single_image_inference_ms", latency_ms)
        tracker.report_single_value("peak_gpu_memory_gb", peak_gpu_memory_gb)
        if best_metrics:
            for key in [
                "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                "f1", "auroc", "average_precision", "mcc"
            ]:
                tracker.report_single_value(f"best_internal_val_{key}", best_metrics[key])
        tracker.upload_model(output_path, model_name=f"{args.model}-best")
        tracker.upload_artifact("stage1_history", run_dir / "history.csv")
        tracker.upload_artifact("stage1_summary", summary_path)

        print(f"Best epoch: {best_epoch}")
        print(f"Best internal validation AUROC: {best_auroc:.4f}")
        print(f"Tuned internal validation threshold: {best_threshold:.4f}")
        print(f"Single-image inference: {latency_ms:.2f} ms")
        print(f"Peak GPU memory: {peak_gpu_memory_gb:.2f} GB")
        print("Summary:", summary_path)
    finally:
        tracker.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=SUPPORTED_STAGE1_MODELS, default="deit_tiny")
    p.add_argument("--dataset-id", default=DEFAULT_TINY_DATASET_ID)
    p.add_argument("--dataset-local-dir", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--scratch", action="store_true", help="Train without ImageNet weights.")
    p.add_argument("--holdout-generators", nargs="+", default=["Midjourney", "Wukong"])
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--results-dir", default="outputs/stage1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--clearml-project", default="Evidence-Based AI Image Forensics")
    p.add_argument("--clearml-task-name", default=None)
    p.add_argument("--clearml-output-uri", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
