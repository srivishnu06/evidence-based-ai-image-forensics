from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import ARTIFACT_DISPLAY_NAMES, ARTIFACT_LABELS
from src.datasets.clearml_data import DEFAULT_XAIGD_DATASET_ID
from src.datasets.xaigd import load_labeled_test
from src.evaluation.metrics import localization_metrics, multilabel_metrics
from src.models.deit_forensics import DeiTForensicsModel
from src.models.deit_forensics_spatial import DeiTForensicsSpatialDecoder
from src.utils.checkpoint import load_checkpoint
from src.utils.clearml_tracking import create_tracker
from src.utils.results import save_csv, save_json


@torch.no_grad()
def main(args):
    tracker = create_tracker(
        enabled=args.clearml,
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or "Final Stage2 Evidence Evaluation",
        task_type="testing",
        tags=["evaluation", "stage2", "artifact-classification", "localization"],
    )
    tracker.connect_parameters(vars(args), name="EvidenceEvaluation")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
        model_type = ckpt.get("model_type")
        if model_type == "deit_forensics_spatial":
            model = DeiTForensicsSpatialDecoder(pretrained=False)
        elif model_type == "deit_forensics":
            model = DeiTForensicsModel(pretrained=False)
        else:
            raise ValueError(f"Unsupported Stage 2 model: {model_type!r}")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device).eval()

        artifact_thresholds = (
            np.full(len(ARTIFACT_LABELS), args.artifact_threshold, dtype=float)
            if args.artifact_threshold is not None
            else np.asarray(ckpt.get("artifact_thresholds", [0.5] * len(ARTIFACT_LABELS)), dtype=float)
        )
        localization_threshold = (
            float(args.localization_threshold)
            if args.localization_threshold is not None
            else float(ckpt.get("localization_threshold", 0.5))
        )
        dataset_id = args.dataset_id or ckpt.get("dataset_id") or DEFAULT_XAIGD_DATASET_ID

        ds, dataset_root = load_labeled_test(
            image_size=args.image_size,
            patch_grid=args.patch_grid,
            max_samples=args.max_samples,
            dataset_id=dataset_id,
            local_dir=args.dataset_local_dir,
            cache_dir=args.hf_cache_dir,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

        artifact_true, artifact_prob, patch_true, patch_prob = [], [], [], []
        for batch in tqdm(loader, desc="Evaluating X-AIGD labeled_test"):
            images = batch["image"].to(device, non_blocking=True)
            out = model(images)
            artifact_true.append(batch["artifact_target"].numpy())
            artifact_prob.append(torch.sigmoid(out["artifact_logits"]).cpu().numpy())
            patch_true.append(batch["patch_target"].numpy())
            patch_prob.append(torch.sigmoid(out["patch_logits"]).cpu().numpy())

        artifact_true = np.concatenate(artifact_true)
        artifact_prob = np.concatenate(artifact_prob)
        patch_true = np.concatenate(patch_true)
        patch_prob = np.concatenate(patch_prob)

        art_metrics = multilabel_metrics(artifact_true, artifact_prob, threshold=artifact_thresholds)
        loc_metrics = localization_metrics(patch_true, patch_prob, threshold=localization_threshold)

        print("\nARTIFACT CLASSIFICATION")
        for key in [
            "macro_precision", "macro_recall", "macro_f1",
            "micro_precision", "micro_recall", "micro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
            "map_macro", "map_micro", "map_weighted",
            "exact_match_accuracy", "hamming_loss",
        ]:
            print(f"{key:28s}: {art_metrics[key]}")

        class_rows = []
        print("\nPER-CLASS ARTIFACT METRICS")
        for i, label in enumerate(ARTIFACT_LABELS):
            row = {
                "artifact": label,
                "display_name": ARTIFACT_DISPLAY_NAMES[label],
                "threshold": float(artifact_thresholds[i]),
                "support": int(art_metrics["per_class_support"][i]),
                "precision": art_metrics["per_class_precision"][i],
                "recall": art_metrics["per_class_recall"][i],
                "f1": art_metrics["per_class_f1"][i],
                **loc_metrics["per_class"][i],
            }
            class_rows.append(row)
            print(
                f"{row['display_name']:32s} P={row['precision']:.4f} "
                f"R={row['recall']:.4f} F1={row['f1']:.4f} "
                f"IoU={row['mean_iou']:.4f}"
            )

        print("\nLOCALIZATION")
        for key in [
            "mean_iou", "mean_pixel_precision", "mean_pixel_recall",
            "mean_pixel_f1", "pointing_accuracy", "evaluated_artifact_instances",
        ]:
            print(f"{key:28s}: {loc_metrics[key]}")

        output_dir = Path(args.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "artifact_thresholds": artifact_thresholds.tolist(),
            "localization_threshold": localization_threshold,
            "artifact_metrics": art_metrics,
            "localization_metrics": loc_metrics,
            "per_class": class_rows,
        }
        json_path = save_json(output_dir / "final_evidence_evaluation.json", payload)
        csv_path = save_csv(output_dir / "per_class_evidence_metrics.csv", class_rows)

        for key in ["macro_f1", "micro_f1", "weighted_f1", "map_macro"]:
            tracker.report_single_value(f"artifact_{key}", art_metrics[key])
        for key in ["mean_iou", "mean_pixel_f1", "pointing_accuracy"]:
            tracker.report_single_value(f"localization_{key}", loc_metrics[key])
        for row in class_rows:
            tracker.report_single_value(f"artifact_f1_{row['artifact']}", row["f1"])
        tracker.upload_artifact("stage2_final_evaluation", json_path)
        tracker.upload_artifact("stage2_per_class_metrics", csv_path)

        print("\nSaved:", json_path)
        print("Saved:", csv_path)
    finally:
        tracker.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--dataset-local-dir", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-grid", type=int, default=14)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--artifact-threshold", type=float, default=None)
    p.add_argument("--localization-threshold", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--results-dir", default="outputs/stage2_evaluation")
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--clearml-project", default="Evidence-Based AI Image Forensics")
    p.add_argument("--clearml-task-name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
