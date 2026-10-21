from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.clearml_data import DEFAULT_TINY_DATASET_ID
from src.datasets.tiny_genimage import (
    generator_mapping,
    generator_name,
    load_official_validation_dataset,
)
from src.evaluation.metrics import binary_metrics
from src.models.stage1_classifiers import SUPPORTED_STAGE1_MODELS, build_stage1_model
from src.training.train_detector import predict_binary
from src.utils.checkpoint import load_checkpoint
from src.utils.clearml_tracking import create_tracker
from src.utils.results import save_csv, save_json


def _subset_metrics(y_true, y_prob, mask, threshold):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    return binary_metrics(np.asarray(y_true)[mask], np.asarray(y_prob)[mask], threshold)


def main(args):
    tracker = create_tracker(
        enabled=args.clearml,
        project_name=args.clearml_project,
        task_name=args.clearml_task_name or f"Evaluation {args.model}",
        task_type="testing",
        tags=["evaluation", "stage1", "authenticity", args.model],
    )
    tracker.connect_parameters(vars(args), name="Evaluation")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        checkpoint_type = checkpoint.get("model_type")
        if checkpoint_type and checkpoint_type != args.model:
            raise ValueError(
                f"Checkpoint model_type={checkpoint_type!r}, but --model={args.model!r}."
            )

        model = build_stage1_model(args.model, pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(checkpoint.get("decision_threshold", 0.5))
        )
        dataset_id = args.dataset_id or checkpoint.get("dataset_id") or DEFAULT_TINY_DATASET_ID

        ds, raw, dataset_root = load_official_validation_dataset(
            image_size=args.image_size,
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

        pred = predict_binary(model, loader, device)
        y_true = np.asarray(pred["y_true"], dtype=int)
        y_prob = np.asarray(pred["y_prob"], dtype=float)
        generators = np.asarray(pred["generators"], dtype=int)

        mapping = generator_mapping(raw)
        reverse = {v: k for k, v in mapping.items()}
        holdout_ids = {mapping[name] for name in args.holdout_generators}

        overall = binary_metrics(y_true, y_prob, threshold)
        overall["loss"] = pred["loss"]

        real_mask = y_true == 0
        fake_mask = y_true == 1
        seen_mask = real_mask | (fake_mask & ~np.isin(generators, list(holdout_ids)))
        unseen_mask = real_mask | (fake_mask & np.isin(generators, list(holdout_ids)))

        seen = _subset_metrics(y_true, y_prob, seen_mask, threshold)
        unseen = _subset_metrics(y_true, y_prob, unseen_mask, threshold)

        split_metrics = {"overall": overall, "seen": seen, "unseen": unseen}
        print(f"Decision threshold (from internal validation): {threshold:.4f}")
        for split_name, metrics in split_metrics.items():
            print(f"\n{split_name.upper()}")
            if metrics is None:
                print("No samples")
                continue
            for key in [
                "support", "accuracy", "balanced_accuracy", "precision", "recall",
                "specificity", "f1", "auroc", "average_precision", "mcc"
            ]:
                print(f"{key:22s}: {metrics[key]}")

        per_generator = []
        fake_generator_ids = sorted(set(generators[fake_mask].tolist()))
        for gid in fake_generator_ids:
            gen_fake_mask = fake_mask & (generators == gid)
            binary_mask = real_mask | gen_fake_mask
            metrics = binary_metrics(y_true[binary_mask], y_prob[binary_mask], threshold)
            fake_probs = y_prob[gen_fake_mask]
            row = {
                "generator_id": int(gid),
                "generator": reverse.get(int(gid), generator_name(int(gid), raw)),
                "is_holdout": bool(int(gid) in holdout_ids),
                "fake_samples": int(gen_fake_mask.sum()),
                "fake_detection_rate": float((fake_probs >= threshold).mean()) if len(fake_probs) else 0.0,
                "mean_fake_probability": float(fake_probs.mean()) if len(fake_probs) else 0.0,
                **{
                    k: metrics[k]
                    for k in [
                        "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                        "f1", "auroc", "average_precision", "mcc"
                    ]
                },
            }
            per_generator.append(row)

        output_dir = Path(args.results_dir) / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "threshold": threshold,
            "holdout_generators": args.holdout_generators,
            "generator_mapping": mapping,
            "splits": split_metrics,
            "per_generator": per_generator,
        }
        json_path = save_json(output_dir / "final_evaluation.json", payload)
        csv_path = save_csv(output_dir / "per_generator_metrics.csv", per_generator)

        for split_name, metrics in split_metrics.items():
            if metrics is None:
                continue
            for metric_name in [
                "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                "f1", "auroc", "average_precision", "mcc"
            ]:
                tracker.report_single_value(f"{split_name}_{metric_name}", metrics[metric_name])
        tracker.report_single_value("decision_threshold", threshold)
        tracker.report_text(f"Generator mapping: {mapping}")
        tracker.upload_artifact("stage1_final_evaluation", json_path)
        tracker.upload_artifact("stage1_per_generator_metrics", csv_path)

        print("\nPer-generator metrics saved to:", csv_path)
        print("Full evaluation saved to:", json_path)
    finally:
        tracker.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=SUPPORTED_STAGE1_MODELS, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--dataset-local-dir", default=None)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--holdout-generators", nargs="+", default=["Midjourney", "Wukong"])
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override checkpoint threshold. Default uses threshold tuned on internal validation.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--results-dir", default="outputs/stage1_evaluation")
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--clearml-project", default="Evidence-Based AI Image Forensics")
    p.add_argument("--clearml-task-name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
