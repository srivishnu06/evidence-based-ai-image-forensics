from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import ARTIFACT_LABELS, ARTIFACT_TO_INDEX
from src.datasets.clearml_data import (
    DEFAULT_XAIGD_DATASET_ID,
    find_parquet_files,
    load_parquet_dataset_dict,
    resolve_dataset_root,
)
from src.datasets.tiny_genimage import IMAGENET_MEAN, IMAGENET_STD


def _as_pil(image_value) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        raw = image_value.get("bytes")
        path = image_value.get("path")
        if raw is not None:
            return Image.open(BytesIO(raw)).convert("RGB")
        if path:
            return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image representation: {type(image_value)!r}")


def load_xaigd_labeled(
    dataset_id: str = DEFAULT_XAIGD_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    root = resolve_dataset_root(dataset_id=dataset_id, local_dir=local_dir)
    train_files = find_parquet_files(root, ["labeled_train-*.parquet"])
    test_files = find_parquet_files(root, ["labeled_test-*.parquet"])
    ds = load_parquet_dataset_dict(
        {"labeled_train": train_files, "labeled_test": test_files},
        cache_dir=cache_dir,
    )
    return ds, root


def labels_to_multihot(annotations) -> np.ndarray:
    target = np.zeros(len(ARTIFACT_LABELS), dtype=np.int64)
    for ann in annotations or []:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label")
        if label in ARTIFACT_TO_INDEX:
            target[ARTIFACT_TO_INDEX[label]] = 1
    return target


class XAIGDEvidenceDataset(Dataset):
    """X-AIGD labeled images converted to multi-label and patch-map targets."""

    def __init__(
        self,
        raw_dataset,
        image_size: int = 224,
        patch_grid: int = 14,
        train: bool = False,
    ):
        self.ds = raw_dataset
        self.image_size = image_size
        self.patch_grid = patch_grid
        # Keep Stage-2 geometry deterministic so polygon-derived masks stay aligned
        # with the image. Colour/texture augmentation is also avoided because those
        # signals are themselves forensic evidence.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ds)

    def _build_masks(self, image: Image.Image, annotations):
        width, height = image.size
        masks = np.zeros((len(ARTIFACT_LABELS), height, width), dtype=np.float32)
        target = np.zeros(len(ARTIFACT_LABELS), dtype=np.float32)

        for ann in annotations or []:
            if not isinstance(ann, dict):
                continue
            label = ann.get("label")
            if label not in ARTIFACT_TO_INDEX:
                continue

            idx = ARTIFACT_TO_INDEX[label]
            points = ann.get("points") or []
            if len(points) < 3:
                continue

            # Points are expected as [[x, y], ...]. Skip malformed polygons safely.
            try:
                polygon = [(float(p[0]), float(p[1])) for p in points]
            except (TypeError, ValueError, IndexError):
                continue

            layer = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(layer)
            draw.polygon(polygon, fill=255)

            arr = np.asarray(layer, dtype=np.float32) / 255.0
            masks[idx] = np.maximum(masks[idx], arr)
            target[idx] = 1.0

        mask_tensor = torch.from_numpy(masks).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.patch_grid, self.patch_grid),
            mode="area",
        ).squeeze(0)

        return torch.from_numpy(target), mask_tensor

    def __getitem__(self, index):
        sample = self.ds[index]
        image = _as_pil(sample["image"])
        artifact_target, patch_target = self._build_masks(image, sample.get("labels", []))

        return {
            "image": self.transform(image),
            "artifact_target": artifact_target.float(),
            "patch_target": patch_target.float(),
            "uid": sample.get("uid", ""),
            "generator": sample.get("generator", ""),
        }


def _multilabel_targets(raw_dataset) -> np.ndarray:
    return np.stack([labels_to_multihot(labels) for labels in raw_dataset["labels"]])


def split_labeled_train(
    val_fraction: float = 0.15,
    seed: int = 42,
    image_size: int = 224,
    patch_grid: int = 14,
    max_samples: int | None = None,
    dataset_id: str = DEFAULT_XAIGD_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    dataset_dict, root = load_xaigd_labeled(
        dataset_id=dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    base = dataset_dict["labeled_train"]

    if max_samples is not None and max_samples < len(base):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(base), size=max_samples, replace=False))
        base = base.select(selected.tolist())

    y = _multilabel_targets(base)
    indices = np.arange(len(base))

    # Keep images sharing a UID in the same split.
    # Select a deterministic candidate with balanced multilabel prevalence.
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray([str(uid).strip() for uid in base["uid"]])
    if any(not uid for uid in groups):
        raise ValueError("Missing X-AIGD UID; cannot safely group.")

    class_totals = y.sum(axis=0)
    required_val = np.where(
        class_totals >= 20, 2,
        np.where(class_totals >= 5, 1, 0)
    )
    required_train = (class_totals > 0).astype(int)

    splitter = GroupShuffleSplit(
        n_splits=128,
        test_size=val_fraction,
        random_state=seed,
    )

    best_split = None
    best_score = float("inf")
    overall_rate = y.mean(axis=0)

    for candidate_train, candidate_val in splitter.split(
        indices, groups=groups
    ):
        train_counts = y[candidate_train].sum(axis=0)
        val_counts = y[candidate_val].sum(axis=0)
        val_ratio = len(candidate_val) / len(base)

        if np.any(train_counts < required_train):
            continue
        if np.any(val_counts < required_val):
            continue
        if not 0.10 <= val_ratio <= 0.20:
            continue

        val_rate = y[candidate_val].mean(axis=0)
        class_error = np.mean(
            np.abs(val_rate - overall_rate) /
            np.maximum(overall_rate, 0.05)
        )
        score = class_error + 2 * abs(val_ratio - val_fraction)

        if score < best_score:
            best_score = score
            best_split = (candidate_train, candidate_val)

    if best_split is None:
        raise RuntimeError("No suitable UID-grouped split found.")

    train_idx, val_idx = best_split
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    split_method = "uid_group_shuffle_balanced"

    train_raw = base.select(np.asarray(train_idx).tolist())
    val_raw = base.select(np.asarray(val_idx).tolist())

    info = {
        "dataset_root": str(root),
        "split_method": split_method,
        "train_positive_counts": _multilabel_targets(train_raw).sum(axis=0).astype(int).tolist(),
        "val_positive_counts": _multilabel_targets(val_raw).sum(axis=0).astype(int).tolist(),
    }

    return (
        XAIGDEvidenceDataset(train_raw, image_size=image_size, patch_grid=patch_grid, train=False),
        XAIGDEvidenceDataset(val_raw, image_size=image_size, patch_grid=patch_grid, train=False),
        info,
    )


def load_labeled_test(
    image_size: int = 224,
    patch_grid: int = 14,
    max_samples: int | None = None,
    dataset_id: str = DEFAULT_XAIGD_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    dataset_dict, root = load_xaigd_labeled(
        dataset_id=dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    raw = dataset_dict["labeled_test"]
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))
    return XAIGDEvidenceDataset(raw, image_size=image_size, patch_grid=patch_grid, train=False), root


def artifact_pos_weights(dataset: XAIGDEvidenceDataset, max_weight: float = 20.0):
    y = _multilabel_targets(dataset.ds)
    pos = y.sum(axis=0).astype(float)
    neg = len(y) - pos
    weights = np.divide(neg, pos, out=np.ones_like(neg), where=pos > 0)
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32), pos.astype(int).tolist()
