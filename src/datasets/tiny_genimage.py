from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import TINY_GENERATOR_TO_ID, TINY_ID_TO_GENERATOR
from src.datasets.clearml_data import (
    DEFAULT_TINY_DATASET_ID,
    find_parquet_files,
    load_parquet_dataset_dict,
    resolve_dataset_root,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def image_transform(train: bool, image_size: int = 224):
    if train:
        # Mild geometric augmentation only. Strong colour/texture transforms are avoided
        # because those cues are themselves relevant to AI-image forensics.
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TinyGenImageTorchDataset(Dataset):
    def __init__(self, hf_dataset, train: bool, image_size: int = 224):
        self.ds = hf_dataset
        self.transform = image_transform(train=train, image_size=image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        image = _as_pil(sample["image"])
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "generator": torch.tensor(int(sample["generator"]), dtype=torch.long),
        }


def load_tiny_genimage(
    dataset_id: str = DEFAULT_TINY_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    root = resolve_dataset_root(dataset_id=dataset_id, local_dir=local_dir)
    train_files = find_parquet_files(root, ["train-*.parquet"])
    validation_files = find_parquet_files(root, ["validation-*.parquet"])
    ds = load_parquet_dataset_dict(
        {"train": train_files, "validation": validation_files},
        cache_dir=cache_dir,
    )
    return ds, root


def generator_mapping(hf_dataset) -> dict[str, int]:
    """Use dataset feature metadata when available; otherwise use verified mapping."""
    feature = hf_dataset.features.get("generator")
    names = getattr(feature, "names", None)
    if names:
        return {str(name): i for i, name in enumerate(names)}
    return dict(TINY_GENERATOR_TO_ID)


def generator_name(generator_id: int, hf_dataset=None) -> str:
    if hf_dataset is not None:
        mapping = generator_mapping(hf_dataset)
        reverse = {v: k for k, v in mapping.items()}
        if int(generator_id) in reverse:
            return reverse[int(generator_id)]
    return TINY_ID_TO_GENERATOR.get(int(generator_id), f"generator_{int(generator_id)}")


def _holdout_ids(names: Sequence[str], hf_dataset) -> set[int]:
    mapping = generator_mapping(hf_dataset)
    unknown = [name for name in names if name not in mapping]
    if unknown:
        raise ValueError(
            f"Unknown generator name(s): {unknown}. Available mapping: {mapping}"
        )
    return {int(mapping[name]) for name in names}


def _stratified_subsample(indices, strata, max_samples, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples is None or max_samples >= len(indices):
        return indices
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    strata = np.asarray(strata)
    try:
        selected, _ = train_test_split(
            indices,
            train_size=max_samples,
            random_state=seed,
            stratify=strata,
        )
        return np.asarray(selected, dtype=np.int64)
    except ValueError:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(indices, size=max_samples, replace=False))


def load_stage1_train_val(
    holdout_generators=("Midjourney", "Wukong"),
    val_fraction: float = 0.10,
    seed: int = 42,
    image_size: int = 224,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    dataset_id: str = DEFAULT_TINY_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    dataset_dict, root = load_tiny_genimage(
        dataset_id=dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    raw = dataset_dict["train"]
    holdout = _holdout_ids(holdout_generators, raw)

    keep = [
        i
        for i, (label, gen) in enumerate(zip(raw["label"], raw["generator"]))
        if int(label) == 0 or int(gen) not in holdout
    ]
    filtered = raw.select(keep)

    labels = np.asarray(filtered["label"], dtype=np.int64)
    generators = np.asarray(filtered["generator"], dtype=np.int64)
    indices = np.arange(len(filtered))
    strata = np.asarray([f"{y}_{g}" for y, g in zip(labels, generators)])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=strata,
    )

    train_strata = strata[train_idx]
    val_strata = strata[val_idx]
    train_idx = _stratified_subsample(train_idx, train_strata, max_train_samples, seed)
    val_idx = _stratified_subsample(val_idx, val_strata, max_val_samples, seed + 1)

    return (
        TinyGenImageTorchDataset(
            filtered.select(train_idx.tolist()), train=True, image_size=image_size
        ),
        TinyGenImageTorchDataset(
            filtered.select(val_idx.tolist()), train=False, image_size=image_size
        ),
        {
            "dataset_root": str(root),
            "generator_mapping": generator_mapping(raw),
            "holdout_ids": sorted(holdout),
            "filtered_train_pool": len(filtered),
        },
    )


def load_official_validation_dataset(
    image_size: int = 224,
    max_samples: int | None = None,
    dataset_id: str = DEFAULT_TINY_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    dataset_dict, root = load_tiny_genimage(
        dataset_id=dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    raw = dataset_dict["validation"]
    if max_samples is not None:
        n = min(max_samples, len(raw))
        raw = raw.select(range(n))
    return TinyGenImageTorchDataset(raw, train=False, image_size=image_size), raw, root


def load_official_validation_subsets(
    holdout_generators=("Midjourney", "Wukong"),
    image_size: int = 224,
    max_samples: int | None = None,
    dataset_id: str = DEFAULT_TINY_DATASET_ID,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    dataset_dict, _ = load_tiny_genimage(
        dataset_id=dataset_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )
    raw = dataset_dict["validation"]
    holdout = _holdout_ids(holdout_generators, raw)

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    overall = TinyGenImageTorchDataset(raw, train=False, image_size=image_size)
    seen_idx, unseen_idx = [], []
    for i, (label, gen) in enumerate(zip(raw["label"], raw["generator"])):
        label, gen = int(label), int(gen)
        if label == 0:
            seen_idx.append(i)
            unseen_idx.append(i)
        elif gen in holdout:
            unseen_idx.append(i)
        else:
            seen_idx.append(i)

    seen = TinyGenImageTorchDataset(raw.select(seen_idx), train=False, image_size=image_size)
    unseen = TinyGenImageTorchDataset(raw.select(unseen_idx), train=False, image_size=image_size)
    return overall, seen, unseen
