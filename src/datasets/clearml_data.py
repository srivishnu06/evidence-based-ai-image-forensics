from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import load_dataset


DEFAULT_TINY_DATASET_ID = "fa50f8f521e84d869134e01276c31843"
DEFAULT_XAIGD_DATASET_ID = "e89e43f6109e4020aac0952af699531b"


def resolve_dataset_root(
    dataset_id: str | None,
    local_dir: str | Path | None = None,
) -> Path:
    """Resolve a dataset root from either a local directory or ClearML Dataset ID."""
    if local_dir is not None:
        root = Path(local_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        return root

    if not dataset_id:
        raise ValueError("Either dataset_id or local_dir must be supplied.")

    try:
        from clearml import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "ClearML is required when using a ClearML Dataset ID. "
            "Install/configure clearml or pass a local dataset directory."
        ) from exc

    dataset = Dataset.get(dataset_id=dataset_id)
    local_copy = dataset.get_local_copy()
    root = Path(local_copy).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"ClearML returned a local dataset path that does not exist: {root}"
        )
    return root


def find_parquet_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No parquet files matching {list(patterns)} were found under {root}"
        )
    return files


def load_parquet_dataset_dict(
    split_to_files: dict[str, list[Path]],
    cache_dir: str | Path | None = None,
):
    data_files = {
        split: [str(p) for p in paths]
        for split, paths in split_to_files.items()
    }
    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
