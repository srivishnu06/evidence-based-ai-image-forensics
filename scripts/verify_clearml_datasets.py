from pathlib import Path

from src.datasets.clearml_data import (
    DEFAULT_TINY_DATASET_ID,
    DEFAULT_XAIGD_DATASET_ID,
    resolve_dataset_root,
)


def check(name, dataset_id, expected):
    root = resolve_dataset_root(dataset_id)
    print(f"\n{name}")
    print("Dataset ID:", dataset_id)
    print("Local root:", root)
    for pattern, expected_count in expected.items():
        files = sorted(Path(root).rglob(pattern))
        total_gb = sum(p.stat().st_size for p in files) / 1024**3
        print(f"{pattern}: {len(files)} files, {total_gb:.2f} GB")
        if expected_count is not None and len(files) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} files for {pattern}, found {len(files)}"
            )


check(
    "Tiny-GenImage",
    DEFAULT_TINY_DATASET_ID,
    {"train-*.parquet": 14, "validation-*.parquet": 4},
)
check(
    "X-AIGD-Labeled",
    DEFAULT_XAIGD_DATASET_ID,
    {"labeled_train-*.parquet": None, "labeled_test-*.parquet": None},
)
print("\nClearML dataset verification: PASS")
