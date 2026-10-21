import cv2
import numpy as np
from PIL import Image


def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    denom = arr.max()
    if denom > 0:
        arr = arr / denom
    return arr


def resize_heatmap(heatmap, size):
    """Resize probabilities without per-image normalization."""
    hm = np.asarray(heatmap, dtype=np.float32)

    if hm.ndim != 2 or not np.isfinite(hm).all():
        raise ValueError("Expected a finite 2D probability map")

    hm = np.clip(hm, 0.0, 1.0)

    resized = cv2.resize(
        hm,
        dsize=size,
        interpolation=cv2.INTER_LINEAR,
    )

    return np.clip(resized, 0.0, 1.0)


def overlay_heatmap(image: Image.Image, heatmap, alpha: float = 0.55):
    """Absolute-probability red overlay, not min-max normalized."""
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    hm = resize_heatmap(heatmap, image.size)

    opacity = np.where(
        hm >= 0.10,
        hm * alpha,
        0.0,
    )[..., None]

    red = np.zeros_like(arr)
    red[..., 0] = 255.0

    blended = (
        arr * (1.0 - opacity)
        + red * opacity
    )

    return Image.fromarray(
        np.uint8(np.clip(blended, 0, 255))
    )


def extract_top_regions(
    image: Image.Image,
    heatmap,
    threshold: float = 0.60,
    top_k: int = 3,
    min_area_fraction: float = 0.01,
    padding_fraction: float = 0.05,
):
    image = image.convert("RGB")
    width, height = image.size
    hm = resize_heatmap(heatmap, image.size)

    binary = np.uint8(hm >= threshold) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    min_area = width * height * min_area_fraction
    candidates = []

    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]

        if area < min_area:
            continue

        region_mask = labels == idx
        mean_evidence = float(hm[region_mask].mean())
        score = float(mean_evidence * area)

        pad_x = int(w * padding_fraction)
        pad_y = int(h * padding_fraction)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        candidates.append(
            {
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "mean_evidence": mean_evidence,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = candidates[:top_k]

    for item in selected:
        item["crop"] = image.crop(item["bbox"])

    return selected
