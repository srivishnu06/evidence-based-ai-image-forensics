"""Read-only evidence diagnostics: never changes CV predictions or calibration.

This diagnostic distinguishes model scores and spatial candidate regions from
independently visually confirmed image defects. It does not verify origin.
"""

from typing import Any
import numpy as np
from src.constants import ARTIFACT_LABELS, ARTIFACT_DISPLAY_NAMES


def summarize_evidence(cv: dict[str, Any]) -> dict[str, Any]:
    names = [ARTIFACT_DISPLAY_NAMES[label] for label in ARTIFACT_LABELS]
    chosen = cv["strongest_artifact"]
    idx = names.index(chosen)
    maps = np.asarray(cv["patch_probabilities"], dtype=np.float32)
    if maps.shape[0] != len(names) or maps.shape[1:] != (14, 14):
        raise ValueError(f"Expected 7x14x14 evidence maps; got {maps.shape}")
    if not np.isfinite(maps).all():
        raise ValueError("Evidence probability maps contain NaN or infinity")
    maps = np.clip(maps, 0.0, 1.0)
    threshold = float(cv["localization_threshold"])
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Unexpected localisation threshold")
    selected = maps[idx]
    peak = float(selected.max())
    above = int((selected >= threshold).sum())
    flagged = bool(cv["artifact_detected"].get(chosen, False))
    regions = int(len(cv.get("regions", [])))
    has_candidate = flagged and regions > 0 and above > 0
    rank = sorted(
        [(name, float(maps[i].max()), bool(cv["artifact_detected"].get(name, False)))
         for i, name in enumerate(names)],
        key=lambda entry: entry[1], reverse=True,
    )
    status = "UNVERIFIED CANDIDATE" if has_candidate else "NO SUPPORTED CANDIDATE REGION"
    summary = (
        f"The independent Stage 1 classifier predicts {'AI-generated' if cv['is_ai_generated'] else 'authentic'} "
        f"with model score {float(cv['ai_probability']):.4f} (threshold {float(cv['auth_threshold']):.4f}). "
        f"This is a prediction, not proven image origin. Stage 2's highest category score is "
        f"{chosen} ({float(cv['strongest_artifact_probability']):.4f}; "
        f"category threshold {'passed' if flagged else 'not passed'}). "
        f"The corresponding spatial map peaks at {peak:.3f}, with {above} of 196 grid "
        f"cells at or above its localisation cutoff {threshold:.2f}, and "
        f"{regions} extracted candidate region(s). "
        f"Evidence status: {status}. Neither grid activation nor Qwen agreement "
        f"establishes a visible defect, and Stage 2 regions do not explain which "
        f"pixels the independent Stage 1 classifier actually used."
    )
    return {
        "status": status,
        "has_candidate": has_candidate,
        "selected_category": chosen,
        "selected_category_flagged": flagged,
        "spatial_peak": round(peak, 6),
        "localization_threshold": threshold,
        "cells_above_threshold": above,
        "fraction_above_threshold": round(above / 196, 6),
        "candidate_regions": regions,
        "highest_spatial_peak_category": rank[0][0],
        "highest_spatial_peak": round(rank[0][1], 6),
        "summary": summary,
        "human_verified": False,
    }
