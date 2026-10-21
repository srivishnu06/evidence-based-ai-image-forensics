import numpy as np
from src.inference.evidence_quality import summarize_evidence
from src.vlm.structured_observer import clean_visual_observation, render_visual_observation
from src.constants import ARTIFACT_LABELS, ARTIFACT_DISPLAY_NAMES


def sample_cv():
    names = [ARTIFACT_DISPLAY_NAMES[label] for label in ARTIFACT_LABELS]
    maps = np.zeros((len(names), 14, 14), dtype=np.float32)
    maps[4, 4, 5] = 0.346
    return {
        "ai_probability": .6659, "is_ai_generated": True, "auth_threshold": .514108,
        "artifact_scores": {name: (.6548 if i == 4 else .1) for i, name in enumerate(names)},
        "artifact_detected": {name: (i == 4) for i, name in enumerate(names)},
        "strongest_artifact": names[4],
        "strongest_artifact_probability": .6548,
        "patch_probabilities": maps,
        "localization_threshold": .14,
        "regions": [{"dummy": True}],
    }


def test_no_claim_of_verified_evidence():
    review = summarize_evidence(sample_cv())
    assert review["has_candidate"]
    assert review["human_verified"] is False
    assert review["spatial_peak"] == .346
    assert review["cells_above_threshold"] == 1
    assert "not proven" in review["summary"]


def test_no_candidate_on_missing_flag():
    cv = sample_cv()
    cv["artifact_detected"][cv["strongest_artifact"]] = False
    assert not summarize_evidence(cv)["has_candidate"]



def test_qwen_uncertainty():
    text = (
        "The image shows a parrot with a rounded eye. "
        "No clear visual inconsistency is visible in the crop."
    )
    assert clean_visual_observation(text) == text
    result = render_visual_observation(text, "Semantic inconsistency")
    assert "Unverified candidate" in result

def test_malformed_response_rejected():
    assert clean_visual_observation("```json { broken }") is None
    result = render_visual_observation(
        "```json { broken }", "Physics violation"
    )
    assert "No dependable visual description" in result
