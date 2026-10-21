from src.vlm.structured_observer_v2 import (
    clean_visual_observation, render_visual_observation,
)


def test_valid_description():
    raw = "The picture shows a building with tall columns. The crop includes a statue and a row of windows. No clear defect is visible."
    assert clean_visual_observation(raw) == raw
    text = render_visual_observation(raw, "Edge / shape inconsistency")
    assert "No clear defect is visible" in text
    assert "Unverified candidate" in text


def test_broken_json_rejected():
    raw = "```json { I/O{ crop_details (none provided) assessment (unclear) visible_basis (a statue) } ```"
    assert clean_visual_observation(raw) is None
    text = render_visual_observation(raw, "Semantic inconsistency")
    assert "No dependable visual description" in text
    assert "I/O{" not in text


def test_short_empty_rejected():
    assert clean_visual_observation("") is None
    assert clean_visual_observation("The eye is strange.") is None


def test_does_not_claim_verification():
    text = render_visual_observation("", "Physics violation")
    assert "not been independently" in text
    assert "Stage 1" in text
