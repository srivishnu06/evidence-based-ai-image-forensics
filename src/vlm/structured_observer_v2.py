"""Conservative, plain-language Qwen image observation (experimental).

The model is *not* a forensic judge. Its statements are not independently
verified and it cannot override the CV detector. An invalid response fails
closed with an explicit no-observation statement.
"""
import re
from pathlib import Path
from tempfile import TemporaryDirectory

VERSION = "2026-10-20-plain-observation-v2"


def clean_visual_observation(raw: str) -> str | None:
    """Reject obvious generation/format failures; NOT a factuality detector."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    words = text.split()
    if not 9 <= len(words) <= 155:
        return None
    junk = (
        "```", "i/o{", "crop_details", "visible_basis", "assessment (",
        "scene (", "none provided", "{\\n", "<|", "{/", "error traceback",
    )
    lower = text.lower()
    if any(token in lower for token in junk):
        return None
    if text.startswith(('{', '[', '}', ']')) or text.count('{') + text.count('}') >= 2:
        return None
    if sum(ch.isalnum() or ch.isspace() or ch in '.,:;!?()\'"-–—/%' for ch in text) / max(len(text), 1) < .90:
        return None
    return text


def render_visual_observation(raw: str, artifact_category: str) -> str:
    description = clean_visual_observation(raw)
    heading = "QWEN VISUAL OBSERVATION — EXPERIMENTAL, NOT VERIFIED\n"
    if description is None:
        detail = (
            "No dependable visual description was produced by Qwen for this input. "
            "Its raw output was withheld because it failed a basic format-quality check. "
            "This does not determine whether the candidate artifact is genuine."
        )
    else:
        detail = description
    return (
        heading + detail + "\n\n"
        "STAGE 2 MODEL SUGGESTION\n"
        f"The separate evidence model proposed: {artifact_category}. "
        "This category is a prediction and is not a fact established by Qwen.\n\n"
        "EVIDENCE STATUS\n"
        "Unverified candidate. The visual description above has not been independently "
        "checked, and it may contain mistakes. No observable inconsistency should be "
        "treated as confirmed without comparison to the original image and an appropriate "
        "ground-truth or human annotation. The VLM cannot validate image origin or change "
        "the independent Stage 1 authenticity result."
    )


class StructuredQwenObserver:
    """Compatible with the existing FastAPI observer interface and Qwen engine."""

    def __init__(self, engine):
        self.engine = engine

    def observe(self, original_image, evidence_crop, artifact_category: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        if evidence_crop is None:
            return render_visual_observation("", artifact_category)

        with TemporaryDirectory(prefix="forensics_qwen_") as temporary:
            base = Path(temporary)
            full = base / "full.png"
            crop = base / "crop.png"
            original_image.convert("RGB").save(full)
            evidence_crop.convert("RGB").save(crop)

            messages = [
                {"role": "system", "content": (
                    "You are a cautious assistant describing photographs. "
                    "Do not judge whether an image is AI-generated. "
                    "Do not assume an image contains a visual defect. "
                    "Use ordinary sentences, not JSON, bullet points or code. "
                    "Do not invent missing objects, distortions, lettering or physical impossibilities."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        "The next two images show the same scene: first the full original, "
                        "then an unedited close-up. Describe the main visible subject and "
                        "two or three concrete details in the close-up. "
                        "If an irregularity is clearly visible you may describe exactly what "
                        "looks irregular; otherwise say that you cannot identify a clear defect. "
                        "Do not speculate about image origin. Answer in two to four complete "
                        "plain-English sentences. Do not output JSON or field names."
                    )},
                    {"type": "image", "image": full.resolve().as_uri()},
                    {"type": "image", "image": crop.resolve().as_uri()},
                ]},
            ]

            proc = self.engine.processor
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = proc(
                text=[prompt], images=images, videos=videos,
                padding=True, return_tensors="pt",
            ).to(self.engine.model.device)
            with torch.inference_mode():
                generated = self.engine.model.generate(
                    **inputs, max_new_tokens=175, do_sample=False,
                )
            suffixes = [gen[len(source):] for source, gen in zip(inputs.input_ids, generated)]
            raw = proc.batch_decode(
                suffixes, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return render_visual_observation(raw, artifact_category)
