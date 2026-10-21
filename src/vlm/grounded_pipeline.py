
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import numpy as np
from PIL import Image
from src.constants import ARTIFACT_LABELS, ARTIFACT_DISPLAY_NAMES
from qwen_vl_utils import process_vision_info

from src.vlm.report_builder import build_forensic_report


class QwenVisualObserver:
    """Generate visual observations, not authenticity verdicts."""

    def __init__(self, explainer):
        self.model = explainer.model
        self.processor = explainer.processor

    def observe(
        self,
        original_image,
        evidence_crop,
        artifact_category,
        max_new_tokens=130,
    ):
        with TemporaryDirectory(
            prefix="image_forensics_vlm_"
        ) as temporary:
            folder = Path(temporary)

            original_path = folder / "original.png"
            crop_path = folder / "crop.png"

            original_image.convert("RGB").save(
                original_path
            )

            evidence_crop.convert("RGB").save(
                crop_path
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful visual observation "
                        "assistant. Describe only details "
                        "you can see in the supplied images. "
                        "Do not decide whether an image is "
                        "authentic or AI-generated. "
                        "Do not invent details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": original_path.resolve().as_uri(),
                        },
                        {
                            "type": "image",
                            "image": crop_path.resolve().as_uri(),
                        },
                        {
                            "type": "text",
                            "text": (
                                "Image 1 is the original image. "
                                "Image 2 is an enlarged crop "
                                "from the original, without "
                                "heatmap colouring. "
                                "A computer-vision model selected "
                                "the crop for inspection. "
                                "Its suggested artifact category is "
                                f"'{artifact_category}', but "
                                "that prediction may be wrong. "
                                "Inspect the crop for a specific "
                                "visually supported irregularity "
                                "related to that category. "
                                "If no clear irregularity is visible, "
                                "say that the evidence is unclear. "
                                "Do not guess unreadable words. "
                                "Do not invent defects. "
                                "Do not determine authenticity. "
                                "Answer in one or two short sentences."
                            ),
                        },
                    ],
                },
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = (
                process_vision_info(messages)
            )

            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            new_tokens = generated[
                0, inputs.input_ids.shape[1]:
            ]

            return self.processor.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()



def _select_unaltered_crop(image, cv, category):
    """Select a crop using CV probabilities, not overlay pixels."""

    names = [
        ARTIFACT_DISPLAY_NAMES[label]
        for label in ARTIFACT_LABELS
    ]

    if category not in names:
        raise ValueError(
            f"Unknown artifact category: {category}"
        )

    probabilities = np.asarray(
        cv["patch_probabilities"], dtype=np.float32
    )

    assert probabilities.ndim == 3
    assert probabilities.shape[0] == len(names)

    heatmap = probabilities[names.index(category)]

    assert heatmap.shape == (14, 14)
    assert np.isfinite(heatmap).all()

    row, col = np.unravel_index(
        int(np.argmax(heatmap)), heatmap.shape
    )

    width, height = image.size

    cx = (col + 0.5) * width / 14
    cy = (row + 0.5) * height / 14

    side = min(
        width,
        height,
        max(1, int(min(width, height) * 0.52)),
    )

    left = max(
        0,
        min(int(round(cx - side / 2)), width - side),
    )

    top = max(
        0,
        min(int(round(cy - side / 2)), height - side),
    )

    return image.crop(
        (left, top, left + side, top + side)
    ).resize(
        (448, 448), Image.Resampling.LANCZOS
    )


def analyze_image(image, cv_pipeline, observer=None):
    """
    Execute the CV models, optional VLM observation,
    and deterministic forensic reporting.

    The VLM is only invoked when candidate localised
    evidence exists.
    """

    original = image.convert("RGB")

    cv = cv_pipeline.predict(original)

    category = cv["strongest_artifact"]

    evidence_available = (
        bool(
            cv["artifact_detected"].get(
                category, False
            )
        )
        and len(cv["regions"]) > 0
    )

    observation = None
    evidence_crop = None

    if evidence_available:
        evidence_crop = _select_unaltered_crop(
            original, cv, category
        )

    if evidence_available and observer is not None:
        observation = observer.observe(
            original_image=original,
            evidence_crop=evidence_crop,
            artifact_category=category,
        )

    report = build_forensic_report(
        cv,
        vlm_observation=observation,
    )

    return {
        "cv_result": cv,
        "vlm_observation": observation,
        "evidence_crop": evidence_crop,
        "vlm_invoked": (
            evidence_available and observer is not None
        ),
        "forensic_report": report,
    }
