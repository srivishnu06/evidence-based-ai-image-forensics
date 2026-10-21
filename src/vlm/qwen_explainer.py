from pathlib import Path
import os

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "2")

import torch


SYSTEM_PROMPT = """You are an evidence-reporting assistant for an AI image-forensics system.

The computer-vision model has already made the authenticity decision and identified
artifact categories and locations. Your job is only to convert that supplied
evidence into clear, cautious natural language.

Rules:
1. Do not independently change the detector verdict.
2. Do not invent visual defects.
3. Describe only evidence visible in the supplied evidence image/regions and
   supported by the structured CV results.
4. If evidence is weak or unclear, explicitly say so.
5. Do not present the result as definitive proof of image origin.
"""


class QwenEvidenceExplainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit: bool = True,
    ):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }

        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **kwargs,
            )
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )

    def explain(
        self,
        original_image_path,
        evidence_image_path,
        evidence_summary: str,
        max_new_tokens: int = 220,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        original_uri = Path(
            original_image_path
        ).resolve().as_uri()

        evidence_uri = Path(
            evidence_image_path
        ).resolve().as_uri()

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": original_uri,
                    },
                    {
                        "type": "image",
                        "image": evidence_uri,
                    },
                    {
                        "type": "text",
                        "text": (
                            "Structured CV results:\n"
                            f"{evidence_summary}\n\n"
                            "Write a short forensic explanation grounded "
                            "only in these results and the highlighted evidence."
                        ),
                    },
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(
                inputs.input_ids,
                generated,
            )
        ]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
