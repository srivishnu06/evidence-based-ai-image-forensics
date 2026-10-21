import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.constants import ARTIFACT_LABELS, ARTIFACT_DISPLAY_NAMES
from src.datasets.tiny_genimage import IMAGENET_MEAN, IMAGENET_STD
from src.models.deit_forensics import DeiTForensicsModel
from src.models.deit_forensics_spatial import DeiTForensicsSpatialDecoder
from src.models.stage1_classifiers import build_stage1_model
from src.utils.checkpoint import load_checkpoint
from src.inference.evidence import overlay_heatmap, extract_top_regions


class ForensicsPipeline:
    def __init__(self, checkpoint_path, device=None, detector_checkpoint_path="checkpoints/deit_tiny_detector.pt"):
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = load_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )

        model_type = ckpt.get("model_type")

        if model_type == "deit_forensics":
            self.model = DeiTForensicsModel(pretrained=False)
        elif model_type == "deit_forensics_spatial":
            required = (
                "artifact_thresholds",
                "localization_threshold",
            )
            missing = [
                key for key in required
                if key not in ckpt
            ]
            if missing:
                raise ValueError(
                    "Spatial checkpoint is not calibrated. "
                    f"Missing: {missing}"
                )
            self.model = DeiTForensicsSpatialDecoder(
                pretrained=False
            )
        else:
            raise ValueError(
                f"Unsupported Stage 2 model type: {model_type!r}"
            )

        self.model.load_state_dict(
            ckpt["model_state"],
            strict=True,
        )
        self.model.to(self.device).eval()

        detector_ckpt = load_checkpoint(
            detector_checkpoint_path, map_location="cpu"
        )
        if detector_ckpt.get("model_type") != "deit_tiny":
            raise ValueError(
                "Authenticity checkpoint must be Stage-1 DeiT-Tiny"
            )

        self.detector = build_stage1_model(
            "deit_tiny", pretrained=False
        )
        self.detector.load_state_dict(
            detector_ckpt["model_state"], strict=True
        )
        self.detector.to(self.device).eval()
        self.detector_checkpoint_path = str(detector_checkpoint_path)

        self.auth_threshold = float(
            detector_ckpt.get("decision_threshold", 0.5)
        )
        self.artifact_thresholds = np.asarray(
            ckpt.get("artifact_thresholds", [0.5] * len(ARTIFACT_LABELS)),
            dtype=np.float32,
        )
        self.localization_threshold = float(ckpt.get("localization_threshold", 0.5))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image):
        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)

        out = self.model(tensor)

        ai_probability = torch.softmax(
            self.detector(tensor),
            dim=1,
        )[0, 1].item()

        artifact_probability = torch.sigmoid(
            out["artifact_logits"]
        )[0].cpu().numpy()

        patch_probability = torch.sigmoid(
            out["patch_logits"]
        )[0].cpu().numpy()

        artifact_scores = {
            ARTIFACT_DISPLAY_NAMES[label]: float(artifact_probability[i])
            for i, label in enumerate(ARTIFACT_LABELS)
        }
        artifact_detected = {
            ARTIFACT_DISPLAY_NAMES[label]: bool(artifact_probability[i] >= self.artifact_thresholds[i])
            for i, label in enumerate(ARTIFACT_LABELS)
        }

        strongest_idx = int(np.argmax(artifact_probability))
        strongest_label = ARTIFACT_LABELS[strongest_idx]
        strongest_map = patch_probability[strongest_idx]

        overlay = overlay_heatmap(
            original,
            strongest_map,
        )

        regions = extract_top_regions(
            original,
            strongest_map,
            threshold=self.localization_threshold,
            top_k=3,
        )

        return {
            "ai_probability": float(ai_probability),
            "is_ai_generated": bool(ai_probability >= self.auth_threshold),
            "auth_threshold": self.auth_threshold,
            "artifact_scores": artifact_scores,
            "artifact_detected": artifact_detected,
            "artifact_thresholds": self.artifact_thresholds.copy(),
            "localization_threshold": self.localization_threshold,
            "strongest_artifact": ARTIFACT_DISPLAY_NAMES[
                strongest_label
            ],
            "strongest_artifact_probability": float(
                artifact_probability[strongest_idx]
            ),
            "patch_probabilities": patch_probability,
            "evidence_overlay": overlay,
            "regions": regions,
            "input_tensor": tensor,
        }
