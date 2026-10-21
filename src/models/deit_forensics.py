import math
import torch.nn as nn
import timm

from src.constants import ARTIFACT_LABELS
from src.models.deit_detector import DEFAULT_DEIT_MODEL


class DeiTForensicsModel(nn.Module):
    """DeiT-Tiny with authenticity, artifact and localization heads."""

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = DEFAULT_DEIT_MODEL,
        num_artifacts: int = len(ARTIFACT_LABELS),
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = int(self.backbone.num_features)
        self.num_artifacts = num_artifacts

        self.authenticity_head = nn.Linear(self.embed_dim, 2)
        self.artifact_head = nn.Linear(self.embed_dim, num_artifacts)
        self.patch_head = nn.Linear(self.embed_dim, num_artifacts)

    def forward_features(self, x):
        tokens = self.backbone.forward_features(x)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected DeiT token features [B,N,C], got {tuple(tokens.shape)}"
            )

        num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, num_prefix:]

        n_patches = patch_tokens.shape[1]
        grid = int(math.sqrt(n_patches))

        if grid * grid != n_patches:
            raise RuntimeError(
                f"Patch count {n_patches} cannot be reshaped to a square grid."
            )

        return cls_token, patch_tokens, grid

    def forward(self, x):
        cls_token, patch_tokens, grid = self.forward_features(x)

        auth_logits = self.authenticity_head(cls_token)
        artifact_logits = self.artifact_head(cls_token)

        patch_logits = self.patch_head(patch_tokens)
        patch_logits = patch_logits.permute(0, 2, 1).contiguous()
        patch_logits = patch_logits.view(
            patch_logits.shape[0],
            self.num_artifacts,
            grid,
            grid,
        )

        return {
            "auth_logits": auth_logits,
            "artifact_logits": artifact_logits,
            "patch_logits": patch_logits,
        }

    def load_stage1_detector(self, detector_state_dict: dict):
        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in detector_state_dict.items()
            if k.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state, strict=True)

        weight = detector_state_dict.get("classifier.weight")
        bias = detector_state_dict.get("classifier.bias")

        if weight is None or bias is None:
            raise KeyError(
                "Stage-1 checkpoint is missing classifier.weight/bias."
            )

        self.authenticity_head.weight.data.copy_(weight)
        self.authenticity_head.bias.data.copy_(bias)

    def freeze_authenticity_head(self):
        for p in self.authenticity_head.parameters():
            p.requires_grad = False

    def configure_backbone_finetuning(self, last_n_blocks: int = 2):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if last_n_blocks <= 0:
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError("Expected DeiT backbone to expose .blocks")

        for block in blocks[-last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
