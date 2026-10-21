
import torch
import torch.nn as nn

from src.models.deit_forensics import DeiTForensicsModel


class DeiTForensicsSpatialDecoder(DeiTForensicsModel):
    """DeiT with a lightweight convolutional evidence decoder."""

    def __init__(self, pretrained=False, hidden_dim=64):
        super().__init__(pretrained=pretrained)

        self.patch_head = nn.Sequential(
            nn.Conv2d(
                self.embed_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                self.num_artifacts,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        cls_token, patch_tokens, grid = self.forward_features(x)

        batch_size = patch_tokens.shape[0]

        spatial_features = patch_tokens.transpose(1, 2)
        spatial_features = spatial_features.reshape(
            batch_size,
            self.embed_dim,
            grid,
            grid,
        )

        return {
            "auth_logits": self.authenticity_head(cls_token),
            "artifact_logits": self.artifact_head(cls_token),
            "patch_logits": self.patch_head(spatial_features),
        }
