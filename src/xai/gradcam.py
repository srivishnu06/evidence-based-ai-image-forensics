import math
import numpy as np
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class AuthenticityWrapper(nn.Module):
    def __init__(self, forensics_model):
        super().__init__()
        self.model = forensics_model

    def forward(self, x):
        return self.model(x)["auth_logits"]


def make_reshape_transform(num_prefix_tokens: int = 1):
    def reshape_transform(tensor):
        patches = tensor[:, num_prefix_tokens:, :]
        n = patches.shape[1]
        grid = int(math.sqrt(n))

        if grid * grid != n:
            raise RuntimeError(
                f"Cannot reshape {n} tokens into a square grid."
            )

        result = patches.reshape(
            patches.size(0),
            grid,
            grid,
            patches.size(2),
        )
        return result.permute(0, 3, 1, 2)

    return reshape_transform


def authenticity_gradcam(
    forensics_model,
    input_tensor,
    target_class: int = 1,
):
    wrapper = AuthenticityWrapper(forensics_model)
    backbone = forensics_model.backbone

    target_layers = [backbone.blocks[-1].norm1]
    num_prefix = int(getattr(backbone, "num_prefix_tokens", 1))

    cam = GradCAM(
        model=wrapper,
        target_layers=target_layers,
        reshape_transform=make_reshape_transform(num_prefix),
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )

    return np.asarray(grayscale_cam[0], dtype=np.float32)
