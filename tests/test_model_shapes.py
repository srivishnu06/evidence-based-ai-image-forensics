import torch
import pytest

from src.models.stage1_classifiers import SUPPORTED_STAGE1_MODELS, build_stage1_model
from src.models.deit_forensics import DeiTForensicsModel


@pytest.mark.parametrize("name", SUPPORTED_STAGE1_MODELS)
def test_stage1_model_shape(name):
    model = build_stage1_model(name, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_forensics_shapes():
    model = DeiTForensicsModel(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out["auth_logits"].shape == (2, 2)
    assert out["artifact_logits"].shape == (2, 7)
    assert out["patch_logits"].shape == (2, 7, 14, 14)
