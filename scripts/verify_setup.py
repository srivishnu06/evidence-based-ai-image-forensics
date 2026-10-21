import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())

import torch
import torchvision
import timm
import datasets
import sklearn
import PIL
import clearml

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("datasets:", datasets.__version__)
print("scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)
print("ClearML:", clearml.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory (GB):", round(props.total_memory / 1024**3, 2))

from src.models.stage1_classifiers import SUPPORTED_STAGE1_MODELS, build_stage1_model
from src.models.deit_forensics import DeiTForensicsModel

x = torch.randn(1, 3, 224, 224)
for name in SUPPORTED_STAGE1_MODELS:
    model = build_stage1_model(name, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2), (name, output.shape)
    print(f"{name} shape: PASS")

forensics = DeiTForensicsModel(pretrained=False).eval()
with torch.no_grad():
    out = forensics(x)
assert out["auth_logits"].shape == (1, 2)
assert out["artifact_logits"].shape == (1, 7)
assert out["patch_logits"].shape == (1, 7, 14, 14)
print("Stage-2 model shapes: PASS")
print("Setup looks usable.")
