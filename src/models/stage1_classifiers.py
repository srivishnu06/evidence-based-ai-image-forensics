import torch.nn as nn
import timm
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


DEFAULT_DEIT_MODEL = "deit_tiny_patch16_224.fb_in1k"
SUPPORTED_STAGE1_MODELS = (
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "deit_tiny",
)


class ResNet18BinaryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0BinaryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)


class MobileNetV3LargeBinaryClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v3_large(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)


class DeiTBinaryClassifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = DEFAULT_DEIT_MODEL,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = int(self.backbone.num_features)
        self.classifier = nn.Linear(self.embed_dim, 2)

    def forward_features(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim != 3:
            raise RuntimeError(
                f"Expected token features [B,N,C], got {tuple(features.shape)}"
            )
        num_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        cls_token = features[:, 0]
        patch_tokens = features[:, num_prefix:]
        return cls_token, patch_tokens

    def forward(self, x):
        cls_token, _ = self.forward_features(x)
        return self.classifier(cls_token)


def build_stage1_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        return ResNet18BinaryClassifier(pretrained=pretrained)
    if name == "efficientnet_b0":
        return EfficientNetB0BinaryClassifier(pretrained=pretrained)
    if name == "mobilenet_v3_large":
        return MobileNetV3LargeBinaryClassifier(pretrained=pretrained)
    if name == "deit_tiny":
        return DeiTBinaryClassifier(pretrained=pretrained)
    raise ValueError(f"Unsupported model: {name}")
