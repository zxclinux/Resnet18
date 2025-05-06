from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone(True)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self, freeze: bool = True):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = not freeze
