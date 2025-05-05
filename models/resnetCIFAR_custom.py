from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
