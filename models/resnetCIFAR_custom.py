from torchvision.models import resnet18
import torch.nn as nn

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
