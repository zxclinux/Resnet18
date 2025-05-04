from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class FilteredCIFAR10(Dataset):
    def __init__(self, root="data", train=True, transform=None, num_classes=10):
        self.transform = transform
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        for img, label in zip(self.dataset.data, self.dataset.targets):
            if label < num_classes:
                self.data.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
