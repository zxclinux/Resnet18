from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class FilteredMNIST(Dataset):
    def __init__(self, root="data", train=True, transform=None, num_classes=10):
        self.transform = transform
        self.dataset = MNIST(root=root, train=train, download=True)
        self.num_classes = num_classes
        self.data = [img for img, label in zip(self.dataset.data, self.dataset.targets) if label < num_classes]
        self.labels = [label for label in self.dataset.targets if label < num_classes]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].unsqueeze(0).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
