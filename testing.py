import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import time
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

# 1. Dataset
class FilteredMNIST(Dataset):
    def __init__(self, root="data", train=True, transform=None, num_classes=10):
        self.transform = transform
        self.dataset = MNIST(root=root, train=train, download=True)
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
        img = self.data[idx].unsqueeze(0).float() / 255.0
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# 2. ResNet18 for 10 classes
class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Custom, self).__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 3. Training with full test evaluation
def train_and_evaluate(epochs=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    model = ResNet18Custom(num_classes=10).to(device)
    train_dataset = FilteredMNIST(transform=transform_train, num_classes=10)
    test_dataset = FilteredMNIST(train=False, transform=transform_test, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    total_samples = len(train_dataset)
    print(f"Тренувальних зображень: {total_samples}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics_per_epoch = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - epoch_start
        fps = total_samples / epoch_duration
        print(f"⏱️ Epoch {epoch + 1}/{epochs} завершено за {epoch_duration:.2f} с ({fps:.2f} зображень/сек)")

        # Evaluation
        model.eval()
        correct_per_class = [0] * 10
        total_per_class = [0] * 10
        prob_sums_per_class = [0.0] * 10

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                for i in range(images.size(0)):
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    prob_correct = probs[i][true_label].item()

                    total_per_class[true_label] += 1
                    prob_sums_per_class[true_label] += prob_correct
                    if pred_label == true_label:
                        correct_per_class[true_label] += 1

        row = {"Epoch": epoch + 1}
        for i in range(10):
            acc = correct_per_class[i] / total_per_class[i]
            avg_prob = prob_sums_per_class[i] / total_per_class[i]
            row[f"Class{i}_Acc"] = acc
            row[f"Class{i}_Prob"] = avg_prob
        metrics_per_epoch.append(row)

    total_time = time.time() - start_time
    overall_fps = (total_samples * epochs) / total_time
    print(f"\n✅ Повний час тренування: {total_time:.2f} секунд")
    print(f"🚀 Середній FPS: {overall_fps:.2f} зображень/секунда")

    df = pd.DataFrame(metrics_per_epoch)
    df.to_csv("mnist_metrics.csv", index=False)
    return df

# 4. Run
df = train_and_evaluate()

# 5. Plot probabilities
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(df["Epoch"], df[f"Class{i}_Prob"], label=f"Class {i}")
plt.title("Average Correct Class Probabilities (MNIST 0–9)")
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mnist_avg_probabilities.png")
plt.show()

# 6. Plot accuracy
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(df["Epoch"], df[f"Class{i}_Acc"], label=f"Class {i}")
plt.title("Test Accuracy Per Class (MNIST 0–9)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mnist_accuracy.png")
plt.show()
