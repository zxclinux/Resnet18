import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.filtered_mnist import FilteredMNIST
from models.resnet_custom import ResNet18Custom

def train_and_evaluate(epochs=12, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
    ])
    transform_test = transforms.Compose([transforms.Resize((64, 64))])

    model = ResNet18Custom(num_classes=num_classes).to(device)
    train_dataset = FilteredMNIST(transform=transform_train, num_classes=num_classes)
    test_dataset = FilteredMNIST(train=False, transform=transform_test, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_samples = len(train_dataset)
    metrics_per_epoch = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        fps = total_samples / (time.time() - epoch_start)
        print(f"⏱️ Epoch {epoch+1}/{epochs}: {fps:.2f} images/sec")

        # Evaluation
        model.eval()
        correct = [0] * num_classes
        total = [0] * num_classes
        prob_sums = [0.0] * num_classes

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                for i in range(images.size(0)):
                    true = labels[i].item()
                    pred = preds[i].item()
                    prob = probs[i][true].item()
                    total[true] += 1
                    prob_sums[true] += prob
                    if pred == true:
                        correct[true] += 1

        row = {"Epoch": epoch + 1}
        for i in range(num_classes):
            row[f"Class{i}_Acc"] = correct[i] / total[i]
            row[f"Class{i}_Prob"] = prob_sums[i] / total[i]
        metrics_per_epoch.append(row)

    df = pd.DataFrame(metrics_per_epoch)
    df.to_csv("mnist_metrics.csv", index=False)
    return df
