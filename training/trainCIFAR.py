from datasets.cifar10_dataset import FilteredCIFAR10
from models.resnetCIFAR_custom import ResNet18Custom
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, time
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

def train_and_evaluate(epochs=100, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовується: {device}")

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]  # ImageNet std
        )
    ])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = ResNet18Custom(num_classes=num_classes, pretrained=True, freeze_backbone=False).to(device)
    train_dataset = FilteredCIFAR10(train=True, transform=transform_train, num_classes=num_classes)
    test_dataset = FilteredCIFAR10(train=False, transform=transform_test, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics_per_epoch = []
    total_samples = len(train_dataset)
    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_start = time.time()

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        fps = total_train / (time.time() - epoch_start)

        # Evaluation
        model.eval()
        correct = [0] * num_classes
        total = [0] * num_classes
        prob_sums = [0.0] * num_classes
        correct_val_total = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                correct_val_total += (preds == labels).sum().item()
                val_total += labels.size(0)

                for i in range(images.size(0)):
                    true = labels[i].item()
                    pred = preds[i].item()
                    prob = probs[i][true].item()
                    total[true] += 1
                    prob_sums[true] += prob
                    if pred == true:
                        correct[true] += 1

        val_acc = 100.0 * correct_val_total / val_total
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, FPS={fps:.2f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"📦 Збережено найкращу модель (val acc = {best_val_acc:.2f}%)")

        row = {"Epoch": epoch + 1}
        for i in range(num_classes):
            row[f"Class{i}_Acc"] = correct[i] / total[i]
            row[f"Class{i}_Prob"] = prob_sums[i] / total[i]
        metrics_per_epoch.append(row)

    total_time = time.time() - start_time
    overall_fps = (total_samples * epochs) / total_time
    print(f"\n✅ Повний час тренування: {total_time:.2f} секунд")
    print(f"🚀 Середній FPS: {overall_fps:.2f} зображень/секунда")

    df = pd.DataFrame(metrics_per_epoch)
    df.to_csv("cifar10_metrics.csv", index=False)
    return df
