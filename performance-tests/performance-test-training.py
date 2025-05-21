import os

# --- Вимикаємо CUDA та GPU-бібліотеки, щоб працювати лише на CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("CUDA_HOME", None)
os.environ.pop("CUDA_PATH", None)
os.environ.pop("CUDA_PATH_V10_2", None)

# Список потоків для бенчмарку
thread_list = [1, 2, 4, 8]

import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cifar10_dataset import FilteredCIFAR10
from models.resnet_CIFAR_18 import ResNet18Custom

def train_and_evaluate(threads, epochs=1, num_classes=10):
    """
    Прогін training+validation, повертає:
    - total_time_s: загальний затрачений час
    - avg_fps: середній fps за епоху
    - best_val_acc: найкраща точність валідації
    """
    # Встановлюємо внутрішній паралелізм PyTorch (intra-op)
    torch.set_num_threads(threads)

    device = torch.device("cpu")
    print(f"\n=== Threads: {threads} | Device: {device} ===")

    # Підготовка датасету та трансформацій
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    train_ds = FilteredCIFAR10(train=True,  transform=transform_train, num_classes=num_classes)
    test_ds  = FilteredCIFAR10(train=False, transform=transform_test,  num_classes=num_classes)

    # DataLoader без надлишкового num_workers
    num_workers = threads
    prefetch = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=False,          # фіксуємо порядок для чистого бенчмарку
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch,
        persistent_workers=True
    )

    # Модель, оптимізатор, лосс і scheduler
    model = ResNet18Custom(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=False
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    fps_per_epoch = []

    # Вимір часу з високою роздільною здатністю
    t0 = time.perf_counter()

    for epoch in range(epochs):
        # --- тренувальний цикл
        model.train()
        start_epoch = time.perf_counter()
        total_train = 0
        for images, labels in tqdm(train_loader, desc=f"[Threads {threads}] Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train += labels.size(0)

        epoch_time = time.perf_counter() - start_epoch
        fps = total_train / epoch_time
        fps_per_epoch.append(fps)

        # --- валідація
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100.0 * correct_val / total_val
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # збереження моделі для подальшого аналізу
            torch.save(model.state_dict(), "best_model.pth")

        print(f" Threads={threads} | Epoch {epoch+1}: FPS={fps:.1f}, Val Acc={val_acc:.2f}%")

    total_time_s = time.perf_counter() - t0
    avg_fps = sum(fps_per_epoch) / len(fps_per_epoch)

    print(f"-> Threads={threads} | Total time: {total_time_s:.1f}s | Avg FPS: {avg_fps:.1f} | Best Val Acc: {best_val_acc:.2f}%")

    return {
        "threads": threads,
        "total_time_s": total_time_s,
        "avg_fps": avg_fps,
        "best_val_acc": best_val_acc
    }


if __name__ == "__main__":
    n_runs = 3  # кількість прогонів для усереднення
    records = []

    for t in thread_list:
        # встановлюємо OMP/MKL перед кожним набором потоків
        os.environ["OMP_NUM_THREADS"] = str(t)
        os.environ["MKL_NUM_THREADS"] = str(t)

        # Warm-up (не вимірюємо час)
        print(f"\n--- Warm-up for {t} threads ---")
        _ = train_and_evaluate(threads=t, epochs=1)

        # Основні прогони для усереднення
        times = []
        fps_list = []
        acc_list = []
        for run in range(n_runs):
            print(f"\n--- Run {run+1}/{n_runs} for {t} threads ---")
            res = train_and_evaluate(threads=t, epochs=1)
            times.append(res["total_time_s"])
            fps_list.append(res["avg_fps"])
            acc_list.append(res["best_val_acc"])

        records.append({
            "threads": t,
            "avg_time_s": sum(times) / n_runs,
            "avg_fps":    sum(fps_list) / n_runs,
            "avg_val_acc":sum(acc_list) / n_runs
        })

    # Записуємо результати в CSV
    df = pd.DataFrame(records)
    df.to_csv("thread_results.csv", index=False)
    print("\nAll averaged results saved to thread_results.csv")
