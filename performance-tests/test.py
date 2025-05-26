import torch
import torch.nn as nn
import time
import pandas as pd

# Проста модель
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.linear(x)

# Функція запуску на обраному пристрої з вказаною кількістю потоків
def run_model(device_name: str, num_threads: int = None):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    else:
        torch.set_num_threads(torch.get_num_threads())  # за замовчуванням

    device = torch.device(device_name)
    model = SimpleModel().to(device)
    x = torch.randn(1000, 1000).to(device)

    start = time.time()
    for _ in range(1000):
        output = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    return end - start

# Запуск тестів
results = {}

# Послідовно на CPU (1 потік)
results["CPU (1 thread)"] = run_model("cpu", num_threads=1)

# Паралельно на CPU (макс. потоки)
max_threads = torch.get_num_threads()
results["CPU (multi-thread)"] = run_model("cpu", num_threads=max_threads)

# Паралельно на GPU (якщо доступно)
if torch.cuda.is_available():
    results["GPU"] = run_model("cuda")
else:
    results["GPU"] = None

# Обчислення прискорення
speedups = {}
base_time = results["CPU (1 thread)"]
for key, val in results.items():
    if val is not None:
        speedups[key] = base_time / val
    else:
        speedups[key] = None

# Побудова таблиці
df = pd.DataFrame({
    "Execution Time (s)": results,
    "Speedup vs CPU (1 thread)": speedups
})

# Виведення результату
print(df.to_string(float_format="%.4f"))
