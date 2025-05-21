import matplotlib.pyplot as plt
import pandas as pd

# Переносимо повну побудову графіків у функцію, яка читає з файлу
def plot_thread_results_from_csv(filepath):
    df = pd.read_csv(filepath)

    # Обчислення прискорення та ефективності
    baseline_time = df[df['threads'] == 'CPU-1']['time_s'].values[0]
    df['speedup'] = baseline_time / df['time_s']

    def extract_threads(t):
        return 1 if 'GPU' in t else int(t.split('-')[1])

    df['threads_num'] = df['threads'].apply(extract_threads)
    df['efficiency'] = df['speedup'] / df['threads_num']

    # Графік часу виконання
    plt.figure(figsize=(10, 5))
    plt.bar(df['threads'], df['time_s'], color='skyblue')
    plt.title('Час тренування (секунди)')
    plt.xlabel('Конфігурація')
    plt.ylabel('Час, с')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Графік FPS
    plt.figure(figsize=(10, 5))
    plt.bar(df['threads'], df['fps'], color='limegreen')
    plt.title('Швидкодія (FPS)')
    plt.xlabel('Конфігурація')
    plt.ylabel('Кадрів за секунду')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Графік прискорення
    plt.figure(figsize=(10, 5))
    plt.bar(df['threads'], df['speedup'], color='orange')
    plt.title('Прискорення (відносно CPU-1)')
    plt.xlabel('Конфігурація')
    plt.ylabel('Прискорення')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Графік ефективності
    plt.figure(figsize=(10, 5))
    plt.bar(df['threads'], df['efficiency'], color='purple')
    plt.title('Ефективність (прискорення / кількість потоків)')
    plt.xlabel('Конфігурація')
    plt.ylabel('Ефективність')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Викликаємо функцію для побудови всіх графіків з файлу
plot_thread_results_from_csv('performance-tests/thread_results.csv')