import matplotlib.pyplot as plt

def plot_probabilities(df, num_classes=10):
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        plt.plot(df["Epoch"], df[f"Class{i}_Prob"], label=f"Class {i}")
    plt.title("Average Correct Class Probabilities")
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_avg_probabilities.png")
    plt.show()

def plot_accuracies(df, num_classes=10):
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        plt.plot(df["Epoch"], df[f"Class{i}_Acc"], label=f"Class {i}")
    plt.title("Accuracy Per Class")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_accuracy.png")
    plt.show()
