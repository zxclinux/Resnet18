from training.train_resnet_CIFAR_18 import train_and_evaluate
from plots.visualizations import plot_probabilities, plot_accuracies

if __name__ == "__main__":
    df = train_and_evaluate(epochs=10)
    plot_probabilities(df)
    plot_accuracies(df)
