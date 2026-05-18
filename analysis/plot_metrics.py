import matplotlib.pyplot as plt


def plot_loss(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")

    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["eval_loss"], label="Evaluation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig("loss_plot.png")
    plt.show()


def plot_accuracy(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Evaluation Accuracy")

    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["eval_acc"], label="Evaluation Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig("acc_plot.png")
    plt.show()
