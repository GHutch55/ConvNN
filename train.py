import os

import torch
import torch.nn as nn
from torch import optim

from models.cnn import CNN
from utils.dataloader import get_train_loader, get_test_loader
from utils.train_utils import train_epoch, evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import init_history, log_epoch
from analysis.plot_metrics import plot_loss, plot_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    checkpoint_path = "checkpoints/best_model.pt"

    model = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load previous best model if checkpoint exists
    if os.path.exists(checkpoint_path):
        model, optimizer = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            device,
        )

        print("Loaded previous best checkpoint.")

    epochs = 2

    history: dict[str, list] = init_history()

    best_acc = 0.0

    for epoch in range(epochs):

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )

        eval_loss, eval_acc = evaluate(model, test_loader, loss_fn, device)

        log_epoch(
            history,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc,
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(model, optimizer, checkpoint_path)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
            f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc * 100:.2f}%"
        )

    plot_loss(history)
    plot_accuracy(history)


if __name__ == "__main__":
    main()
