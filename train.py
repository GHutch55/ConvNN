import torch
import torch.nn as nn
from torch import optim

from models.cnn import CNN
from utils.train_utils import train_epoch, evaluate, predict
from utils.dataloader import get_train_loader, get_test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_test_loader()
    train_loader = get_train_loader()

    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 10

    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        eval_loss, eval_acc = evaluate(model, test_loader, loss_fn, device)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
