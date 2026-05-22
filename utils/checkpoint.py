import torch
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, Optimizer]:

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer
