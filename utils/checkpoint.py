import torch
import torch.nn as nn
from pathlib import Path


def save_checkpoint(model: nn.Module, path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module, path: str, device: torch.device = torch.device("cpu")
) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
