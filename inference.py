import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models.cnn import CNN
from utils.dataloader import get_test_loader
from utils.checkpoint import load_checkpoint


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> Tensor:
    model = model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())

    return torch.cat(all_preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model = load_checkpoint(model, "checkpoints/best_model.pt", device)

    test_loader = get_test_loader()

    preds = predict(model, test_loader, device)

    print("Predictions:", preds)
    print("Total samples:", len(preds))


if __name__ == "__main__":
    main()
