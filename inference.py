import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.cnn import CNN
from utils.checkpoint import load_checkpoint


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
) -> str:

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(x)

        pred = outputs.argmax(dim=1).item()

    return CLASSES[pred]


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
    )

    model, optimizer = load_checkpoint(
        model,
        optimizer,
        "checkpoints/best_model.pt",
        device,
    )

    model = model.to(device)

    model.eval()

    os.makedirs(
        "saved_images",
        exist_ok=True,
    )

    dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
    )

    indices = random.sample(
        range(len(dataset)),
        10,
    )

    for i, idx in enumerate(indices):

        image, label = dataset[idx]

        true_label = CLASSES[label]

        image_path = f"saved_images/{i}_{true_label}.png"

        image.save(image_path)

        pred_label = predict_image(
            model,
            image_path,
            device,
        )

        print(f"[{i}] " f"true={true_label} " f"pred={pred_label}")


if __name__ == "__main__":
    main()
