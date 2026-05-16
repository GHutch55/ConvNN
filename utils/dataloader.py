from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Data preprocessing (applied to every image)
transformation = transforms.Compose(
    [
        # Randomly crop image after padding
        # (adds small shifts so model generalizes better)
        transforms.RandomCrop(32, padding=4),
        # Randomly flip image left-right (50% chance)
        transforms.RandomHorizontalFlip(0.5),
        # Convert image to tensor
        # (PIL image → PyTorch tensor, scales to [0, 1])
        transforms.ToTensor(),
        # Normalize pixel values per channel
        # (helps training stability)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load CIFAR-10 training data
train_data = datasets.CIFAR10(
    root="../data/train",
    train=True,
    download=True,  # download if needed
    transform=transformation,  # apply preprocessing above
)

# Create batches for training
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,  # number of images per batch
    shuffle=True,  # shuffle each epoch
)
