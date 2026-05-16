from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# Data preprocessing applied to training images
def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # random spatial shift
            transforms.RandomHorizontalFlip(0.5),  # random flip
            transforms.ToTensor(),  # convert image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize
        ]
    )


# Preprocessing for test/validation (no augmentation)
def get_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


# Create training dataloader
def get_train_loader(
    root: str = "../data",
    batch_size: int = 64,
    download: bool = True,
) -> DataLoader:

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=get_train_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


# Create test/validation dataloader
def get_test_loader(
    root: str = "../data",
    batch_size: int = 64,
    download: bool = True,
) -> DataLoader:

    dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=get_test_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
