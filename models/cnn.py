import torch
import torch.nn as nn


class CNN(nn.Module):

    # Initialize Module and define the Convolutional and Fully Connected Layers
    def __init__(self):
        super().__init__()

        self.conv: nn.Sequential = nn.Sequential(
            # Each Block will have 2 Convolutional Layers + Activation Functions (ReLU)
            # MaxPooling at the end for downsampling and Dropping for regularization
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Cutting size from 32x32 to 16x16
            nn.Dropout2d(p=0.2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Cutting size from 16x16 to 8x8
            nn.Dropout2d(p=0.2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Cutting size from 8x8 to 4x4
            nn.Dropout2d(p=0.2),
        )

        self.fullyconn: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # Pass a Tensor through the layers to transform into predictions
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fullyconn(x)
        return x
