import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer


# Train model for one full epoch
# Returns:
# - average training loss
# - training accuracy
def train_epoch(
    model: nn.Module,  # CNN model being trained
    dataloader: DataLoader[tuple[Tensor, Tensor]],  # batches of images + labels
    loss_fn: nn.Module,  # loss function (ex: CrossEntropyLoss)
    optimizer: Optimizer,  # updates model weights
    device: torch.device,  # CPU or GPU device
) -> tuple[float, float]:

    total_loss = 0  # sum of batch losses
    correct = 0  # number of correct predictions
    total = 0  # total number of samples

    model.train()  # set model to training mode

    for images, labels in dataloader:
        images = images.to(device)  # move inputs to device
        labels = labels.to(device)  # move labels to device

        optimizer.zero_grad()  # reset gradients

        output = model(images)  # forward pass through model
        predictions = output.argmax(dim=1)  # predicted class per sample

        correct += (predictions == labels).sum().item()  # count correct predictions
        total += labels.size(0)  # batch size

        loss = loss_fn(output, labels)  # compute loss
        total_loss += loss.item()  # accumulate loss

        loss.backward()  # backpropagation
        optimizer.step()  # update model weights

    avg_loss = total_loss / len(dataloader)  # average loss over epoch
    accuracy = correct / total  # accuracy over epoch

    return avg_loss, accuracy


# Evaluate model on validation/test data
# Returns:
# - average loss
# - accuracy
def evaluate(
    model: nn.Module,  # CNN model being evaluated
    dataloader: DataLoader[tuple[Tensor, Tensor]],  # batches of images + labels
    loss_fn: nn.Module,  # loss function (ex: CrossEntropyLoss)
    device: torch.device,  # CPU or GPU device
) -> tuple[float, float]:

    total_loss = 0  # sum of batch losses
    correct = 0  # number of correct predictions
    total = 0  # total number of samples

    model.eval()  # set model to evaluation mode

    with torch.no_grad():  # disable gradient tracking
        for images, labels in dataloader:
            images = images.to(device)  # move inputs to device
            labels = labels.to(device)  # move labels to device

            output = model(images)  # forward pass through model
            predictions = output.argmax(dim=1)  # predicted class per sample

            correct += (predictions == labels).sum().item()  # count correct predictions
            total += labels.size(0)  # batch size

            loss = loss_fn(output, labels)  # compute loss
            total_loss += loss.item()  # accumulate loss

    avg_loss = total_loss / len(dataloader)  # average loss over epoch
    accuracy = correct / total  # accuracy over dataset

    return avg_loss, accuracy


# Run inference on dataset (no labels required)
# Returns:
# - tensor of predicted class indices
def predict(
    model: nn.Module,  # trained CNN model
    dataloader: DataLoader[Tensor],  # batches of images
    device: torch.device,  # CPU or GPU device
) -> Tensor:

    model.eval()  # set model to evaluation mode

    all_predictions = []  # store predictions for entire dataset

    with torch.no_grad():  # disable gradients for inference
        for images in dataloader:
            images = images.to(device)  # move inputs to device

            output = model(images)  # forward pass through model
            predictions = output.argmax(dim=1)  # predicted class per sample

            all_predictions.append(predictions.cpu())  # move to CPU and store

    return torch.cat(all_predictions)  # concatenate all predictions
