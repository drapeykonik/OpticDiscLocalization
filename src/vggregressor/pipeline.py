from typing import Callable, List

import torch
from torch import nn
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> List[float]:
    """
    Function to perform one epoch of training

    Parameters
    ----------
    model: nn.Module
        Model to train
    criterion: Callable
        Loss function
    optimizer: torch.optim.Optimizer
        Optimizer object to perform model weights
    train_loader: torch.utils.data.DataLoader
        Data loader object for train set samples extraction
    device: str, Optional
        Device to perform train. "cuda" default
    """
    train_losses_epoch = []

    model.train()
    for (images, locations) in tqdm(train_loader):
        optimizer.zero_grad()
        pred_locations = model(images.to(device).float())
        loss = criterion(locations.to(device).float(), pred_locations)
        train_losses_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
    return train_losses_epoch
