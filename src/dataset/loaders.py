import logging

import torch
from torch.utils.data import DataLoader

from .dataset import SlidingWindowDataset

log = logging.getLogger("dataset")


def build_dataloaders(
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    val_fraction: float,
) -> tuple[DataLoader, DataLoader]:
    """Hold out the tail of ``data`` for validation, then batch both splits."""
    split = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:split], data[split:]
    train_dataset = SlidingWindowDataset(train_data, context_length)
    val_dataset = SlidingWindowDataset(val_data, context_length)
    log.info("Split into %d train / %d val tokens", len(train_data), len(val_data))

    # DataLoader batches windows together and reshuffles every pass.
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return loader, val_loader
