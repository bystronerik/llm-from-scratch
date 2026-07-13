import torch
from torch.utils.data import DataLoader

from model import SimpleLanguageModel


@torch.no_grad()
def evaluate(
    model: SimpleLanguageModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """Average the loss over up to ``max_batches`` batches, without training."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        batches += 1
        if batches >= max_batches:
            break
    if was_training:
        model.train()
    return total_loss / batches if batches else float("nan")
