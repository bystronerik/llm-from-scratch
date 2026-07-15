import torch

from dataset import BatchSampler
from model import SimpleLanguageModel


@torch.no_grad()
def evaluate(
    model: SimpleLanguageModel,
    batches: BatchSampler,
    max_batches: int,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for _ in range(max_batches):
        _, loss = model(*batches.batch())
        total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss / max_batches if max_batches else float("nan")
