import logging

import numpy as np
import torch

log = logging.getLogger("dataset")


class BatchSampler:
    def __init__(
        self,
        data: np.memmap,
        lo: int,
        hi: int,
        context_length: int,
        batch_size: int,
        device: torch.device,
    ):
        if hi - lo <= context_length:
            raise ValueError(
                f"split length ({hi - lo}) must be greater than "
                f"context_length ({context_length}) to form at least one window"
            )
        self.data = data
        self.lo = lo
        self.hi = hi
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.context_length

        high = self.hi - T
        starts = torch.randint(self.lo, high, (self.batch_size,))

        offsets = torch.arange(T + 1)
        idx = (starts[:, None] + offsets[None, :]).numpy()

        window = torch.from_numpy(self.data[idx].astype(np.int64))
        window = window.to(self.device)
        inputs = window[:, :-1]
        targets = window[:, 1:]
        return inputs, targets


def build_batch_samplers(
    data: np.memmap,
    length: int,
    context_length: int,
    batch_size: int,
    val_fraction: float,
    device: torch.device,
) -> tuple[BatchSampler, BatchSampler]:
    split = int(length * (1 - val_fraction))

    log.info("Split into %d train / %d val tokens", split, length - split)

    train_batches = BatchSampler(data, 0, split, context_length, batch_size, device)
    val_batches = BatchSampler(data, split, length, context_length, batch_size, device)

    return train_batches, val_batches
