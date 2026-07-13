import logging

import torch

log = logging.getLogger("dataset")


class BatchSampler:
    """Draws random training windows straight from a device-resident corpus.

    The whole corpus fits in memory, so we copy it onto the device once and
    then build each step's batch with a single ``torch.randint`` of start
    offsets plus a gather. This avoids the per-step CPU->GPU transfer, the
    DataLoader iteration overhead, and the full-length shuffle permutation a
    map-style DataLoader rebuilds every epoch.
    """

    def __init__(
        self,
        data: torch.Tensor,
        context_length: int,
        batch_size: int,
        device: torch.device,
    ):
        if len(data) <= context_length:
            raise ValueError(
                f"data length ({len(data)}) must be greater than "
                f"context_length ({context_length}) to form at least one window"
            )
        self.data = data.to(device)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` random (input, target) windows, on device."""
        T = self.context_length
        # Highest start index that still leaves room for a window and its
        # one-step-shifted target: we read data[i : i+T] and data[i+1 : i+T+1].
        high = len(self.data) - T
        starts = torch.randint(high, (self.batch_size,), device=self.device)  # (B,)

        # Turn each start offset into a full row of positions, so one gather
        # pulls every window at once instead of looping in Python.
        offsets = torch.arange(T, device=self.device)  # (T,)
        idx = starts[:, None] + offsets[None, :]  # (B, T)
        inputs = self.data[idx]  # (B, T)
        targets = self.data[idx + 1]  # (B, T), each target shifted one ahead
        return inputs, targets


def build_batch_samplers(
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    val_fraction: float,
    device: torch.device,
) -> tuple[BatchSampler, BatchSampler]:
    """Hold out the tail of ``data`` for validation, then wrap both splits."""
    split = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:split], data[split:]
    log.info("Split into %d train / %d val tokens", len(train_data), len(val_data))
    train_batches = BatchSampler(train_data, context_length, batch_size, device)
    val_batches = BatchSampler(val_data, context_length, batch_size, device)
    return train_batches, val_batches
