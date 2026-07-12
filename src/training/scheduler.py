import math

import torch


class CosineWarmupScheduler:
    """Learning-rate schedule: linear warmup, then cosine decay to a floor.

    Steps 1..warmup_steps ramp the LR linearly from 0 up to base_lr, so the
    first weight updates stay small while the optimizer's moment estimates
    settle. After warmup the LR follows a half cosine from base_lr down to
    min_lr across the remaining steps, then holds at min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps < warmup_steps:
            raise ValueError("total_steps must be >= warmup_steps")
        if min_lr > base_lr:
            raise ValueError("min_lr must be <= base_lr")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.last_step = 0  # number of step() calls so far

    def get_lr(self, step: int) -> float:
        """Learning rate for a 1-indexed training step."""
        if step <= self.warmup_steps:
            # Linear warmup: 0 -> base_lr. Guard against warmup_steps == 0.
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * step / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        # Cosine decay: base_lr -> min_lr over the post-warmup steps.
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> float:
        """Advance one step, apply the new LR to the optimizer, and return it."""
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"last_step": self.last_step}

    def load_state_dict(self, state: dict) -> None:
        self.last_step = state["last_step"]
