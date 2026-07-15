import math

import torch


class CosineWarmupScheduler:
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

    def get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            if self.warmup_steps == 0:
                return self.base_lr
            return self.base_lr * step / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self, step: int) -> float:
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
