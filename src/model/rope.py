import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_size: int, context_length: int, base: float = 10000.0):
        super().__init__()
        assert head_size % 2 == 0, "head_size must be even for RoPE"

        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))

        positions = torch.arange(context_length).float()
        angles = torch.outer(positions, inv_freq)
        angles = torch.cat([angles, angles], dim=-1)

        self.register_buffer("cos", angles.cos(), persistent=True)
        self.register_buffer("sin", angles.sin(), persistent=True)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos[offset : offset + T].to(x.dtype)
        sin = self.sin[offset : offset + T].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin
