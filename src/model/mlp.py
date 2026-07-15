import torch
from torch import Tensor

from .linear import Linear


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0):
        super().__init__()

        hidden_dim = int(embedding_dim * 4 * 2 / 3)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)

        self.gate = Linear(embedding_dim, hidden_dim)
        self.up = Linear(embedding_dim, hidden_dim)
        self.down = Linear(hidden_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gated = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(gated))
