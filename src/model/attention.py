import math

import torch
from torch import Tensor

from .linear import Linear
from .rope import RotaryEmbedding


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must divide evenly into num_heads"
        )
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

        self.key = Linear(embedding_dim, embedding_dim)
        self.query = Linear(embedding_dim, embedding_dim)
        self.value = Linear(embedding_dim, embedding_dim)
        self.proj = Linear(embedding_dim, embedding_dim)

        self.rotary = RotaryEmbedding(self.head_size, context_length)

        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_size)
        return x.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape

        k = self._split_heads(self.key(x))
        q = self._split_heads(self.query(x))
        v = self._split_heads(self.value(x))

        q = self.rotary(q)
        k = self.rotary(k)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v

        B = out.size(0)

        out = out.transpose(1, 2).reshape(B, T, -1)

        return self.resid_dropout(self.proj(out))
