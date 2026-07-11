import math

import torch
from torch import Tensor

from linear import Linear


class SelfAttention(torch.nn.Module):
    """A single head of causal self-attention.

    Each token looks at every earlier token (and itself) and pulls in a
    weighted mix of their values. The weights come from how well a token's
    query matches another token's key.
    """

    def __init__(self, embedding_dim: int, head_size: int):
        super().__init__()
        self.key = Linear(embedding_dim, head_size)  # what each token offers
        self.query = Linear(embedding_dim, head_size)  # what each token wants
        self.value = Linear(embedding_dim, head_size)  # what each token shares
        self.head_size = head_size

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)  ->  output: (B, T, head_size)
        _, T, _ = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Affinity of every query against every key. Scale by 1/sqrt(head_size)
        # so the softmax stays smooth instead of collapsing onto one token.
        scores = q @ k.transpose(-2, -1)  # (B, T, T)
        scores = scores / math.sqrt(self.head_size)

        # Causal mask: a token may only attend to positions at or before it,
        # so future tokens are set to -inf and vanish after the softmax.
        mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, T, T), rows sum to 1
        return weights @ v  # (B, T, head_size)


class MultiHeadAttention(torch.nn.Module):
    """Several self-attention heads running in parallel.

    Each head attends independently in its own subspace, so different heads
    can specialise on different relationships. Their outputs are concatenated
    and passed through a linear projection to mix them back together.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must divide evenly into num_heads"
        )
        head_size = embedding_dim // num_heads
        self.heads = torch.nn.ModuleList(
            [SelfAttention(embedding_dim, head_size) for _ in range(num_heads)]
        )
        self.proj = Linear(embedding_dim, embedding_dim)  # mix the heads together

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)  ->  output: (B, T, C)
        heads = [head(x) for head in self.heads]  # num_heads x (B, T, head_size)
        out = torch.cat(heads, dim=-1)  # (B, T, C)
        return self.proj(out)  # (B, T, C)
