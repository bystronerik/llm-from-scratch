import math

import torch
from torch import Tensor

from .linear import Linear


class MultiHeadAttention(torch.nn.Module):
    """Several causal self-attention heads running in parallel.

    Each token looks at every earlier token (and itself) and pulls in a
    weighted mix of their values. The weights come from how well a token's
    query matches another token's key. Each head attends independently in its
    own subspace, so different heads can specialise on different relationships.

    Rather than looping over heads, all heads share one Q/K/V projection and
    are computed together as an extra batch dimension: the maths is identical
    to independent heads, but it runs as a handful of large matmuls instead of
    many small ones.
    """

    def __init__(self, embedding_dim: int, num_heads: int, context_length: int):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must divide evenly into num_heads"
        )
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

        # One projection each for keys/queries/values, covering every head at
        # once. Splitting the output into `num_heads` chunks recovers the
        # per-head projections.
        self.key = Linear(embedding_dim, embedding_dim)  # what each token offers
        self.query = Linear(embedding_dim, embedding_dim)  # what each token wants
        self.value = Linear(embedding_dim, embedding_dim)  # what each token shares
        self.proj = Linear(embedding_dim, embedding_dim)  # mix the heads together

        # The causal mask is the same every forward, so build it once here and
        # slice it to the current sequence length. Registered as a buffer so it
        # follows the model onto the GPU but is not a learnable parameter.
        self.register_buffer(
            "mask", torch.tril(torch.ones(context_length, context_length))
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, T, C) -> (B, num_heads, T, head_size), so heads ride alongside
        # the batch dimension and every matmul below covers all heads at once.
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_size)
        return x.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)  ->  output: (B, T, C)
        _, T, _ = x.shape

        k = self._split_heads(self.key(x))  # (B, num_heads, T, head_size)
        q = self._split_heads(self.query(x))  # (B, num_heads, T, head_size)
        v = self._split_heads(self.value(x))  # (B, num_heads, T, head_size)

        # Affinity of every query against every key, per head. Scale by
        # 1/sqrt(head_size) so the softmax stays smooth instead of collapsing
        # onto one token.
        scores = q @ k.transpose(-2, -1)  # (B, num_heads, T, T)
        scores = scores / math.sqrt(self.head_size)

        # Causal mask: a token may only attend to positions at or before it,
        # so future tokens are set to -inf and vanish after the softmax.
        mask = self.mask[:T, :T]  # (T, T), sliced from the prebuilt buffer
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        out = weights @ v  # (B, num_heads, T, head_size)

        # Concatenate the heads back into one vector per token, then mix.
        B = out.size(0)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, C)
        return self.proj(out)  # (B, T, C)
