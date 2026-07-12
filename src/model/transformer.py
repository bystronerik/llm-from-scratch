import torch
from torch import Tensor

from .attention import MultiHeadAttention
from .layernorm import LayerNorm
from .mlp import FeedForward


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, context_length)
        self.mlp = FeedForward(embedding_dim)
        self.ln_1 = LayerNorm(embedding_dim)
        self.ln_2 = LayerNorm(embedding_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        embeddings = embeddings + self.attention(
            self.ln_1(embeddings)
        )  # (B, T, C) mixed across tokens
        embeddings = embeddings + self.mlp(self.ln_2(embeddings))
        return embeddings
