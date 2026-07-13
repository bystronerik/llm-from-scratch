import torch
from torch import Tensor

from .attention import MultiHeadAttention
from .mlp import FeedForward
from .rmsnorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dim, num_heads, context_length, dropout
        )
        self.mlp = FeedForward(embedding_dim, dropout)
        self.ln_1 = RMSNorm(embedding_dim)
        self.ln_2 = RMSNorm(embedding_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        embeddings = embeddings + self.attention(
            self.ln_1(embeddings)
        )  # (B, T, C) mixed across tokens
        embeddings = embeddings + self.mlp(self.ln_2(embeddings))
        return embeddings
