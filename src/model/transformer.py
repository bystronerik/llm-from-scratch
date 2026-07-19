import torch
from torch import Tensor

from .attention import MultiHeadAttention
from .cache import KVCache
from .mlp import FeedForward
from .rmsnorm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dim, num_heads, context_length, dropout, rope_base
        )
        self.mlp = FeedForward(embedding_dim, dropout)
        self.ln_1 = RMSNorm(embedding_dim, rms_norm_eps)
        self.ln_2 = RMSNorm(embedding_dim, rms_norm_eps)

    def forward(self, embeddings: Tensor, cache: KVCache | None = None) -> Tensor:
        embeddings = embeddings + self.attention(self.ln_1(embeddings), cache)
        embeddings = embeddings + self.mlp(self.ln_2(embeddings))
        return embeddings
