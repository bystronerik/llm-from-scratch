import torch
from torch import Tensor

from .embedding import Embedding
from .rmsnorm import RMSNorm
from .transformer import TransformerBlock


class SimpleLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
        num_transformers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.blocks = torch.nn.Sequential(
            *[
                TransformerBlock(embedding_dim, context_length, num_heads, dropout)
                for _ in range(num_transformers)
            ]
        )
        self.ln_f = RMSNorm(embedding_dim)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        embeddings = self.token_embedding(token_ids)  # (B, T, C)
        embeddings = self.embedding_dropout(embeddings)

        embeddings = self.blocks(embeddings)

        embeddings = self.ln_f(embeddings)

        logits = embeddings @ self.token_embedding.weight.T

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
