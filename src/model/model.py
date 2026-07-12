import torch
from torch import Tensor

from .embedding import Embedding
from .layernorm import LayerNorm
from .transformer import TransformerBlock


class SimpleLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
        num_transformers: int = 4,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(context_length, embedding_dim)
        self.blocks = torch.nn.Sequential(
            *[
                TransformerBlock(embedding_dim, context_length, num_heads)
                for _ in range(num_transformers)
            ]
        )
        self.ln_f = LayerNorm(embedding_dim)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, T = token_ids.shape

        token_emb = self.token_embedding(token_ids)  # (B, T, C)
        positions = torch.arange(T, device=token_ids.device)  # (T,)
        pos_emb = self.position_embedding(positions)  # (T, C)
        embeddings = token_emb + pos_emb  # (B, T, C), broadcast over batch

        embeddings = self.blocks(embeddings)

        embeddings = self.ln_f(embeddings)

        logits = embeddings @ self.token_embedding.weight.T

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
