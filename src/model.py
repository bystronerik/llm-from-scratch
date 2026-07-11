import torch
from torch import Tensor

from attention import MultiHeadAttention
from embedding import Embedding
from layernorm import LayerNorm
from linear import Linear
from mlp import FeedForward


class SimpleLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(context_length, embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.head = Linear(embedding_dim, vocab_size)
        self.mlp = FeedForward(embedding_dim)
        self.ln_1 = LayerNorm(embedding_dim)
        self.ln_2 = LayerNorm(embedding_dim)
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

        embeddings = embeddings + self.attention(
            self.ln_1(embeddings)
        )  # (B, T, C) mixed across tokens
        embeddings = embeddings + self.mlp(self.ln_2(embeddings))

        embeddings = self.ln_f(embeddings)

        logits = self.head(embeddings)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
