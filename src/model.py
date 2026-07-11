import torch
from torch import Tensor

from attention import SelfAttention
from embedding import Embedding
from linear import Linear


class SimpleLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(context_length, embedding_dim)
        self.attention = SelfAttention(embedding_dim, embedding_dim)
        self.head = Linear(embedding_dim, vocab_size)

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

        embeddings = self.attention(embeddings)  # (B, T, C) mixed across tokens
        logits = self.head(embeddings)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
