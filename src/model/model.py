import torch
from torch import Tensor

from .cache import KVCache
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
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(embedding_dim, context_length, num_heads, dropout)
                for _ in range(num_transformers)
            ]
        )
        self.ln_f = RMSNorm(embedding_dim)

    def new_caches(self, sink_tokens: int, window_size: int) -> list[KVCache]:
        assert sink_tokens + window_size <= self.context_length, (
            "sink_tokens + window_size must fit in context_length; the causal "
            "mask and RoPE tables are only built out to context_length"
        )
        return [KVCache(sink_tokens, window_size) for _ in self.blocks]

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        embeddings = self.token_embedding(token_ids)
        embeddings = self.embedding_dropout(embeddings)

        for i, block in enumerate(self.blocks):
            embeddings = block(embeddings, caches[i] if caches is not None else None)

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
