import torch

from embedding import Embedding
from linear import Linear


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.head = Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)  -> a batch of B sequences, each T tokens long
        embeddings = self.embedding(token_ids)  # (B, T, C)  look up a vector per token
        logits = self.head(
            embeddings
        )  # (B, T, vocab_size)  a score per possible next token
        return logits
