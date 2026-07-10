import torch
from torch import Tensor

from embedding import Embedding
from linear import Linear


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.head = Linear(embedding_dim, vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        embeddings = self.embedding(token_ids)
        logits = self.head(embeddings)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
