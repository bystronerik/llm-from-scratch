import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.02
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
