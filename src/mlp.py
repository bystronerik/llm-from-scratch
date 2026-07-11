import torch

from linear import Linear


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.net = torch.nn.Sequential(
            Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
