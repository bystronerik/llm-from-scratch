import torch


class RMSNorm(torch.nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(embedding_dim))
        self.eps = eps

    def forward(self, x):
        # Root mean square across the embedding dimension. eps inside the sqrt
        # keeps the division stable when a vector is all zeros.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.gamma * (x / rms)
