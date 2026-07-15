import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(embedding_dim))
        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.gamma * (x / rms)
