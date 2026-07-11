import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(embedding_dim))
        self.beta = torch.nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x + self.beta
