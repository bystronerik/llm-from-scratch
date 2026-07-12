import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias
