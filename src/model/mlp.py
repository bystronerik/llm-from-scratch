import torch
from torch import Tensor

from .linear import Linear


class FeedForward(torch.nn.Module):
    """SwiGLU feed-forward network (Llama/PaLM-style).

    A plain transformer MLP is ``Linear -> ReLU -> Linear``: project up to a
    wider hidden space, apply a pointwise non-linearity, project back down.

    SwiGLU replaces the single non-linear path with a *gate*. The input is
    projected twice in parallel:
      - ``gate`` is squashed through SiLU (a smooth, differentiable ReLU:
        ``x * sigmoid(x)``);
      - ``up`` is left linear.
    Their elementwise product lets the network learn, per hidden unit, how much
    of the ``up`` signal to let through -- a data-dependent gate rather than a
    fixed ReLU threshold. This consistently trains a little better.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.0):
        super().__init__()

        # 4 * dim scaled by 2/3 (two input projections), rounded to a multiple
        # of 8 so the matmuls stay hardware-friendly.
        hidden_dim = int(embedding_dim * 4 * 2 / 3)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)

        self.gate = Linear(embedding_dim, hidden_dim)  # becomes the SiLU gate
        self.up = Linear(embedding_dim, hidden_dim)  # the value being gated
        self.down = Linear(hidden_dim, embedding_dim)  # project back down
        self.dropout = torch.nn.Dropout(dropout)  # regularise the block's output

    def forward(self, x: Tensor) -> Tensor:
        gated = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(gated))
