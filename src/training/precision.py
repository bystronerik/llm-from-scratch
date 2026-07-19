import torch
from torch import Tensor


class MasterWeights:
    def __init__(
        self,
        model: torch.nn.Module,
        state: dict[str, Tensor] | None = None,
    ):
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]

        source = self.params if state is None else [state[n] for n in self.names]
        self.master = [
            t.detach().to(dtype=torch.float32, device=self.params[0].device).clone()
            for t in source
        ]

        if state is not None:
            self.copy_to_model()

    def parameters(self) -> list[Tensor]:
        return self.master

    def copy_grads(self) -> None:
        for p, mp in zip(self.params, self.master):
            if p.grad is None:
                mp.grad = None
            elif mp.grad is None:
                mp.grad = p.grad.to(torch.float32)
            else:
                mp.grad.copy_(p.grad)

    def copy_to_model(self) -> None:
        with torch.no_grad():
            for p, mp in zip(self.params, self.master):
                p.copy_(mp)

    def state_dict(self) -> dict[str, Tensor]:
        return {n: mp.detach().clone() for n, mp in zip(self.names, self.master)}
