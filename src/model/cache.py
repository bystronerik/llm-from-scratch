import torch
from torch import Tensor


class KVCache:
    def __init__(self, sink_tokens: int, window_size: int):
        self.sink_tokens = sink_tokens
        self.window_size = window_size
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def capacity(self) -> int:
        return self.sink_tokens + self.window_size

    def __len__(self) -> int:
        return 0 if self.keys is None else self.keys.size(-2)

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = torch.cat([self.keys, k], dim=-2)
            self.values = torch.cat([self.values, v], dim=-2)

        if self.keys.size(-2) > self.capacity:
            self.keys = self._evict(self.keys)
            self.values = self._evict(self.values)

        return self.keys, self.values

    def _evict(self, x: Tensor) -> Tensor:
        sinks = x[..., : self.sink_tokens, :]
        recent = x[..., -self.window_size :, :]
        return torch.cat([sinks, recent], dim=-2)
