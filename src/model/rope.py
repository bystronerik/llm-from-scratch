import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings (RoPE)

    Instead of *adding* a learned position vector to each token (the classic
    approach, which fixes a maximum length and encodes position absolutely),
    RoPE *rotates* the query and key vectors by an angle that grows with the
    token's position.

    This module precomputes the per-position cos/sin tables once and applies
    them to a (B, num_heads, T, head_size) tensor.
    """

    def __init__(self, head_size: int, context_length: int, base: float = 10000.0):
        super().__init__()
        assert head_size % 2 == 0, "head_size must be even for RoPE"

        # One rotation frequency per feature *pair*. Pair k rotates at
        # base**(-2k/head_size): the first pairs turn slowly (long wavelength),
        # later pairs turn fast (short wavelength).
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )  # (head_size/2,)

        # Angle for every (position, pair): outer product of positions and
        # frequencies. Duplicate along the last axis so the table lines up with
        # the two halves that rotate_half pairs together below.
        positions = torch.arange(context_length).float()  # (T,)
        angles = torch.outer(positions, inv_freq)  # (T, head_size/2)
        angles = torch.cat([angles, angles], dim=-1)  # (T, head_size)

        # persistent=False: derived from config, not learned, kept out of checkpoints.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        # Split the feature dim in half and swap the halves with a sign flip:
        # this realises the 90-degree part of each pair's 2-D rotation so that
        #   rotated = x * cos + rotate_half(x) * sin
        # is exactly a rotation of each (x_i, x_{i+d/2}) pair.
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, num_heads, T, head_size). cos/sin slice to T and broadcast
        # across the batch and head dimensions.
        T = x.shape[-2]
        cos = self.cos[:T]  # (T, head_size)
        sin = self.sin[:T]  # (T, head_size)
        return x * cos + self._rotate_half(x) * sin
