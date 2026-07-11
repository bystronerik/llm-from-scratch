import os

import torch


def get_device() -> torch.device:
    """Pick the best available device: Metal GPU (mps) on Apple Silicon, else CPU.

    Overridable with the DEVICE env var (e.g. DEVICE=cpu) for benchmarking.
    """
    override = os.environ.get("DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
