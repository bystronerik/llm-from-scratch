import os

import torch


def get_device() -> torch.device:
    override = os.environ.get("DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
