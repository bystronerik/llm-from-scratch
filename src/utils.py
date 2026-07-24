import os

import torch


def pipeline_mode() -> bool:
    return os.getenv("PIPELINE", "").lower() in {"true", "1"}


def get_device() -> torch.device:
    override = os.environ.get("DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
