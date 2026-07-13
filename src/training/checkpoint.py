import os

import torch


def save_checkpoint(checkpoint: dict, path: str) -> None:
    """Write a checkpoint dict to `path`, creating parent dirs as needed.

    The dict is saved with torch.save, so it may hold tensors (state dicts,
    RNG state) alongside plain Python objects (step, config, tokenizer
    state).
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint saved by save_checkpoint().

    Tensors are mapped to CPU so a checkpoint trained on one device (e.g. mps)
    loads cleanly anywhere; the caller moves the model to its device afterward.
    weights_only=False because we store non-tensor objects (tokenizer state).
    """
    return torch.load(path, map_location="cpu", weights_only=False)
