import glob
import os

import torch

from config import Config
from model import SimpleLanguageModel
from tokenizer import BPETokenizer


def make_checkpoint(
    config: Config,
    tokenizer: BPETokenizer,
    model: SimpleLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> str:
    path = os.path.join(config.checkpoint_dir, f"step_{step}.pt")

    parent = os.path.dirname(os.path.abspath(path))

    os.makedirs(parent, exist_ok=True)

    torch.save(
        {
            "step": step,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer": tokenizer.get_state(),
            "rng_state": torch.get_rng_state(),
        },
        path,
    )

    return path


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def latest_checkpoint(cache_dir: str) -> str | None:
    paths = glob.glob(os.path.join(cache_dir, "checkpoints", "*", "*.pt"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)
