import os

import torch

from config import Config
from model import SimpleLanguageModel
from tokenizer import BPETokenizer


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


def build_model(config: Config, device: torch.device) -> SimpleLanguageModel:
    """Construct a model from the config and move it onto `device`."""
    model = SimpleLanguageModel(
        config.vocab_size,
        config.embedding_dim,
        config.context_length,
        config.num_heads,
        config.num_transformers,
        config.dropout,
    )
    return model.to(device)


def make_checkpoint(
    model: SimpleLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Config,
    tokenizer: BPETokenizer,
) -> dict:
    """Bundle everything needed to resume training exactly from this point."""
    return {
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer": tokenizer.get_state(),
        "rng_state": torch.get_rng_state(),
    }
