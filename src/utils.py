import os

import torch

from config import Config
from model import SimpleLanguageModel
from tokenizer import BPETokenizer


def get_device() -> torch.device:
    override = os.environ.get("DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: Config, device: torch.device) -> SimpleLanguageModel:
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
    return {
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer": tokenizer.get_state(),
        "rng_state": torch.get_rng_state(),
    }
