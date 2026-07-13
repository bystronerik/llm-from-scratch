import logging
import os
from dataclasses import dataclass

import torch

from config import Config
from dataset import load_tinystories
from model import SimpleLanguageModel
from tokenizer import BPETokenizer
from utils import build_model

from .checkpoint import load_checkpoint
from .scheduler import CosineWarmupScheduler

log = logging.getLogger("setup")


@dataclass
class RunState:
    """Everything a training run needs, whether started fresh or resumed."""

    config: Config
    model: SimpleLanguageModel
    optimizer: torch.optim.Optimizer
    scheduler: CosineWarmupScheduler
    tokenizer: BPETokenizer
    text: str
    start_step: int


def _build_optimizer_and_scheduler(
    config: Config, model: SimpleLanguageModel
) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
    """AdamW plus the cosine-with-warmup schedule, wired to the config."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
        base_lr=config.learning_rate,
        min_lr=config.min_lr,
    )
    return optimizer, scheduler


def resume_run(path: str, device: torch.device) -> RunState:
    """Rebuild tokenizer, model, optimizer and RNG state from a checkpoint."""
    log.info("Resuming from checkpoint: %s", path)
    ckpt = load_checkpoint(path)
    config = ckpt["config"]
    tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
    text = load_tinystories(config.num_stories)

    model = build_model(config, device)
    model.load_state_dict(ckpt["model_state"])
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    torch.set_rng_state(ckpt["rng_state"])  # continue the same random stream
    start_step = ckpt["step"]
    log.info("Resumed at step %d", start_step)
    return RunState(config, model, optimizer, scheduler, tokenizer, text, start_step)


def build_run(device: torch.device) -> RunState:
    """Start fresh: load the corpus, (re)build the tokenizer and a new model."""
    config = Config()
    text = load_tinystories(config.num_stories)

    if os.path.exists(config.tokenizer_path):
        log.info("Loading cached tokenizer from %s", config.tokenizer_path)
        tokenizer = BPETokenizer.load(config.tokenizer_path)
    else:
        log.info("Training BPE tokenizer (%d merges)...", config.bpe_merges)
        tokenizer = BPETokenizer(text, config.bpe_merges)
        tokenizer.save(config.tokenizer_path)
        log.info("Saved tokenizer to %s", config.tokenizer_path)

    config.vocab_size = tokenizer.vocab_size
    model = build_model(config, device)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)
    return RunState(config, model, optimizer, scheduler, tokenizer, text, start_step=0)
