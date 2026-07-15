import logging
import os
import time

import torch

from config import Config
from dataset import load_alpaca_text, load_wikipedia_text
from model import SimpleLanguageModel
from tokenizer import BPETokenizer
from .checkpoint import load_checkpoint
from .scheduler import CosineWarmupScheduler
from .state import TrainingState

log = logging.getLogger("setup")


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _build_model(config: Config, device: torch.device) -> SimpleLanguageModel:
    model = SimpleLanguageModel(
        config.vocab_size,
        config.embedding_dim,
        config.context_length,
        config.num_heads,
        config.num_transformers,
        config.dropout,
    )
    return model.to(device)


def _build_optimizer_and_scheduler(
    config: Config, model: SimpleLanguageModel
) -> tuple[torch.optim.Optimizer, CosineWarmupScheduler]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
        base_lr=config.learning_rate,
        min_lr=config.min_lr,
    )
    return optimizer, scheduler


def resume(path: str, device: torch.device) -> TrainingState:
    log.info("Resuming from checkpoint: %s", path)

    checkpoint = load_checkpoint(path)

    checkpoint_config = checkpoint["config"]
    config = Config(run_id=_new_run_id())

    config.vocab_size = checkpoint_config.vocab_size
    config.embedding_dim = checkpoint_config.embedding_dim
    config.context_length = checkpoint_config.context_length
    config.batch_size = checkpoint_config.batch_size
    config.num_heads = checkpoint_config.num_heads
    config.num_transformers = checkpoint_config.num_transformers

    tokenizer = BPETokenizer.from_state(checkpoint["tokenizer"])

    model = _build_model(config, device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    torch.set_rng_state(checkpoint["rng_state"])

    step = checkpoint["step"]

    log.info("Resumed at step %d", step)

    return TrainingState(config, model, optimizer, scheduler, tokenizer, step)


def build(device: torch.device) -> TrainingState:
    config = Config(run_id=_new_run_id())

    if os.path.exists(config.tokenizer_path):
        log.info("Loading cached tokenizer from %s", config.tokenizer_path)
        tokenizer = BPETokenizer.load(config.tokenizer_path)
    else:
        wiki = load_wikipedia_text(
            config.num_articles, config.cache_dir, config.wikipedia_path
        )
        alpaca = load_alpaca_text(config.cache_dir, config.alpaca_path)

        log.info(
            "Training BPE tokenizer (%d merges)...",
            config.bpe_merges,
        )

        tokenizer = BPETokenizer.train(wiki + "\n" + alpaca, config.bpe_merges)
        tokenizer.save(config.tokenizer_path)

        log.info("Saved tokenizer to %s", config.tokenizer_path)

    config.vocab_size = tokenizer.vocab_size

    model = _build_model(config, device)

    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    return TrainingState(config, model, optimizer, scheduler, tokenizer, start_step=0)
