import logging
import os
import time
from dataclasses import dataclass

import torch

from config import Config
from dataset import load_alpaca_corpus, load_wikipedia_text
from model import SimpleLanguageModel
from tokenizer import BPETokenizer
from utils import build_model

from .checkpoint import load_checkpoint
from .scheduler import CosineWarmupScheduler

log = logging.getLogger("setup")


@dataclass
class RunState:
    config: Config
    model: SimpleLanguageModel
    optimizer: torch.optim.Optimizer
    scheduler: CosineWarmupScheduler
    tokenizer: BPETokenizer
    start_step: int


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


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


def resume_run(path: str, device: torch.device) -> RunState:
    log.info("Resuming from checkpoint: %s", path)
    ckpt = load_checkpoint(path)
    config = ckpt["config"]
    config.run_id = _new_run_id()
    tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])

    model = build_model(config, device)
    model.load_state_dict(ckpt["model_state"])
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    torch.set_rng_state(ckpt["rng_state"])
    start_step = ckpt["step"]
    log.info("Resumed at step %d", start_step)
    return RunState(config, model, optimizer, scheduler, tokenizer, start_step)


def build_run(device: torch.device) -> RunState:
    config = Config(run_id=_new_run_id())

    if os.path.exists(config.tokenizer_path):
        log.info("Loading cached tokenizer from %s", config.tokenizer_path)
        tokenizer = BPETokenizer.load(config.tokenizer_path)
    else:
        wiki = load_wikipedia_text(
            config.tokenizer_num_articles,
            config.wikipedia_dump,
            config.wikipedia_path,
            config.cache_dir,
        )
        alpaca = load_alpaca_corpus(
            config.tokenizer_alpaca_examples, config.cache_dir, config.alpaca_path
        )
        log.info(
            "Training BPE tokenizer (%d merges) on Wikipedia + Alpaca...",
            config.bpe_merges,
        )
        tokenizer = BPETokenizer.train(wiki + "\n" + alpaca, config.bpe_merges)

        tokenizer.save(config.tokenizer_path)
        log.info("Saved tokenizer to %s", config.tokenizer_path)

    config.vocab_size = tokenizer.vocab_size
    model = build_model(config, device)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)
    return RunState(config, model, optimizer, scheduler, tokenizer, start_step=0)
