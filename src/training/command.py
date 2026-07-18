import argparse
import logging
import time

import numpy as np
import torch

from dataset import build_batch_samplers, load_wikipedia_text
from inference import generate
from training import resume, build, train
from .results import log_result
from utils import get_device

log = logging.getLogger("training")


def train_command(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)

    device = get_device()
    log.info("Using device: %s", device)

    state = resume(args.checkpoint, device) if args.checkpoint else build(device)

    text = load_wikipedia_text(
        state.config.num_articles,
        state.config.cache_dir,
        state.config.wikipedia_path,
    )

    data = np.asarray(state.tokenizer.encode(text), dtype=np.uint16)
    length = len(data)

    log.info("Corpus: %d tokens, vocab size %d", length, state.tokenizer.vocab_size)

    train_batches, val_batches = build_batch_samplers(
        data,
        length,
        state.config.context_length,
        state.config.batch_size,
        state.config.val_fraction,
        device,
    )

    num_params = sum(p.numel() for p in state.model.parameters())

    log.info("Parameters: %d", num_params)

    initial_loss, final_loss, val_loss, train_seconds, step, interrupted = train(
        state, train_batches, val_batches
    )

    prompt = "The "
    outcome = ""
    prompt_seconds = float("nan")

    try:
        prompt_start = time.perf_counter()

        outcome = generate(
            state.model,
            state.tokenizer,
            prompt,
            state.config.max_new_tokens,
            state.config.context_length,
            temperature=state.config.temperature,
            top_k=state.config.top_k,
            top_p=state.config.top_p,
        )

        prompt_seconds = time.perf_counter() - prompt_start

        log.info("Sample after training (%.2fs):\n%s", prompt_seconds, outcome)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted during sample generation; skipping sample.")

    notes = f"INTERRUPTED at step {step}/{state.config.steps}" if interrupted else ""

    log_result(
        state.config,
        num_params,
        initial_loss,
        final_loss,
        val_loss=val_loss,
        prompt=prompt,
        outcome=outcome,
        device=str(device),
        train_seconds=train_seconds,
        prompt_seconds=prompt_seconds,
        notes=notes,
    )
