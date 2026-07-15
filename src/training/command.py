import argparse
import logging
import time

import torch

from dataset import build_batch_samplers, build_wikipedia_corpus
from inference import generate
from training import resume, build, train
from .results import log_result
from utils import get_device

log = logging.getLogger("training")


def train_command(args: argparse.Namespace) -> None:
    torch.manual_seed(42)

    device = get_device()
    log.info("Using device: %s", device)

    run = resume(args.checkpoint, device) if args.checkpoint else build(device)

    data, length = build_wikipedia_corpus(
        run.tokenizer,
        run.config.num_articles,
        run.config.wikipedia_dump,
        run.config.cache_dir,
        run.config.corpus_encode_workers,
    )

    log.info("Corpus: %d tokens, vocab size %d", length, run.tokenizer.vocab_size)

    train_batches, val_batches = build_batch_samplers(
        data,
        length,
        run.config.context_length,
        run.config.batch_size,
        run.config.val_fraction,
        device,
    )

    num_params = sum(p.numel() for p in run.model.parameters())

    log.info("Parameters: %d", num_params)

    initial_loss, final_loss, val_loss, train_seconds = train(
        run, train_batches, val_batches
    )

    prompt = "The "

    prompt_start = time.perf_counter()

    outcome = generate(
        run.model,
        run.tokenizer,
        prompt,
        run.config.max_new_tokens,
        run.config.context_length,
        temperature=run.config.temperature,
        top_k=run.config.top_k,
        top_p=run.config.top_p,
    )

    prompt_seconds = time.perf_counter() - prompt_start

    log.info("Sample after training (%.2fs):\n%s", prompt_seconds, outcome)

    log_result(
        run.config,
        num_params,
        initial_loss,
        final_loss,
        val_loss=val_loss,
        prompt=prompt,
        outcome=outcome,
        device=str(device),
        train_seconds=train_seconds,
        prompt_seconds=prompt_seconds,
    )
