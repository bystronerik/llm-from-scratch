import argparse
import logging
import time

import torch

from inference import generate
from tokenizer import EOS_TOKEN
from training import latest_checkpoint, resume
from utils import get_device

log = logging.getLogger("inference")


def run_command(args: argparse.Namespace) -> None:
    torch.manual_seed(42)

    device = get_device()

    log.info("Using device: %s", device)

    checkpoint = args.checkpoint or latest_checkpoint(".cache")
    if checkpoint is None:
        raise SystemExit(
            "No checkpoint found under .cache/checkpoints. "
            "Train first (./llm train) or pass --checkpoint PATH."
        )

    state = resume(checkpoint, device)

    config = state.config

    max_new_tokens = args.max_new_tokens or config.max_new_tokens

    temperature = config.temperature if args.temperature is None else args.temperature
    top_k = config.top_k if args.top_k is None else args.top_k
    top_p = config.top_p if args.top_p is None else args.top_p

    prompt_start = time.perf_counter()

    outcome = generate(
        state.model,
        state.tokenizer,
        args.prompt,
        max_new_tokens,
        config.context_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_id=state.tokenizer.special_token_id(EOS_TOKEN),
        sink_tokens=config.sink_tokens,
        window_size=config.kv_window or None,
    )

    prompt_seconds = time.perf_counter() - prompt_start

    log.info("Generated in %.2fs:", prompt_seconds)

    print(outcome)
