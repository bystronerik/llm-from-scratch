import logging
import os
import time

import torch

from dataset import BatchSampler
from utils import make_checkpoint

from .checkpoint import save_checkpoint
from .evaluate import evaluate
from .setup import RunState

log = logging.getLogger("train")


def train(
    run: RunState,
    train_batches: BatchSampler,
    val_batches: BatchSampler,
) -> tuple[float, float, float, float]:
    model, optimizer, scheduler = run.model, run.optimizer, run.scheduler
    config, tokenizer = run.config, run.tokenizer

    model.train()
    step = run.start_step
    if step >= config.steps:
        log.info(
            "Checkpoint already at step %d >= steps %d; nothing to train.",
            step,
            config.steps,
        )
    initial_loss = float("nan")
    final_loss = float("nan")
    val_loss = float("nan")
    train_start = time.perf_counter()
    while step < config.steps:
        inputs, targets = train_batches.batch()
        logits, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        lr = scheduler.step(step)

        if (
            step == 1
            or step % 100 == 0
            or step % config.eval_interval == 0
            or step % config.checkpoint_interval == 0
            or step >= config.steps
        ):
            final_loss = loss.item()
            if step == 1:
                initial_loss = final_loss
        if step % 100 == 0:
            log.info(
                "step %d/%d  loss %.4f  lr %.2e", step, config.steps, final_loss, lr
            )
        if step % config.eval_interval == 0 or step >= config.steps:
            val_loss = evaluate(model, val_batches, config.eval_batches)
            log.info(
                "step %d/%d  train loss %.4f  val loss %.4f",
                step,
                config.steps,
                final_loss,
                val_loss,
            )
        if step % config.checkpoint_interval == 0 or step >= config.steps:
            path = os.path.join(config.checkpoint_dir, f"step_{step}.pt")
            save_checkpoint(
                make_checkpoint(model, optimizer, step, config, tokenizer),
                path,
            )
            log.info("Saved checkpoint: %s", path)
    train_seconds = time.perf_counter() - train_start
    log.info("Training took %.2fs (final val loss %.4f)", train_seconds, val_loss)
    return initial_loss, final_loss, val_loss, train_seconds
