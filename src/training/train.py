import logging
import time

import torch

from dataset import BatchSampler

from .checkpoint import make_checkpoint
from .evaluate import evaluate
from .setup import TrainingState

log = logging.getLogger("train")


def train(
    state: TrainingState,
    train_batches: BatchSampler,
    val_batches: BatchSampler,
) -> tuple[float, float, float, float, int, bool]:
    config, tokenizer = state.config, state.tokenizer
    model, optimizer, scheduler = state.model, state.optimizer, state.scheduler
    master = state.master

    model.train()

    step = state.start_step
    if step >= config.steps:
        log.info(
            "Checkpoint already at step %d >= steps %d; nothing to train.",
            step,
            config.steps,
        )

    initial_loss = float("nan")
    final_loss = float("nan")
    val_loss = float("nan")
    interrupted = False

    train_start = time.perf_counter()

    try:
        while step < config.steps:
            inputs, targets = train_batches.batch()

            logits, loss = model(inputs, targets)

            loss.backward()

            master.copy_grads()

            torch.nn.utils.clip_grad_norm_(master.parameters(), max_norm=1.0)
            optimizer.step()

            master.copy_to_model()

            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

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
                path = make_checkpoint(
                    config, tokenizer, model, master, optimizer, step
                )
                log.info("Saved checkpoint: %s", path)
    except KeyboardInterrupt:
        interrupted = True
        log.warning(
            "Interrupted at step %d/%d; shutting down gracefully.", step, config.steps
        )
        if step > 0:
            final_loss = loss.item()
            val_loss = evaluate(model, val_batches, config.eval_batches)
            path = make_checkpoint(config, tokenizer, model, master, optimizer, step)
            log.info("Saved interrupt checkpoint: %s", path)

    train_seconds = time.perf_counter() - train_start

    log.info("Training took %.2fs (final val loss %.4f)", train_seconds, val_loss)

    return initial_loss, final_loss, val_loss, train_seconds, step, interrupted
