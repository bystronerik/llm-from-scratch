import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import SlidingWindowDataset, load_tinystories
from utils import get_device, build_model, make_checkpoint
from model import SimpleLanguageModel
from results import log_result
from tokenizer import Tokenizer, BPETokenizer
from training.checkpoint import save_checkpoint, load_checkpoint
from training import CosineWarmupScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")


def generate(
    model: SimpleLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
) -> str:
    """Feed the prompt in, then repeatedly sample one next character."""
    was_training = model.training
    model.eval()  # disable dropout while sampling
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-context_length:].unsqueeze(0)  # (1, T)
            logits, _ = model(context)  # (1, T, vocab)
            last_step = logits[0, -1]  # (vocab,) scores for the next char
            probs = torch.softmax(last_step, dim=-1)  # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample one
            ids = torch.cat([ids, next_id])
    if was_training:
        model.train()
    return tokenizer.decode(ids.tolist())


@torch.no_grad()
def evaluate(
    model: SimpleLanguageModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """Average the loss over up to ``max_batches`` batches, without training."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        batches += 1
        if batches >= max_batches:
            break
    if was_training:
        model.train()
    return total_loss / batches if batches else float("nan")


def train(
    model: SimpleLanguageModel,
    loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    config: Config,
    tokenizer: Tokenizer,
    start_step: int,
) -> tuple[float, float, float, float]:
    """Run the training loop, returning (initial, final, val) loss and elapsed seconds."""
    model.train()
    step = start_step
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
        for inputs, targets in loader:  # inputs, targets: (B, T)
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)  # (B, T, vocab)

            optimizer.zero_grad()
            loss.backward()  # compute gradients
            optimizer.step()  # nudge every weight to lower the loss

            step += 1

            lr = scheduler.step(step)  # set the LR for this step on the optimizer

            final_loss = loss.item()
            if step == 1:
                initial_loss = final_loss
            if step % 100 == 0:
                log.info(
                    "step %d/%d  loss %.4f  lr %.2e", step, config.steps, final_loss, lr
                )
            if step % config.eval_interval == 0 or step >= config.steps:
                val_loss = evaluate(model, val_loader, device, config.eval_batches)
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
            if step >= config.steps:
                break
    train_seconds = time.perf_counter() - train_start
    log.info("Training took %.2fs (final val loss %.4f)", train_seconds, val_loss)
    return initial_loss, final_loss, val_loss, train_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from this checkpoint file (e.g. checkpoints/step_1000.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = get_device()
    log.info("Using device: %s", device)

    if args.checkpoint:
        # Resume: rebuild tokenizer, model and optimizer from the saved state.
        log.info("Resuming from checkpoint: %s", args.checkpoint)
        ckpt = load_checkpoint(args.checkpoint)
        config = ckpt["config"]
        tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
        text = load_tinystories(config.num_stories)
        model = build_model(config, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.steps,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
        )
        torch.set_rng_state(ckpt["rng_state"])  # continue the same random stream
        start_step = ckpt["step"]
        log.info("Resumed at step %d", start_step)
    else:
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.steps,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
        )
        start_step = 0

    log.info("Loaded %d characters, vocab size %d", len(text), tokenizer.vocab_size)

    # Encode the whole text once, as a tensor so dataset slices are tensors too.
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold out the tail of the data for validation
    split = int(len(data) * (1 - config.val_fraction))
    train_data, val_data = data[:split], data[split:]
    train_dataset = SlidingWindowDataset(train_data, config.context_length)
    val_dataset = SlidingWindowDataset(val_data, config.context_length)
    log.info("Split into %d train / %d val tokens", len(train_data), len(val_data))

    # DataLoader batches windows together and reshuffles every pass.
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    initial_loss, final_loss, val_loss, train_seconds = train(
        model,
        loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
        tokenizer,
        start_step,
    )

    prompt = "The "
    prompt_start = time.perf_counter()
    outcome = generate(model, tokenizer, prompt, 300, config.context_length)
    prompt_seconds = time.perf_counter() - prompt_start
    log.info("Sample after training (%.2fs):\n%s", prompt_seconds, outcome)

    log_result(
        config,
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


if __name__ == "__main__":
    main()
