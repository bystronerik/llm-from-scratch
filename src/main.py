import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import SlidingWindowDataset
from utils import get_device
from model import SimpleLanguageModel
from results import log_result
from tokenizer import Tokenizer, BPETokenizer
from training.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

# --- Hyperparameters ---
CONTEXT_LENGTH = 16  # how many characters the model sees at once (T)
EMBEDDING_DIM = 64  # size of each token's vector (C)
BATCH_SIZE = 64  # how many windows per training step (B)
STEPS = 5000  # number of gradient updates
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1  # portion of the data held out for validation
EVAL_INTERVAL = 500  # how often (in steps) to measure validation loss
EVAL_BATCHES = 50  # how many val batches to average per measurement
NUM_HEADS = 4  # attention heads per transformer block
NUM_TRANSFORMERS = 4  # number of stacked transformer blocks
BPE_MERGES = 200  # how many byte-pair merges the tokenizer learns
CHECKPOINT_INTERVAL = 1000  # save a checkpoint every N steps (plus at the end)
CHECKPOINT_DIR = "checkpoints"  # where checkpoints are written


def generate(
    model: SimpleLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Feed the prompt in, then repeatedly sample one next character."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-CONTEXT_LENGTH:].unsqueeze(0)  # (1, T)
            logits, _ = model(context)  # (1, T, vocab)
            last_step = logits[0, -1]  # (vocab,) scores for the next char
            probs = torch.softmax(last_step, dim=-1)  # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample one
            ids = torch.cat([ids, next_id])
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


def build_model(config: dict, device: torch.device) -> SimpleLanguageModel:
    """Construct a model from a saved config dict and move it onto `device`."""
    model = SimpleLanguageModel(
        config["vocab_size"],
        config["embedding_dim"],
        config["context_length"],
        config["num_heads"],
        config["num_transformers"],
    )
    return model.to(device)


def make_checkpoint(
    model: SimpleLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    tokenizer: BPETokenizer,
    hyperparams: dict,
) -> dict:
    """Bundle everything needed to resume training exactly from this point."""
    return {
        "step": step,
        "model_config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer": tokenizer.get_state(),
        "hyperparams": hyperparams,
        "rng_state": torch.get_rng_state(),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = get_device()
    log.info("Using device: %s", device)

    with open("datasets/basic.txt", encoding="utf-8") as f:
        text = f.read()

    hyperparams = {
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
    }

    if args.checkpoint:
        # Resume: rebuild tokenizer, model and optimizer from the saved state.
        log.info("Resuming from checkpoint: %s", args.checkpoint)
        ckpt = load_checkpoint(args.checkpoint)
        tokenizer = BPETokenizer.from_state(ckpt["tokenizer"])
        config = ckpt["model_config"]
        hyperparams = ckpt["hyperparams"]
        model = build_model(config, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hyperparams["learning_rate"]
        )
        optimizer.load_state_dict(ckpt["optimizer_state"])
        torch.set_rng_state(ckpt["rng_state"])  # continue the same random stream
        start_step = ckpt["step"]
        log.info("Resumed at step %d", start_step)
    else:
        tokenizer = BPETokenizer(text, BPE_MERGES)
        config = {
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "context_length": CONTEXT_LENGTH,
            "num_heads": NUM_HEADS,
            "num_transformers": NUM_TRANSFORMERS,
        }
        model = build_model(config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        start_step = 0

    log.info("Loaded %d characters, vocab size %d", len(text), tokenizer.vocab_size)

    # Encode the whole text once, as a tensor so dataset slices are tensors too.
    context_length = config["context_length"]
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold out the tail of the data for validation
    split = int(len(data) * (1 - VAL_FRACTION))
    train_data, val_data = data[:split], data[split:]
    train_dataset = SlidingWindowDataset(train_data, context_length)
    val_dataset = SlidingWindowDataset(val_data, context_length)
    log.info("Split into %d train / %d val tokens", len(train_data), len(val_data))

    # DataLoader batches windows together and reshuffles every pass.
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    model.train()
    step = start_step
    if step >= STEPS:
        log.info(
            "Checkpoint already at step %d >= STEPS %d; nothing to train.", step, STEPS
        )
    initial_loss = float("nan")
    final_loss = float("nan")
    val_loss = float("nan")
    train_start = time.perf_counter()
    while step < STEPS:
        for inputs, targets in loader:  # inputs, targets: (B, T)
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)  # (B, T, vocab)

            optimizer.zero_grad()
            loss.backward()  # compute gradients
            optimizer.step()  # nudge every weight to lower the loss

            step += 1
            final_loss = loss.item()
            if step == 1:
                initial_loss = final_loss
            if step % 100 == 0:
                log.info("step %d/%d  loss %.4f", step, STEPS, final_loss)
            if step % EVAL_INTERVAL == 0 or step >= STEPS:
                val_loss = evaluate(model, val_loader, device, EVAL_BATCHES)
                log.info(
                    "step %d/%d  train loss %.4f  val loss %.4f",
                    step,
                    STEPS,
                    final_loss,
                    val_loss,
                )
            if step % CHECKPOINT_INTERVAL == 0 or step >= STEPS:
                path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
                save_checkpoint(
                    make_checkpoint(
                        model, optimizer, step, config, tokenizer, hyperparams
                    ),
                    path,
                )
                log.info("Saved checkpoint: %s", path)
            if step >= STEPS:
                break
    train_seconds = time.perf_counter() - train_start
    log.info("Training took %.2fs (final val loss %.4f)", train_seconds, val_loss)

    prompt = "The "
    prompt_start = time.perf_counter()
    outcome = generate(model, tokenizer, prompt, 300)
    prompt_seconds = time.perf_counter() - prompt_start
    log.info("Sample after training (%.2fs):\n%s", prompt_seconds, outcome)

    log_result(
        hyperparams,
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
