import logging
import time

import torch
from torch.utils.data import DataLoader

from dataset import SlidingWindowDataset
from device import get_device
from model import SimpleLanguageModel
from results import log_result
from tokenizer import Tokenizer, BPETokenizer

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


def main() -> None:
    torch.manual_seed(42)

    device = get_device()
    log.info("Using device: %s", device)

    with open("datasets/basic.txt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(text, 200)
    log.info("Loaded %d characters, vocab size %d", len(text), tokenizer.vocab_size)

    # Encode the whole text once, as a tensor so dataset slices are tensors too.
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold out the tail of the data for validation
    split = int(len(data) * (1 - VAL_FRACTION))
    train_data, val_data = data[:split], data[split:]
    train_dataset = SlidingWindowDataset(train_data, CONTEXT_LENGTH)
    val_dataset = SlidingWindowDataset(val_data, CONTEXT_LENGTH)
    log.info("Split into %d train / %d val tokens", len(train_data), len(val_data))

    # DataLoader batches windows together and reshuffles every pass.
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleLanguageModel(tokenizer.vocab_size, EMBEDDING_DIM, CONTEXT_LENGTH)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    model.train()
    step = 0
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
            if step >= STEPS:
                break
    train_seconds = time.perf_counter() - train_start
    log.info("Training took %.2fs (final val loss %.4f)", train_seconds, val_loss)

    prompt = "The "
    prompt_start = time.perf_counter()
    outcome = generate(model, tokenizer, prompt, 300)
    prompt_seconds = time.perf_counter() - prompt_start
    log.info("Sample after training (%.2fs):\n%s", prompt_seconds, outcome)

    hyperparams = {
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
    }
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
