import logging

import torch
from torch.utils.data import DataLoader

from dataset.slidingwindowdataset import SlidingWindowDataset
from model import SimpleLanguageModel
from results import log_result
from tokenizer import CharacterTokenizer

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


def generate(
    model: SimpleLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Feed the prompt in, then repeatedly sample one next character."""
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-CONTEXT_LENGTH:].unsqueeze(0)  # (1, T)
            logits, _ = model(context)  # (1, T, vocab)
            last_step = logits[0, -1]  # (vocab,) scores for the next char
            probs = torch.softmax(last_step, dim=-1)  # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample one
            ids = torch.cat([ids, next_id])
    return tokenizer.decode(ids.tolist())


def main() -> None:
    torch.manual_seed(42)

    with open("datasets/basic.txt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer(text)
    log.info("Loaded %d characters, vocab size %d", len(text), tokenizer.vocab_size)

    # Encode the whole text once, as a tensor so dataset slices are tensors too.
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    dataset = SlidingWindowDataset(data, CONTEXT_LENGTH)

    # DataLoader batches windows together and reshuffles every pass.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleLanguageModel(tokenizer.vocab_size, EMBEDDING_DIM, CONTEXT_LENGTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    model.train()
    step = 0
    initial_loss = float("nan")
    final_loss = float("nan")
    while step < STEPS:
        for inputs, targets in loader:  # inputs, targets: (B, T)
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
            if step >= STEPS:
                break

    hyperparams = {
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
    }
    log_result(hyperparams, num_params, initial_loss, final_loss)

    log.info("Sample after training:\n%s", generate(model, tokenizer, "The ", 300))


if __name__ == "__main__":
    main()
