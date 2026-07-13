import logging
import time

import torch

from cli import parse_args
from dataset import build_dataloaders
from inference import generate
from results import log_result
from training import build_run, resume_run, train
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = get_device()
    log.info("Using device: %s", device)

    run = resume_run(args.checkpoint, device) if args.checkpoint else build_run(device)
    log.info(
        "Loaded %d characters, vocab size %d", len(run.text), run.tokenizer.vocab_size
    )

    # Encode the whole corpus once, as a tensor so dataset slices are tensors too.
    data = torch.tensor(run.tokenizer.encode(run.text), dtype=torch.long)
    loader, val_loader = build_dataloaders(
        data, run.config.context_length, run.config.batch_size, run.config.val_fraction
    )

    num_params = sum(p.numel() for p in run.model.parameters())
    log.info("Parameters: %d", num_params)

    initial_loss, final_loss, val_loss, train_seconds = train(
        run, loader, val_loader, device
    )

    prompt = "The "
    prompt_start = time.perf_counter()
    outcome = generate(run.model, run.tokenizer, prompt, 300, run.config.context_length)
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


if __name__ == "__main__":
    main()
