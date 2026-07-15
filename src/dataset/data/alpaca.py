import logging
import os

from dataset.huggingface import fetch_dataset

log = logging.getLogger("dataset.alpaca")

HF_DATASET = "tatsu-lab/alpaca"


def load_alpaca_text(cache_dir: str, cache_path: str) -> str:
    if os.path.exists(cache_path):
        log.info("Loading cached Alpaca text from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    dataset = fetch_dataset(HF_DATASET, None, cache_dir)

    if hasattr(dataset, "keys"):
        dataset = dataset["train"]

    parts = []
    for row in dataset:
        parts.append(
            "\n".join(f for f in (row["instruction"], row["input"], row["output"]) if f)
        )
    text = "\n".join(parts)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    log.info("Saved %d rows (%d chars) to %s", len(dataset), len(text), cache_path)

    return text
