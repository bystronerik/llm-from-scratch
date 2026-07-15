import logging
import os

from .huggingface import fetch_dataset

log = logging.getLogger("dataset.alpaca")

HF_DATASET = "tatsu-lab/alpaca"
DEFAULT_CACHE_PATH = ".cache/datasets/alpaca.txt"


def load_alpaca_corpus(
    num_examples: int, cache_dir: str, cache_path: str = DEFAULT_CACHE_PATH
) -> str:
    root, ext = os.path.splitext(cache_path)
    cache_path = f"{root}_{num_examples}{ext}"

    if os.path.exists(cache_path):
        log.info("Loading cached Alpaca text from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    dataset = fetch_dataset(HF_DATASET, "train", cache_dir=cache_dir)
    rows = dataset.select(range(min(num_examples, len(dataset))))
    parts = []
    for row in rows:
        parts.append(
            "\n".join(f for f in (row["instruction"], row["input"], row["output"]) if f)
        )
    text = "\n".join(parts)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Saved %d rows (%d chars) to %s", len(rows), len(text), cache_path)
    return text
