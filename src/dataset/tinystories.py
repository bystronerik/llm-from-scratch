import logging
import os

log = logging.getLogger("dataset.tinystories")

HF_DATASET = "roneneldan/TinyStories"
DEFAULT_CACHE_PATH = ".cache/datasets/tinystories.txt"


def load_tinystories(num_stories: int, cache_path: str = DEFAULT_CACHE_PATH) -> str:
    """Load the first ``num_stories`` TinyStories entries as one big string.

    The stories are concatenated with a single newline so the existing
    tokenizer and SlidingWindowDataset can consume the result unchanged.

    On the first run the slice is downloaded via HuggingFace Datasets and the
    joined text is written to a cache file; later runs read that file directly
    and skip the download. ``num_stories`` is baked into the cache filename
    (e.g. ``tinystories_5000.txt``) so changing it uses a distinct cache and
    triggers a fresh download instead of reusing a stale slice.
    """
    root, ext = os.path.splitext(cache_path)
    cache_path = f"{root}_{num_stories}{ext}"

    if os.path.exists(cache_path):
        log.info("Loading cached TinyStories text from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    from datasets import load_dataset

    log.info("Downloading %d stories from %s ...", num_stories, HF_DATASET)
    dataset = load_dataset(HF_DATASET, split=f"train[:{num_stories}]")
    text = "\n".join(dataset["text"])

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Saved %d stories (%d chars) to %s", len(dataset), len(text), cache_path)
    return text
