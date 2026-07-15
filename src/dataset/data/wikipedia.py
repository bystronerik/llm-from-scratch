import logging
import os
from typing import Iterator

from dataset.huggingface import fetch_dataset

log = logging.getLogger("dataset.wikipedia")

HF_DATASET = "wikimedia/wikipedia"
HF_NAME = "20231101.en"


ARTICLE_SEPARATOR = "\n\n"


def iter_wikipedia_articles(num_articles: int, cache_dir: str) -> Iterator[str]:
    log.info("Reading %d articles from %s ...", num_articles, HF_DATASET)

    ds = fetch_dataset(HF_DATASET, HF_NAME, cache_dir)

    if hasattr(ds, "keys"):
        ds = ds["train"]

    for i, row in enumerate(ds):
        if i >= num_articles:
            break
        yield row["text"] + ARTICLE_SEPARATOR


def load_wikipedia_text(num_articles: int, cache_dir: str, cache_path: str) -> str:
    root, ext = os.path.splitext(cache_path)
    cache_path = f"{root}_{num_articles}{ext}"

    if os.path.exists(cache_path):
        log.info("Loading cached Wikipedia text from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    text = "".join(iter_wikipedia_articles(num_articles, cache_dir))

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    log.info(
        "Saved %d-article Wikipedia sample (%d chars) to %s",
        num_articles,
        len(text),
        cache_path,
    )

    return text
