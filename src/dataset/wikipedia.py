import itertools
import logging
import os
from typing import Iterator

from .huggingface import fetch_dataset

log = logging.getLogger("dataset.wikipedia")

HF_DATASET = "wikimedia/wikipedia"


ARTICLE_SEPARATOR = "\n\n"


def iter_wikipedia_articles(
    num_articles: int, dump: str, cache_dir: str
) -> Iterator[str]:
    log.info("Reading %d articles from %s (%s) ...", num_articles, HF_DATASET, dump)
    ds = fetch_dataset(HF_DATASET, "train", name=dump, cache_dir=cache_dir)
    for row in itertools.islice(ds, num_articles):
        yield row["text"] + ARTICLE_SEPARATOR


def load_wikipedia_text(
    num_articles: int, dump: str, cache_path: str, cache_dir: str
) -> str:
    root, ext = os.path.splitext(cache_path)
    cache_path = f"{root}_{num_articles}{ext}"

    if os.path.exists(cache_path):
        log.info("Loading cached Wikipedia tokenizer text from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    text = "".join(iter_wikipedia_articles(num_articles, dump, cache_dir))

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


def build_wikipedia_corpus(
    tokenizer, num_articles: int, dump: str, cache_dir: str, num_workers: int = 0
):
    from .corpus_bin import prepare_corpus

    return prepare_corpus(
        tokenizer,
        lambda: iter_wikipedia_articles(num_articles, dump, cache_dir),
        discriminators=(HF_DATASET, dump, num_articles),
        cache_dir=cache_dir,
        num_workers=num_workers,
    )
