import logging
import os
from multiprocessing import Pool
from typing import Callable, Iterator

import numpy as np

from .cache import cache_key

log = logging.getLogger("dataset.corpus_bin")


_WRITE_CHUNK_TOKENS = 1 << 20


_POOL_CHUNKSIZE = 64


_worker_tokenizer = None


def _init_worker(state: dict) -> None:
    global _worker_tokenizer
    from tokenizer import BPETokenizer

    _worker_tokenizer = BPETokenizer.from_state(state)


def _encode_article(article: str) -> list[int]:
    return _worker_tokenizer.encode(article)


def _resolve_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def prepare_corpus(
    tokenizer,
    article_iter_factory: Callable[[], Iterator[str]],
    discriminators: tuple,
    cache_dir: str,
    base_name: str = "corpus",
    num_workers: int = 0,
) -> tuple[np.memmap, int]:
    dtype = _resolve_dtype(tokenizer.vocab_size)
    key = cache_key(tokenizer, *discriminators)
    path = os.path.join(cache_dir, "datasets", f"{base_name}_{key}.bin")

    if not os.path.exists(path):
        workers = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        _write(tokenizer, article_iter_factory(), path, dtype, workers)
    return _open(path, dtype)


def _encoded_articles(
    tokenizer, articles: Iterator[str], workers: int
) -> Iterator[list[int]]:
    if workers == 1:
        for article in articles:
            yield tokenizer.encode(article)
        return

    log.info("Encoding corpus across %d worker processes", workers)
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(tokenizer.get_state(),),
    ) as pool:
        yield from pool.imap(_encode_article, articles, chunksize=_POOL_CHUNKSIZE)


def _write(
    tokenizer, articles: Iterator[str], path: str, dtype: np.dtype, workers: int
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    log.info("Encoding corpus to %s (dtype=%s) ...", path, dtype.name)

    buffer: list[np.ndarray] = []
    buffered = 0
    total = 0
    with open(tmp_path, "wb") as f:
        for ids in _encoded_articles(tokenizer, articles, workers):
            buffer.append(np.asarray(ids, dtype=dtype))
            buffered += len(ids)
            if buffered >= _WRITE_CHUNK_TOKENS:
                np.concatenate(buffer).tofile(f)
                total += buffered
                buffer, buffered = [], 0
        if buffer:
            np.concatenate(buffer).tofile(f)
            total += buffered

    os.replace(tmp_path, path)
    log.info("Wrote %d tokens (%d bytes) to %s", total, total * dtype.itemsize, path)


def _open(path: str, dtype: np.dtype) -> tuple[np.memmap, int]:
    length = os.path.getsize(path) // dtype.itemsize
    data = np.memmap(path, dtype=dtype, mode="r")
    log.info("Memory-mapped %d tokens from %s", length, path)
    return data, length
