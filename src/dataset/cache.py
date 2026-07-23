import hashlib
import logging
import os
from typing import Callable

import numpy as np
import torch

log = logging.getLogger("dataset.cache")


def cache_key(tokenizer, *discriminators) -> str:
    h = hashlib.sha256()
    payload = (type(tokenizer).__name__, tokenizer.get_state(), *discriminators)
    h.update(repr(payload).encode("utf-8"))
    return h.hexdigest()[:16]


def cached_payload(cache_dir: str, name: str, key: str, build: Callable):
    cache_path = os.path.join(cache_dir, "datasets", f"{name}_{key}.pt")

    if os.path.exists(cache_path):
        log.info("Loading cached %s from %s", name, cache_path)
        return torch.load(cache_path, weights_only=True)

    payload = build()

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(payload, cache_path)

    log.info("Saved %s to %s", name, cache_path)

    return payload


def cached_tokens(
    cache_dir: str, key: str, build: Callable[[], np.ndarray]
) -> np.memmap:
    cache_path = os.path.join(cache_dir, "datasets", f"tokens_{key}.npy")

    if os.path.exists(cache_path):
        log.info("Loading cached tokens from %s", cache_path)
        return np.load(cache_path, mmap_mode="r")

    tokens = build()

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)

    staged = f"{cache_path}.partial"
    np.save(staged, tokens, allow_pickle=False)
    os.replace(f"{staged}.npy", cache_path)

    log.info("Saved %d tokens to %s", len(tokens), cache_path)

    return np.load(cache_path, mmap_mode="r")
