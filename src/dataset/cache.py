import hashlib
import logging
import os
from typing import Callable

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
