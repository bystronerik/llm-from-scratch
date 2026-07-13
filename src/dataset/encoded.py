import hashlib
import logging
import os

import torch

log = logging.getLogger("dataset.encoded")


def encode_corpus(tokenizer, text: str, cache_dir: str) -> torch.Tensor:
    """Encode ``text`` to a token tensor, caching the result on disk.

    Encoding the full corpus with a BPE tokenizer is expensive (it replays
    every merge over the whole text), so the tensor is cached under
    ``cache_dir`` keyed by a hash of both the tokenizer state and the text.
    A later run with the same tokenizer and corpus loads the tensor directly
    instead of re-encoding.
    """
    key = _cache_key(tokenizer, text)
    cache_path = os.path.join(cache_dir, "datasets", f"encoded_{key}.pt")

    if os.path.exists(cache_path):
        log.info("Loading cached encoded corpus from %s", cache_path)
        return torch.load(cache_path, weights_only=True)

    log.info("Encoding %d characters ...", len(text))
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(data, cache_path)
    log.info("Saved encoded corpus (%d tokens) to %s", len(data), cache_path)
    return data


def _cache_key(tokenizer, text: str) -> str:
    """Stable hash over the tokenizer state and the text.

    Both matter: the encoding must be recomputed if the merges change or the
    corpus changes.
    """
    state = tokenizer.get_state()

    h = hashlib.sha256()
    h.update(repr((type(tokenizer).__name__, state)).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]
