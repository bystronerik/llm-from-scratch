import json
import os
import re
from collections import Counter
from typing import Protocol

UNKNOWN = 0

_CHUNK_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+|\s+")


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, numbers: list[int]) -> str: ...


class BPETokenizer:
    """Byte-pair-encoding tokenizer.

    Starts from single characters and learns `num_merges` merges, each time
    fusing the most common adjacent pair into a new token.

    Merges are learned and applied *within* a pre-tokenized chunk (a run of
    whitespace or non-whitespace), never across chunk boundaries. Because
    natural text repeats the same words, both training and encoding work over
    the set of *unique* chunks and cache per-chunk results, so cost scales with
    the vocabulary of the corpus rather than its raw length.
    """

    def __init__(self, text: str, num_merges: int):
        # Count how often each chunk occurs, and hold one token list per unique
        # chunk. Merges below operate on this small set, weighted by frequency.
        chunk_counts = Counter(_CHUNK_RE.findall(text))
        chunks = {chunk: list(chunk) for chunk in chunk_counts}

        self.merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            counts: Counter = Counter()
            for chunk, freq in chunk_counts.items():
                tokens = chunks[chunk]
                for pair in zip(tokens, tokens[1:]):
                    counts[pair] += freq
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:  # nothing left worth merging
                break
            self.merges.append(pair)
            for chunk in chunks:
                chunks[chunk] = self._merge_pair(chunks[chunk], pair)

        # Vocabulary is every distinct token the merges can produce, plus the
        # base characters. Sorted so ids are stable across runs.
        self.vocabulary = sorted(set(text) | {a + b for a, b in self.merges})
        self.token_to_id = {t: i for i, t in enumerate(self.vocabulary, start=1)}
        self._chunk_cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        """Number of tokens, including the UNKNOWN (0) token."""
        return len(self.vocabulary) + 1

    def get_state(self) -> dict:
        """The learned state needed to reconstruct this tokenizer exactly."""
        return {"merges": self.merges, "vocabulary": self.vocabulary}

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        """Rebuild a tokenizer from get_state() output, skipping BPE training."""
        obj = cls.__new__(cls)  # bypass __init__: no text to re-learn from
        obj.merges = [tuple(pair) for pair in state["merges"]]
        obj.vocabulary = list(state["vocabulary"])
        obj.token_to_id = {t: i for i, t in enumerate(obj.vocabulary, start=1)}
        obj._chunk_cache = {}
        return obj

    def save(self, path: str) -> None:
        """Write the learned state to `path` as JSON so training can be skipped
        on later runs. Creates parent directories as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Rebuild a tokenizer from a file written by save(), skipping training."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state(state)

    def _tokenize(self, text: str) -> list[str]:
        """Split text into characters, then replay the learned merges in order."""
        tokens = list(text)
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    def _merge_pair(self, tokens: list[str], pair: tuple[str, str]) -> list[str]:
        """Replace every adjacent occurrence of `pair` with the merged string."""
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one chunk to ids, caching the result for repeated chunks."""
        ids = self._chunk_cache.get(chunk)
        if ids is None:
            ids = [self.token_to_id.get(t, UNKNOWN) for t in self._tokenize(chunk)]
            self._chunk_cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string as a list of token ids. Unknown tokens map to 0.

        The text is pre-split into chunks and each unique chunk is tokenized
        once, so repeated words cost only a dictionary lookup.
        """
        ids: list[int] = []
        for chunk in _CHUNK_RE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, numbers: list[int]) -> str:
        """Inverse of encode(). Unknowns come back as '?'."""
        return "".join("?" if n == UNKNOWN else self.vocabulary[n - 1] for n in numbers)
