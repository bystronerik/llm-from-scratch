import json
import os
import re
from collections import Counter, defaultdict

_CHUNK_RE = re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]+|\s+")


def _bytes_to_unicode() -> dict[int, str]:
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + spare)
            spare += 1
    return mapping


_BYTE_TO_CHAR = _bytes_to_unicode()
_CHAR_TO_BYTE = {c: b for b, c in _BYTE_TO_CHAR.items()}
_BASE_VOCAB = set(_BYTE_TO_CHAR.values())


def _byte_chars(chunk: str) -> list[str]:
    return [_BYTE_TO_CHAR[b] for b in chunk.encode("utf-8")]


def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def _apply_merge(chunk, pair, freq, chunks, pair_freq, pair_chunks) -> None:
    old = chunks[chunk]
    new = _merge_pair(old, pair)
    chunks[chunk] = new
    old_pairs = Counter(zip(old, old[1:]))
    new_pairs = Counter(zip(new, new[1:]))
    for p in old_pairs.keys() | new_pairs.keys():
        delta = new_pairs[p] - old_pairs[p]
        if delta:
            pair_freq[p] += delta * freq
            if pair_freq[p] <= 0:
                del pair_freq[p]
        if new_pairs[p] and not old_pairs[p]:
            pair_chunks[p].add(chunk)
        elif old_pairs[p] and not new_pairs[p]:
            pair_chunks[p].discard(chunk)
            if not pair_chunks[p]:
                del pair_chunks[p]


def train_merges(text: str, num_merges: int) -> list[tuple[str, str]]:
    chunk_counts = Counter(_CHUNK_RE.findall(text))
    chunks = {chunk: _byte_chars(chunk) for chunk in chunk_counts}

    pair_freq: Counter = Counter()
    pair_chunks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for chunk, freq in chunk_counts.items():
        tokens = chunks[chunk]
        for pair in zip(tokens, tokens[1:]):
            pair_freq[pair] += freq
            pair_chunks[pair].add(chunk)

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        if not pair_freq:
            break
        pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if pair_freq[pair] < 2:
            break
        merges.append(pair)
        for chunk in list(pair_chunks[pair]):
            _apply_merge(
                chunk, pair, chunk_counts[chunk], chunks, pair_freq, pair_chunks
            )
    return merges


class BPETokenizer:
    def __init__(self, merges, vocabulary, special_tokens=None):
        self.merges: list[tuple[str, str]] = [tuple(m) for m in merges]
        self.vocabulary: list[str] = list(vocabulary)
        self.token_to_id = {t: i for i, t in enumerate(self.vocabulary)}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._chunk_cache: dict[str, list[int]] = {}
        self.special_tokens: list[str] = []
        self._special_re: re.Pattern | None = None
        if special_tokens:
            self.add_special_tokens(special_tokens)

    @classmethod
    def train(cls, text: str, num_merges: int) -> "BPETokenizer":
        merges = train_merges(text, num_merges)
        vocabulary = sorted(_BASE_VOCAB | {a + b for a, b in merges})
        return cls(merges, vocabulary)

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(
            state["merges"], state["vocabulary"], state.get("special_tokens", [])
        )

    def get_state(self) -> dict:
        return {
            "merges": self.merges,
            "vocabulary": self.vocabulary,
            "special_tokens": self.special_tokens,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state(state)

    def add_special_tokens(self, tokens: list[str]) -> None:
        for tok in tokens:
            if tok not in self.special_tokens:
                self.special_tokens.append(tok)
        if self.special_tokens:
            pattern = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re = re.compile(f"({pattern})")
        else:
            self._special_re = None

    @property
    def _first_special_id(self) -> int:
        return len(self.vocabulary)

    def _special_id(self, token: str) -> int:
        return self._first_special_id + self.special_tokens.index(token)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary) + len(self.special_tokens)

    def _bpe(self, tokens: list[str]) -> list[str]:
        while len(tokens) >= 2:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_rank is None:
                break
            tokens = _merge_pair(tokens, (tokens[best_i], tokens[best_i + 1]))
        return tokens

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = self._chunk_cache.get(chunk)
        if ids is None:
            tokens = self._bpe(_byte_chars(chunk))
            ids = [self.token_to_id[t] for t in tokens]
            self._chunk_cache[chunk] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _CHUNK_RE.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_plain(text)

        ids: list[int] = []
        for segment in self._special_re.split(text):
            if not segment:
                continue
            if segment in self.special_tokens:
                ids.append(self._special_id(segment))
            else:
                ids.extend(self._encode_plain(segment))
        return ids

    def decode(self, numbers: list[int]) -> str:
        out: list[str] = []
        pending: list[str] = []

        def flush() -> None:
            if pending:
                data = bytes(_CHAR_TO_BYTE[c] for tok in pending for c in tok)
                out.append(data.decode("utf-8", errors="replace"))
                pending.clear()

        for n in numbers:
            if n >= self._first_special_id:
                flush()
                out.append(self.special_tokens[n - self._first_special_id])
            else:
                pending.append(self.vocabulary[n])
        flush()
        return "".join(out)
