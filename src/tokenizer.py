UNKNOWN = 0


class CharacterTokenizer:
    def __init__(self, text: str):
        self.vocabulary = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(self.vocabulary, start=1)}

    @property
    def vocab_size(self) -> int:
        """Number of tokens, including the UNKNOWN (0) token."""
        return len(self.vocabulary) + 1

    def encode(self, text: str) -> list[int]:
        """Encode a string as a list of numbers, one per character.

        Characters outside the vocabulary map to UNKNOWN (0).
        """
        return [self.char_to_id.get(c, UNKNOWN) for c in text]

    def decode(self, numbers: list[int]) -> str:
        """Inverse of encode(). Unknowns come back as '?'."""
        return "".join("?" if n == UNKNOWN else self.vocabulary[n - 1] for n in numbers)
