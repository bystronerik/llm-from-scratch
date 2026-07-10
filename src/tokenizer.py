UNKNOWN = -1


class CharacterTokenizer:
    def __init__(self, text: str):
        self.alphabet = sorted(set(text))
        self.char_to_num = {c: i for i, c in enumerate(self.alphabet)}

    def encode(self, text: str) -> list[int]:
        """Encode a string as a list of numbers, one per character.

        Characters outside the alphabet map to UNKNOWN (-1).
        """
        return [self.char_to_num.get(c, UNKNOWN) for c in text]

    def decode(self, numbers: list[int]) -> str:
        """Inverse of encode(). Unknowns come back as '?'."""
        return "".join(
            "?" if n == UNKNOWN else self.alphabet[n] for n in numbers
        )
