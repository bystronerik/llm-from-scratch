from typing import Sequence


class Dataset:
    """A sliding-window dataset over a sequence of numbers.

    Indexing returns an (input, target) tuple where the target is the
    input shifted forward by one (next-token prediction). For example,
    with context_length=4, index 0 yields (data[0:4], data[1:5]), index 1
    yields (data[1:5], data[2:6]), and so on.

    len() is the number of valid indices, i.e. how many windows fit.
    """

    def __init__(self, data: Sequence[int], context_length: int):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.data = data
        self.context_length = context_length

    def __len__(self):
        n = len(self.data) - self.context_length
        return max(0, n)

    def __getitem__(self, index):
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("dataset index out of range")

        c = self.context_length
        inputs = self.data[index : index + c]
        targets = self.data[index + 1 : index + c + 1]
        return inputs, targets
