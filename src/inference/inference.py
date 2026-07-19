import torch
from torch import Tensor

from model import SimpleLanguageModel
from tokenizer import BPETokenizer


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))

        threshold = torch.topk(logits, k).values[-1]

        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs

        remove = mass_before >= top_p
        remove[0] = False

        logits = logits.index_fill(0, sorted_idx[remove], float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)


def generate(
    model: SimpleLanguageModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_id: int | None = None,
    sink_tokens: int = 4,
    window_size: int | None = None,
) -> str:
    was_training = model.training

    model.eval()

    device = next(model.parameters()).device

    if window_size is None:
        window_size = context_length - sink_tokens

    caches = model.new_caches(sink_tokens, window_size)
    capacity = sink_tokens + window_size

    ids = torch.tensor(
        tokenizer.encode(prompt)[-capacity:], dtype=torch.long, device=device
    )

    with torch.no_grad():
        logits, _ = model(ids.unsqueeze(0), caches=caches)

        for _ in range(max_new_tokens):
            next_id = _sample(logits[0, -1], temperature, top_k, top_p)

            ids = torch.cat([ids, next_id])

            if stop_id is not None and next_id.item() == stop_id:
                break

            logits, _ = model(next_id.view(1, 1), caches=caches)

    if was_training:
        model.train()

    return tokenizer.decode(ids.tolist())
