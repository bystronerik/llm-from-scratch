import torch
from torch import Tensor

from model import SimpleLanguageModel
from tokenizer import BPETokenizer


def _filter_logits(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Tensor:
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

    return logits


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
) -> str:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-context_length:].unsqueeze(0)
            logits, _ = model(context)
            last_step = logits[0, -1]

            if temperature == 0:
                next_id = last_step.argmax(dim=-1, keepdim=True)
            else:
                filtered = _filter_logits(last_step, temperature, top_k, top_p)
                probs = torch.softmax(filtered, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id])
            if stop_id is not None and next_id.item() == stop_id:
                break
    if was_training:
        model.train()
    return tokenizer.decode(ids.tolist())
