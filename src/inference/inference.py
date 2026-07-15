import torch

from model import SimpleLanguageModel
from tokenizer import BPETokenizer


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
                if temperature != 1.0:
                    last_step = last_step / temperature

                if top_k is not None:
                    k = min(top_k, last_step.size(-1))

                    threshold = torch.topk(last_step, k).values[-1]

                    last_step = last_step.masked_fill(
                        last_step < threshold, float("-inf")
                    )

                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(last_step, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)

                    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs

                    remove = mass_before >= top_p
                    remove[0] = False

                    last_step = last_step.index_fill(
                        0, sorted_idx[remove], float("-inf")
                    )

                probs = torch.softmax(last_step, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id])

            if stop_id is not None and next_id.item() == stop_id:
                break

    if was_training:
        model.train()

    return tokenizer.decode(ids.tolist())
