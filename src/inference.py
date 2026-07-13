import torch
from torch import Tensor

from model import SimpleLanguageModel
from tokenizer import Tokenizer


def _filter_logits(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Tensor:
    """Shape the next-token distribution before sampling.

    Operates on a 1-D ``(vocab,)`` logits vector and returns a filtered copy.
    The three controls stack, applied in this order:

    - **temperature** rescales the logits. ``< 1`` sharpens the distribution
      (more confident, more repetitive); ``> 1`` flattens it (more surprising,
      more mistakes); ``1.0`` leaves it unchanged.
    - **top-k** keeps only the ``k`` highest-scoring tokens, discarding the long
      tail of unlikely tokens outright.
    - **top-p** (nucleus) keeps the smallest set of tokens whose probabilities
      already sum to ``p`` -- an adaptive cutoff that widens when the model is
      unsure and narrows when it is confident.

    Discarded tokens are set to ``-inf`` so they vanish under the later softmax.
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        # The k-th largest logit is the cutoff; anything below it is dropped.
        threshold = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        # Sort high-to-low and walk down the cumulative probability. A token is
        # dropped once the mass *before* it already reaches p -- i.e. the
        # nucleus is complete. The top token is always kept.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        remove = mass_before >= top_p
        remove[0] = False
        logits = logits.index_fill(0, sorted_idx[remove], float("-inf"))

    return logits


def generate(
    model: SimpleLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> str:
    """Feed the prompt in, then repeatedly sample one next token.

    With the defaults (``temperature=1.0``, no top-k/top-p) this samples from
    the model's full distribution. Set ``temperature=0`` for greedy (always the
    single most likely token) decoding.
    """
    was_training = model.training
    model.eval()  # disable dropout while sampling
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-context_length:].unsqueeze(0)  # (1, T)
            logits, _ = model(context)  # (1, T, vocab)
            last_step = logits[0, -1]  # (vocab,) scores for the next token

            if temperature == 0:
                # Greedy: no randomness, just take the single best token.
                next_id = last_step.argmax(dim=-1, keepdim=True)
            else:
                filtered = _filter_logits(last_step, temperature, top_k, top_p)
                probs = torch.softmax(filtered, dim=-1)  # scores -> probabilities
                next_id = torch.multinomial(probs, num_samples=1)  # sample one

            ids = torch.cat([ids, next_id])
    if was_training:
        model.train()
    return tokenizer.decode(ids.tolist())
