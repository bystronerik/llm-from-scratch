import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import SimpleLanguageModel  # noqa: E402
from training import MasterWeights  # noqa: E402

VOCAB, EMB, CTX = 64, 32, 16
LR = 1e-4  # our min_lr: updates here land below the bf16 gap at gamma's scale


@pytest.fixture(autouse=True)
def bf16_default():
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(torch.float32)


def _model():
    torch.manual_seed(0)
    return SimpleLanguageModel(VOCAB, EMB, CTX, num_heads=2, num_transformers=1)


def _steps(model, params, n=50, master=None):
    opt = torch.optim.AdamW(params, lr=LR)
    x = torch.randint(0, VOCAB, (4, CTX))
    y = torch.randint(0, VOCAB, (4, CTX))
    for _ in range(n):
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if master is not None:
            master.copy_grads()
        opt.step()
        if master is not None:
            master.copy_to_model()


def test_bf16_alone_freezes_rmsnorm_gamma():
    """Documents the bug the masters exist to fix."""
    model = _model()
    gamma = model.blocks[0].ln_1.gamma
    assert gamma.dtype is torch.bfloat16

    _steps(model, list(model.parameters()))

    assert torch.all(gamma == 1.0), "expected bf16 gamma to be stuck at its init"


def test_master_weights_unfreeze_rmsnorm_gamma():
    model = _model()
    master = MasterWeights(model)
    gamma = model.blocks[0].ln_1.gamma

    _steps(model, master.parameters(), master=master)

    assert not torch.all(gamma == 1.0), "gamma should move once updates accumulate"
    assert gamma.dtype is torch.bfloat16, (
        "model must stay bf16 to keep activations cheap"
    )


def test_masters_are_fp32_and_hold_subulp_detail():
    model = _model()
    master = MasterWeights(model)

    state = master.state_dict()
    assert all(t.dtype is torch.float32 for t in state.values())

    _steps(model, master.parameters(), master=master)

    name = "blocks.0.ln_1.gamma"
    fp32 = master.state_dict()[name]
    bf16 = dict(model.named_parameters())[name]
    # The master carries precision the bf16 copy cannot represent; that residue
    # is exactly what keeps sub-ulp updates from being discarded.
    assert not torch.equal(fp32, bf16.float())


def test_restoring_from_state_preserves_masters():
    model = _model()
    master = MasterWeights(model)
    _steps(model, master.parameters(), master=master)
    saved = master.state_dict()

    fresh = _model()
    restored = MasterWeights(fresh, saved)

    for name, tensor in saved.items():
        assert torch.equal(restored.state_dict()[name], tensor)
        # ...and the bf16 model was refreshed to match.
        assert torch.equal(
            dict(fresh.named_parameters())[name], tensor.to(torch.bfloat16)
        )
