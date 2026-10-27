"""CPU tests on a small random-weight model (no download needed).

    python -m pytest tests          # or: python tests/test_model.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nomad_eeg import CHANNELS_21, Classifier, SplitSSM, SplitSSMConfig, channel_coords   # noqa: E402

SMALL = SplitSSMConfig(d_model=32, n_blocks=4, d_state=16, n_heads=4, d_cond=32,
                       objective="diffusion", feature_blocks=(1, 3))


def small_model():
    torch.manual_seed(0)
    return SplitSSM(SMALL).double().eval()


def test_any_montage_and_length():
    m = small_model()
    for chans, T in [(["CZ"], 400), (["FPZ-CZ", "PZ-OZ"], 1000), (CHANNELS_21, 1000), (CHANNELS_21[:5], 700)]:
        f = m.features(torch.randn(2, len(chans), T, dtype=torch.float64), channel_coords(chans).double(),
                       torch.full((2,), 3), n_windows=5)
        assert f.shape == (2, 5, 2 * 2 * SMALL.d_model)
        assert torch.isfinite(f).all()


def test_electrode_order_does_not_matter():
    m = small_model()
    x, c = torch.randn(2, 21, 800, dtype=torch.float64), channel_coords().double()
    perm = torch.randperm(21)
    cid = torch.full((2,), 3)
    a = m.features(x, c, cid, n_windows=5)
    b = m.features(x[:, perm], c[perm], cid, n_windows=5)
    assert torch.allclose(a, b, atol=1e-8), (a - b).abs().max()


def test_classifier_head():
    m = small_model().float()
    clf = Classifier(m, n_classes=4, d=32)
    logits = clf(torch.randn(3, 8, 800), channel_coords(CHANNELS_21[:8]), torch.full((3,), 3),
                 train_encoder=True)
    assert logits.shape == (3, 4)
    logits.sum().backward()
    assert m.stem.weight.grad is not None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
