"""Entraînement : arcface à marge nulle = nsoftmax, même graine même résultat, lots bien composés."""

import numpy as np
import pytest
import torch
from pytorch_metric_learning import samplers

from facegeom.train import Head, make_loss, sample_params, train

CFG = {
    "head": {"in_dim": 16, "out_dim": 8},
    "batch_size": 8,
    "max_epochs": 3,
    "patience": 5,
    "eval_every": 1,
    "losses": {
        "nsoftmax": {"m_per_class": 1},
        "arcface": {"m_per_class": 1},
        "contrastive": {"m_per_class": 4},
        "triplet": {"m_per_class": 4},
    },
}
PARAMS = {"nsoftmax": {"lr": 1e-3, "s": 16}, "arcface": {"lr": 1e-3, "s": 16, "m": 0.3},
          "contrastive": {"lr": 1e-3, "neg_margin": 0.5}, "triplet": {"lr": 1e-3, "margin": 0.1}}


def _data(n_id, per_id, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_id, 16))
    x = np.repeat(centers, per_id, axis=0) + 0.3 * rng.normal(size=(n_id * per_id, 16))
    return x.astype(np.float32), np.repeat(np.arange(n_id), per_id)


def test_arcface_with_zero_margin_equals_nsoftmax():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    y = torch.arange(10) % 5
    a, _ = make_loss("arcface", {"s": 20, "m": 0.0}, 5, 8)
    b, _ = make_loss("nsoftmax", {"s": 20}, 5, 8)
    b.weight.data.copy_(a.weight.data)  # mêmes poids de classes
    assert a(z, y).item() == pytest.approx(b(z, y).item(), rel=1e-5)


@pytest.mark.parametrize("loss", ["nsoftmax", "arcface", "contrastive", "triplet"])
def test_same_seed_same_result(loss):
    xtr, ytr = _data(10, 6, 0)
    xva, yva = _data(5, 4, 1)
    r1 = train(CFG, loss, PARAMS[loss], 0, xtr, ytr, xva, yva, "cpu", log=lambda s: None)
    r2 = train(CFG, loss, PARAMS[loss], 0, xtr, ytr, xva, yva, "cpu", log=lambda s: None)
    assert r1["history"] == r2["history"]
    for k in r1["head_state"]:
        assert torch.equal(r1["head_state"][k], r2["head_state"][k])
    assert len(r1["history"]) == 3 and r1["best_epoch"] >= 1


def test_batches_have_m_per_class():
    labels = np.repeat(np.arange(20), 5)
    sampler = samplers.MPerClassSampler(labels, 4, 32, num_batches=10, generator=torch.Generator().manual_seed(0))
    for idx in sampler:
        counts = np.unique(labels[idx], return_counts=True)[1]
        assert len(idx) == 32 and set(counts) == {4}


def test_sample_params_respects_ranges_and_seed():
    ranges = {"lr": [1e-5, 1e-2, "log"], "s": [8, 64]}
    a, b = sample_params(ranges, 10, 100), sample_params(ranges, 10, 100)
    assert a == b
    assert all(1e-5 <= p["lr"] <= 1e-2 and 8 <= p["s"] <= 64 for p in a)


def test_head_output_is_unit_norm():
    z = Head(16, 8)(torch.randn(5, 16))
    assert z.shape == (5, 8)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-6)
