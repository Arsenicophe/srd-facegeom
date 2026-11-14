"""Évaluation : alignement, uniformité, seuil choisi ailleurs, paires au hasard."""

import numpy as np
import pytest

from facegeom.eval.evaluate import alignment, evaluate, make_eval_pairs, random_pairs, uniformity
from facegeom.eval.verification import eer_threshold, rates_at_threshold

VCFG = {"n_negative": 2000, "seed": 0, "fars": [0.1, 0.01]}
UCFG = {"n_pairs": 2000, "seed": 0, "t": 2}


def _clusters(n_id=20, per_id=5, dim=16, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_id, dim))
    x = np.repeat(centers, per_id, axis=0) + spread * rng.normal(size=(n_id * per_id, dim))
    return x.astype(np.float32), np.repeat(np.arange(n_id), per_id)


def test_random_pairs_are_distinct_and_reproducible():
    p = random_pairs(50, 5000, seed=1)
    assert p.shape == (5000, 2) and (p[:, 0] != p[:, 1]).all()
    assert p.min() >= 0 and p.max() < 50
    assert (p == random_pairs(50, 5000, seed=1)).all()


def test_alignment_and_uniformity_known_values():
    # cos = 1 : distance nulle ; cos = 0 : ‖x − y‖² = 2 ; cos = −1 : 4
    assert alignment(np.array([1.0, 1.0])) == pytest.approx(0.0)
    assert alignment(np.array([0.0, -1.0])) == pytest.approx(3.0)
    assert uniformity(np.array([1.0, 1.0])) == pytest.approx(0.0)       # tout au même point
    assert uniformity(np.array([0.0, 0.0])) == pytest.approx(-4.0)      # log exp(−2 × 2)


def test_uniformity_is_lower_for_spread_out_vectors():
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(500, 16))
    packed = np.ones((500, 16)) + 0.05 * rng.normal(size=(500, 16))
    pairs = random_pairs(500, 5000)

    def unif(x):
        z = x / np.linalg.norm(x, axis=1, keepdims=True)
        return uniformity(np.einsum("ij,ij->i", z[pairs[:, 0]], z[pairs[:, 1]]))

    assert unif(spread) < unif(packed) - 1


def test_threshold_rates_on_constructed_scores():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.0])
    r = rates_at_threshold(y, s, 0.5)
    assert r["thr_tar"] == pytest.approx(75.0) and r["thr_far"] == pytest.approx(25.0)
    assert r["thr_balanced_acc"] == pytest.approx(75.0)
    assert 0.2 < eer_threshold(y, s) <= 0.7  # un seuil où FAR = FRR = 25 %


def test_evaluate_on_separable_vectors():
    x, labels = _clusters()
    ep = make_eval_pairs(labels, VCFG, UCFG)
    m, scores = evaluate(x, labels, ep)
    assert m["map_at_r"] == pytest.approx(100.0) and m["auc"] == pytest.approx(100.0)
    assert m["alignment"] < 0.01 and m["uniformity"] < -1
    assert "thr_balanced_acc" not in m
    thr = eer_threshold(ep["y"], scores)
    m2, _ = evaluate(x, labels, ep, threshold=thr)
    assert m2["thr_balanced_acc"] == pytest.approx(100.0)
