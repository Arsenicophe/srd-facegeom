"""Tests des métriques d'évaluation."""

import numpy as np
import pytest

from facegeom.eval.bootstrap import bootstrap_diff
from facegeom.eval.retrieval import ap_at_r, nearest_neighbors, retrieval_per_query, summarize
from facegeom.eval.verification import make_pairs, pair_scores, tar_at_far, verification_summary

# --- MAP@R : exemples chiffrés du tableau 3 de Musgrave et al. (2020) -----

@pytest.mark.parametrize(
    "hits, expected",
    [
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.10),  # seul le 1er est correct
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 0.12),  # le 1er et le 10e
        ([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 0.20),  # le 1er et le 2e
        ([1] * 10, 1.00),                        # les 10
    ],
)
def test_map_at_r_matches_musgrave_table3(hits, expected):
    assert ap_at_r(np.array(hits), r=10) == pytest.approx(expected, abs=1e-9)


def test_map_at_r_examples_from_our_explanation():
    # Exemples utilisés pour expliquer MAP@R (R = 4)
    assert ap_at_r(np.array([1, 0, 1, 1]), 4) == pytest.approx((1 + 2 / 3 + 3 / 4) / 4)
    assert ap_at_r(np.array([0, 1, 1, 1]), 4) == pytest.approx((1 / 2 + 2 / 3 + 3 / 4) / 4)


# --- Sur des vecteurs synthétiques ---------------------------------------

def _clusters(n_id=20, per_id=5, dim=16, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_id, dim))
    x = np.repeat(centers, per_id, axis=0) + spread * rng.normal(size=(n_id * per_id, dim))
    labels = np.repeat(np.arange(n_id), per_id)
    return x, labels


def test_separable_vectors_give_perfect_scores():
    x, labels = _clusters()
    s = summarize(retrieval_per_query(x, labels))
    assert s == {"p_at_1": 100.0, "r_precision": 100.0, "map_at_r": 100.0}
    pairs, y = make_pairs(labels, n_negative=2000)
    v = verification_summary(y, pair_scores(x, pairs))
    assert v["auc"] == pytest.approx(100.0)
    assert v["tar_at_far_0.01"] == pytest.approx(100.0)


def test_random_vectors_give_low_scores():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(50), 5)
    x = rng.normal(size=(len(labels), 32))
    s = summarize(retrieval_per_query(x, labels))
    assert s["map_at_r"] < 20 and s["p_at_1"] < 20


def test_query_is_never_its_own_neighbor():
    x, _ = _clusters(n_id=5, per_id=3)
    nn = nearest_neighbors(x / np.linalg.norm(x, axis=1, keepdims=True), k=3, chunk=4)
    assert not (nn == np.arange(len(x))[:, None]).any()


def test_singleton_identities_are_excluded():
    x, labels = _clusters(n_id=4, per_id=3)
    x = np.vstack([x, np.ones((1, x.shape[1]))])
    labels = np.append(labels, 99)  # identité avec une seule image
    pq = retrieval_per_query(x, labels)
    assert len(pq["map_at_r"]) == 12 and not pq["query_mask"][-1]


# --- Vérification ----------------------------------------------------------

def test_pairs_are_correct():
    labels = np.array([0, 0, 0, 1, 1, 2])
    pairs, y = make_pairs(labels, n_negative=50, seed=0)
    assert (y == 1).sum() == 3 + 1  # C(3,2) + C(2,2)
    assert (labels[pairs[y == 1, 0]] == labels[pairs[y == 1, 1]]).all()
    assert (labels[pairs[y == 0, 0]] != labels[pairs[y == 0, 1]]).all()
    p2, _ = make_pairs(labels, n_negative=50, seed=0)
    assert (pairs == p2).all()  # même graine, mêmes paires


def test_tar_at_far_on_constructed_scores():
    # 100 négatives : scores 0..99 ; positives : 50, 95, 98.5, 100.5
    y = np.array([0] * 100 + [1] * 4)
    s = np.concatenate([np.arange(100), [50, 95, 98.5, 100.5]]).astype(float)
    t = tar_at_far(y, s, [0.0, 0.02, 0.05])
    assert t[0.0] == pytest.approx(0.25)   # aucune négative tolérée : seul 100,5 passe
    assert t[0.02] == pytest.approx(0.50)  # 2 tolérées (98, 99) : + 98,5
    assert t[0.05] == pytest.approx(0.75)  # 5 tolérées (95..99) : + 95


def test_far_below_rule_of_three_is_not_reported():
    y = np.array([0] * 100 + [1] * 10)
    s = np.random.default_rng(0).random(110)
    v = verification_summary(y, s, fars=(0.1, 0.01))
    assert v["tar_at_far_0.1"] is not None
    assert v["tar_at_far_0.01"] is None  # 3/100 = 0,03 > 0,01 : non certifiable


# --- Bootstrap -------------------------------------------------------------

def test_bootstrap_identical_inputs_gives_zero():
    labels = np.repeat(np.arange(30), 4)
    v = np.random.default_rng(0).random(len(labels))
    b = bootstrap_diff(v, v, labels)
    assert b["diff"] == 0 and b["ci_low"] == 0 and b["ci_high"] == 0 and b["contains_zero"]


def test_bootstrap_detects_clear_difference():
    labels = np.repeat(np.arange(200), 5)
    rng = np.random.default_rng(0)
    a = rng.random(len(labels))
    b = bootstrap_diff(a + 0.1, a, labels)
    assert b["diff"] == pytest.approx(0.1)
    assert b["ci_low"] == pytest.approx(0.1) and not b["contains_zero"]


def test_bootstrap_coverage_on_pure_noise():
    # Sans vraie différence, l'IC à 95 % doit contenir 0 environ 95 % du temps.
    # (Un seul tirage ne prouve rien : 5 % des tirages l'excluent normalement.)
    labels = np.repeat(np.arange(200), 5)
    hits = sum(
        bootstrap_diff(*np.random.default_rng(k).random((2, len(labels))), labels,
                       n_boot=300, seed=k)["contains_zero"]
        for k in range(100)
    )
    assert 85 <= hits <= 100
