"""Statistiques sur les graines : IC de Student, comparaison appariée."""

import numpy as np
import pytest
from scipy import stats

from facegeom.eval.stats import paired_diff, student_ci


def test_student_ci_on_hand_computed_example():
    # 5 graines : moyenne 15, écart-type 0,158 ; t(0,975 ; 4) = 2,776
    ci = student_ci([14.8, 14.9, 15.0, 15.1, 15.2])
    assert ci["mean"] == pytest.approx(15.0)
    assert ci["std"] == pytest.approx(0.15811, abs=1e-5)
    assert ci["half_width"] == pytest.approx(2.776445 * 0.15811 / np.sqrt(5), abs=1e-4)
    assert ci["ci_low"] == pytest.approx(15.0 - ci["half_width"])
    assert ci["n"] == 5


def test_student_ci_matches_scipy():
    v = np.random.default_rng(0).normal(10, 2, size=5)
    lo, hi = stats.t.interval(0.95, df=4, loc=v.mean(), scale=stats.sem(v))
    ci = student_ci(v)
    assert (ci["ci_low"], ci["ci_high"]) == pytest.approx((lo, hi))


def test_student_ci_needs_two_values():
    with pytest.raises(ValueError):
        student_ci([15.0])


def test_paired_diff_cancels_common_noise():
    # Le bruit commun aux deux pertes (graine) est grand, l'écart est constant : 0,3
    noise = np.array([-1.0, 0.5, 2.0, -0.7, 0.1])
    a, b = 15.3 + noise, 15.0 + noise + np.array([0.01, -0.01, 0.0, 0.02, -0.02])
    d = paired_diff(a, b)
    assert d["diff"] == pytest.approx(0.3, abs=0.01)
    assert d["different"] and d["ci_low"] > 0.25
    # Sans appariement, les deux IC se chevauchent largement
    assert student_ci(a)["ci_low"] < student_ci(b)["ci_high"]


def test_paired_diff_identical_losses_are_not_different():
    v = [15.0, 15.2, 14.9, 15.1, 15.0]
    d = paired_diff(v, v)
    assert d["diff"] == 0 and not d["different"]


def test_paired_diff_is_antisymmetric_and_checks_shape():
    a, b = [15.0, 15.4, 15.1], [14.0, 14.9, 14.2]
    ab, ba = paired_diff(a, b), paired_diff(b, a)
    assert ab["diff"] == pytest.approx(-ba["diff"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"])
    with pytest.raises(ValueError):
        paired_diff(a, b[:2])


def test_coverage_on_pure_noise():
    # Deux pertes identiques en loi : l'IC à 95 % doit contenir 0 environ 95 % du temps
    rng = np.random.default_rng(0)
    hits = sum(not paired_diff(*rng.normal(15, 0.3, size=(2, 5)))["different"] for _ in range(2000))
    assert 0.93 <= hits / 2000 <= 0.97
