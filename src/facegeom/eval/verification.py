"""Vérification 1:1 : toutes les paires positives, échantillon fixe de négatives, score = cosinus."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from facegeom.eval.retrieval import l2_normalize


def make_pairs(labels: np.ndarray, n_negative: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(paires (M, 2), y)`` avec y = 1 pour une même personne."""
    labels = np.asarray(labels)
    pos = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        i, j = np.triu_indices(len(idx), k=1)
        pos.append(np.stack([idx[i], idx[j]], axis=1))
    pos = np.concatenate(pos)

    rng = np.random.default_rng(seed)
    neg = np.empty((0, 2), dtype=np.int64)
    while len(neg) < n_negative:
        a = rng.integers(0, len(labels), size=2 * n_negative)
        b = rng.integers(0, len(labels), size=2 * n_negative)
        ok = labels[a] != labels[b]
        neg = np.concatenate([neg, np.stack([a[ok], b[ok]], axis=1)])
    neg = neg[:n_negative]

    pairs = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    return pairs, y


def pair_scores(x: np.ndarray, pairs: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Similarité cosinus de chaque paire."""
    x = l2_normalize(x)
    out = np.empty(len(pairs), dtype=np.float32)
    for lo in range(0, len(pairs), chunk):
        p = pairs[lo : lo + chunk]
        out[lo : lo + chunk] = np.einsum("ij,ij->i", x[p[:, 0]], x[p[:, 1]])
    return out


def tar_at_far(y: np.ndarray, scores: np.ndarray, fars: list[float]) -> dict[float, float]:
    """TAR au seuil le plus bas dont le FAR ne dépasse pas la cible."""
    fpr, tpr, _ = roc_curve(y, scores)
    return {far: float(tpr[fpr <= far].max()) if (fpr <= far).any() else 0.0 for far in fars}


def eer(y: np.ndarray, scores: np.ndarray) -> float:
    """Taux d'égale erreur."""
    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def verification_summary(y: np.ndarray, scores: np.ndarray, fars: tuple[float, ...] = (1e-1, 1e-2, 1e-3)) -> dict:
    """AUC, EER et TAR aux FAR demandés, en pourcentage."""
    n_neg = int((y == 0).sum())
    out = {
        "auc": 100 * float(roc_auc_score(y, scores)),
        "eer": 100 * eer(y, scores),
        "n_positive": int((y == 1).sum()),
        "n_negative": n_neg,
    }
    for far, tar in tar_at_far(y, scores, list(fars)).items():
        out[f"tar_at_far_{far:g}"] = 100 * tar if far >= 3 / n_neg else None  # règle de trois
    return out
