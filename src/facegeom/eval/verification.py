"""Vérification : « ces deux images sont-elles de la même personne ? ».

Paires positives : **toutes** les paires d'images d'une même identité.
Paires négatives : un échantillon de taille fixe, tiré avec une graine fixe,
de paires d'identités différentes. Score d'une paire = similarité cosinus.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from facegeom.eval.retrieval import l2_normalize


def make_pairs(
    labels: np.ndarray, n_negative: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie ``(paires, étiquettes)`` : ``paires`` est (M, 2), étiquette 1 = même personne."""
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
    """Taux de vraies acceptations au taux de fausses acceptations donné.

    Pour chaque FAR, on prend le seuil le plus bas dont le FAR ne dépasse pas la
    cible, et on renvoie le TAR correspondant.
    """
    fpr, tpr, _ = roc_curve(y, scores)
    res = {}
    for far in fars:
        ok = fpr <= far
        res[far] = float(tpr[ok].max()) if ok.any() else 0.0
    return res


def eer(y: np.ndarray, scores: np.ndarray) -> float:
    """Taux d'égale erreur : point où fausses acceptations = faux rejets."""
    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def verification_summary(
    y: np.ndarray, scores: np.ndarray, fars: tuple[float, ...] = (1e-1, 1e-2, 1e-3)
) -> dict:
    """AUC, EER et TAR aux FAR demandés, en pourcentage."""
    n_neg = int((y == 0).sum())
    out = {
        "auc": 100 * float(roc_auc_score(y, scores)),
        "eer": 100 * eer(y, scores),
        "n_positive": int((y == 1).sum()),
        "n_negative": n_neg,
    }
    for far, tar in tar_at_far(y, scores, list(fars)).items():
        # Règle de trois : sous ~3/n, un FAR ne peut pas être certifié
        out[f"tar_at_far_{far:g}"] = 100 * tar if far >= 3 / n_neg else None
    return out
