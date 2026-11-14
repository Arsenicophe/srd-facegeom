"""Recherche par plus proches voisins : P@1, R-Precision, MAP@R (Musgrave et al. 2020, § 3.2).

Chaque image est requête, les références sont les autres images du même
ensemble. R = nombre d'autres images de la même identité. Les fonctions
renvoient des valeurs par requête (pour le bootstrap) ; la moyenne est la métrique.
"""

from __future__ import annotations

import numpy as np
import torch


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Lignes de longueur 1, en float32."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Vecteur nul : impossible de le normaliser.")
    return x / norms


def ap_at_r(hits: np.ndarray, r: int) -> float:
    """MAP@R d'une requête : (1/R) Σ_{i≤R} P(i) · hits[i]."""
    hits = np.asarray(hits[:r], dtype=np.float64)
    precision_at_i = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at_i * hits).sum() / r)


def nearest_neighbors(x: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Indices des ``k`` plus proches voisins (cosinus) de chaque ligne de ``x`` (normalisé), elle exclue."""
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    n = xt.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sim = xt[lo:hi] @ xt.T
        sim[torch.arange(hi - lo), torch.arange(lo, hi)] = -float("inf")
        out[lo:hi] = torch.topk(sim, k, dim=1).indices.numpy()
    return out


def retrieval_per_query(x: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """P@1, R-Precision et MAP@R par requête ; les identités à une seule image sont exclues."""
    x = l2_normalize(x)
    labels = np.asarray(labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    r = counts[inverse] - 1
    keep = r > 0
    nn = nearest_neighbors(x, int(r.max()))
    hits = labels[nn] == labels[:, None]

    rprec = np.array([hits[i, : r[i]].mean() if r[i] else np.nan for i in range(len(r))])
    mapr = np.array([ap_at_r(hits[i], r[i]) if r[i] else np.nan for i in range(len(r))])
    return {
        "p_at_1": hits[keep, 0].astype(np.float64),
        "r_precision": rprec[keep],
        "map_at_r": mapr[keep],
        "labels": labels[keep],
        "query_mask": keep,
    }


def summarize(per_query: dict[str, np.ndarray]) -> dict[str, float]:
    """Moyennes par requête, en pourcentage."""
    return {m: float(100 * per_query[m].mean()) for m in ["p_at_1", "r_precision", "map_at_r"]}
