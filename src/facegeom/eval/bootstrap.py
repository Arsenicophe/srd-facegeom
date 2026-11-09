"""Bootstrap sur les identités.

On tire les identités avec remise (pas les images : les images d'une même
personne ne sont pas indépendantes), et on recalcule la moyenne des valeurs
par requête des identités tirées. Une identité tirée deux fois compte deux fois.
"""

from __future__ import annotations

import numpy as np


def _per_identity_sums(values: np.ndarray, labels: np.ndarray):
    ids, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(ids))
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.float64)
    return sums, counts


def bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Intervalle de confiance de ``mean(a) − mean(b)``, en bootstrap sur les identités.

    ``values_a`` et ``values_b`` sont des valeurs par requête **pour les mêmes
    requêtes** (même ordre, mêmes ``labels``) : la comparaison est appariée.
    """
    values_a, values_b, labels = map(np.asarray, (values_a, values_b, labels))
    if not (len(values_a) == len(values_b) == len(labels)):
        raise ValueError("values_a, values_b et labels doivent avoir la même longueur.")
    sa, counts = _per_identity_sums(values_a, labels)
    sb, _ = _per_identity_sums(values_b, labels)
    n_id = len(counts)

    rng = np.random.default_rng(seed)
    # Poids de chaque identité dans chaque tirage (nombre de fois tirée)
    weights = rng.multinomial(n_id, np.full(n_id, 1 / n_id), size=n_boot).astype(np.float64)
    diffs = (weights @ sa - weights @ sb) / (weights @ counts)

    alpha = (1 - level) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return {
        "diff": float(values_a.mean() - values_b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
    }
