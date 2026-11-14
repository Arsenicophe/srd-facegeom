"""Statistiques sur les graines : IC de Student et comparaison appariée de deux pertes.

Ces intervalles ne couvrent que la variation due aux graines (protocole, section 9).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def student_ci(values, level: float = 0.95) -> dict[str, float]:
    """Moyenne et IC de Student d'une valeur par graine."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        raise ValueError("Il faut au moins 2 graines pour un intervalle de confiance.")
    mean, sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    half = stats.t.ppf((1 + level) / 2, df=len(v) - 1) * sem
    return {"mean": float(mean), "std": float(v.std(ddof=1)), "ci_low": float(mean - half),
            "ci_high": float(mean + half), "half_width": float(half), "n": len(v)}


def paired_diff(values_a, values_b, level: float = 0.95) -> dict[str, float]:
    """IC de ``mean(a − b)``, graine par graine ; ``a`` et ``b`` dans le même ordre de graines.

    Deux pertes sont déclarées différentes si l'intervalle exclut 0.
    """
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Les deux pertes doivent avoir les mêmes graines.")
    ci = student_ci(a - b, level)
    return {"diff": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "different": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0), "n": ci["n"]}
