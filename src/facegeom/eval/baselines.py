"""Références sans entraînement de perte : ACP (mesure brute réduite) et LDA.

Les projections sont **ajustées sur l'entraînement** et appliquées ensuite à
d'autres identités (validation, puis test à la toute fin).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(train: np.ndarray, dim: int, seed: int = 0) -> PCA:
    """ACP à ``dim`` composantes ajustée sur les vecteurs d'entraînement normalisés L2.

    On normalise d'abord, comme pour la mesure brute : l'ACP réduit donc
    l'espace sur lequel la similarité cosinus est calculée.
    """
    from facegeom.eval.retrieval import l2_normalize

    return PCA(n_components=dim, random_state=seed).fit(l2_normalize(train))


def apply_projection(model, x: np.ndarray) -> np.ndarray:
    """Normalise L2 puis projette avec une ACP ou une LDA déjà ajustée."""
    from facegeom.eval.retrieval import l2_normalize

    return model.transform(l2_normalize(x)).astype(np.float32)
