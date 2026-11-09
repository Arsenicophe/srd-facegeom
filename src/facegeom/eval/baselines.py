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


class CenteredLDA:
    """LDA ajustée sur l'entraînement, projection centrée sur la moyenne d'entraînement.

    Solveur « eigen » de scikit-learn, sans régularisation : aucun
    hyperparamètre à part la dimension de sortie. Ce solveur ne centre pas les
    données dans ``transform`` ; on le fait ici, comme l'ACP, car la similarité
    cosinus y est sensible.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CenteredLDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        n_classes = len(np.unique(y))
        if self.dim > n_classes - 1:
            raise ValueError(f"LDA : au plus {n_classes - 1} dimensions pour {n_classes} classes.")
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        lda = LinearDiscriminantAnalysis(solver="eigen", n_components=self.dim).fit(x, y)
        self.scalings_ = lda.scalings_[:, : self.dim]
        # Part de la séparation (rapport inter/intra) portée par les dimensions gardées
        self.explained_variance_ratio_ = lda.explained_variance_ratio_[: self.dim]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self.scalings_


def fit_lda(train: np.ndarray, labels: np.ndarray, dim: int) -> CenteredLDA:
    """LDA à ``dim`` dimensions ajustée sur les vecteurs d'entraînement normalisés L2."""
    from facegeom.eval.retrieval import l2_normalize

    return CenteredLDA(dim).fit(l2_normalize(train), labels)


def apply_projection(model, x: np.ndarray) -> np.ndarray:
    """Normalise L2 puis projette avec une ACP ou une LDA déjà ajustée."""
    from facegeom.eval.retrieval import l2_normalize

    return model.transform(l2_normalize(x)).astype(np.float32)
