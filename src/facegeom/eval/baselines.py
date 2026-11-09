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
        """Même calcul que ``LinearDiscriminantAnalysis(solver="eigen")`` de
        scikit-learn, mais les matrices de dispersion sont obtenues par deux
        produits de matrices au lieu d'une covariance par classe (7 689 classes
        en 2 048 dimensions rendraient le calcul de scikit-learn très long).
        L'équivalence est vérifiée dans tests/test_baselines.py.
        """
        from scipy.linalg import eigh

        classes, yi = np.unique(np.asarray(y), return_inverse=True)
        n_classes = len(classes)
        if self.dim > n_classes - 1:
            raise ValueError(f"LDA : au plus {n_classes - 1} dimensions pour {n_classes} classes.")
        x = np.asarray(x, dtype=np.float64)
        n, d = x.shape
        self.mean_ = x.mean(axis=0)

        counts = np.bincount(yi).astype(np.float64)
        means = np.zeros((n_classes, d))
        np.add.at(means, yi, x)
        means /= counts[:, None]

        # Dispersion intra-classe : Σ_k (n_k/n) · cov_k avec cov_k biaisée
        # (convention scikit-learn), ce qui se simplifie en xcᵀ xc / n
        xc = x - means[yi]
        sw = xc.T @ xc / n
        # Dispersion totale (covariance biaisée), inter = totale − intra
        xt = x - self.mean_
        st = xt.T @ xt / n
        sb = st - sw

        evals, evecs = eigh(sb, sw)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        self.scalings_ = evecs[:, : self.dim]
        self.explained_variance_ratio_ = (evals / evals.sum())[: self.dim]
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
