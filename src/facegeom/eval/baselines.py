"""Références sans entraînement : ACP (brut réduit) et LDA, ajustées sur l'entraînement."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from facegeom.eval.retrieval import l2_normalize


def fit_pca(train: np.ndarray, dim: int, seed: int = 0) -> PCA:
    """ACP à ``dim`` composantes sur les vecteurs normalisés L2."""
    return PCA(n_components=dim, random_state=seed).fit(l2_normalize(train))


class CenteredLDA:
    """LDA équivalente à ``LinearDiscriminantAnalysis(solver="eigen")``, en plus rapide.

    Les matrices de dispersion viennent de deux produits de matrices au lieu
    d'une covariance par classe (7 689 classes en 2 048 dimensions). La
    projection est centrée sur la moyenne d'entraînement, car le solveur eigen
    ne le fait pas et le cosinus y est sensible. Équivalence testée dans
    tests/test_baselines.py.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def fit(self, x: np.ndarray, y: np.ndarray) -> CenteredLDA:
        from scipy.linalg import eigh

        classes, yi = np.unique(np.asarray(y), return_inverse=True)
        if self.dim > len(classes) - 1:
            raise ValueError(f"LDA : au plus {len(classes) - 1} dimensions pour {len(classes)} classes.")
        x = np.asarray(x, dtype=np.float64)
        n, d = x.shape
        self.mean_ = x.mean(axis=0)

        means = np.zeros((len(classes), d))
        np.add.at(means, yi, x)
        means /= np.bincount(yi)[:, None]

        xc = x - means[yi]
        sw = xc.T @ xc / n  # intra-classe (covariances biaisées, convention scikit-learn)
        xt = x - self.mean_
        sb = xt.T @ xt / n - sw  # inter = totale − intra

        evals, evecs = eigh(sb, sw)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        self.scalings_ = evecs[:, : self.dim]
        self.explained_variance_ratio_ = (evals / evals.sum())[: self.dim]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean_) @ self.scalings_


def fit_lda(train: np.ndarray, labels: np.ndarray, dim: int) -> CenteredLDA:
    """LDA à ``dim`` dimensions sur les vecteurs normalisés L2."""
    return CenteredLDA(dim).fit(l2_normalize(train), labels)


def apply_projection(model, x: np.ndarray) -> np.ndarray:
    """Normalise L2 puis projette (ACP ou LDA ajustée)."""
    return model.transform(l2_normalize(x)).astype(np.float32)
