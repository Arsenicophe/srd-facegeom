"""Tests des références ACP et LDA."""

import numpy as np
import pytest

from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import retrieval_per_query, summarize


def _data(n_classes, per_class, seed):
    """L'identité vit dans 4 directions de faible variance ; 28 autres directions
    portent un bruit fort, commun à tous. L'ACP garde le bruit, la LDA l'identité."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, 4))
    ident = np.repeat(centers, per_class, axis=0) + 0.1 * rng.normal(size=(n_classes * per_class, 4))
    noise = 5.0 * rng.normal(size=(n_classes * per_class, 28))
    return np.hstack([ident, noise]) + 10.0, np.repeat(np.arange(n_classes), per_class)


def test_lda_generalizes_to_unseen_identities_better_than_pca():
    xtr, ytr = _data(60, 10, seed=0)
    xte, yte = _data(30, 10, seed=1)  # identités jamais vues
    lda = fit_lda(xtr, ytr, dim=4)
    pca = fit_pca(xtr, dim=4)
    m_lda = summarize(retrieval_per_query(apply_projection(lda, xte), yte))["map_at_r"]
    m_pca = summarize(retrieval_per_query(apply_projection(pca, xte), yte))["map_at_r"]
    # ~68 % contre ~1 % ici (la normalisation L2 avant projection, commune à tout
    # le pipeline, déforme un peu ces données synthétiques très bruitées)
    assert m_lda > 50 and m_lda > m_pca + 30


def test_lda_output_shape_and_centering():
    xtr, ytr = _data(20, 5, seed=0)
    lda = fit_lda(xtr, ytr, dim=3)
    z = apply_projection(lda, xtr)
    assert z.shape == (len(xtr), 3)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-6)  # centré sur l'entraînement


def test_lda_matches_scikit_learn_eigen_solver():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from facegeom.eval.baselines import CenteredLDA

    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(12), rng.integers(4, 9, size=12))  # classes de tailles inégales
    x = rng.normal(size=(len(y), 10)) + rng.normal(size=(12, 10))[y]
    ours = CenteredLDA(5).fit(x, y)
    ref = LinearDiscriminantAnalysis(solver="eigen", n_components=5).fit(x, y)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_[:5], rtol=1e-8)
    # Mêmes directions, au signe près
    for j in range(5):
        a, b = ours.scalings_[:, j], ref.scalings_[:, j]
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-8 * np.abs(b).max() * 100


def test_lda_dim_limited_by_number_of_classes():
    xtr, ytr = _data(5, 5, seed=0)
    with pytest.raises(ValueError):
        fit_lda(xtr, ytr, dim=5)  # au plus 4 dimensions pour 5 classes
