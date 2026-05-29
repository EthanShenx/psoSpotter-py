import numpy as np
import pytest

from psospotter.selection import (
    _gene_correlation,
    _prune_by_abs_corr_greedy,
    _supervised_effect_score,
)


def test_effect_score_high_for_separated_gene():
    rng = np.random.RandomState(1)
    n = 40
    y = np.repeat([0, 1], n // 2)
    X = rng.randn(n, 3)
    X[y == 1, 0] += 10  # gene 0 strongly separated
    s = _supervised_effect_score(X, y, min_per_class=1)
    assert s.shape == (3,)
    assert s[0] > s[1] and s[0] > s[2]
    assert np.all(s >= 0)


def test_effect_score_too_few_per_class():
    X = np.random.RandomState(0).randn(10, 2)
    y = np.array([0] * 9 + [1])
    with pytest.raises(RuntimeError):
        _supervised_effect_score(X, y, min_per_class=5)


def test_gene_correlation_valid_matrix_and_detects_collinearity():
    rng = np.random.RandomState(2)
    X = rng.randn(20, 5)
    X[:, 1] = X[:, 0]  # perfectly correlated
    corr = _gene_correlation(X, np.arange(20), np.arange(5), max_cells=20)
    assert corr.shape == (5, 5)
    np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-5)
    assert np.all(corr >= -1) and np.all(corr <= 1)
    assert corr[0, 1] > 0.99


def test_gene_correlation_subsample_idx_override():
    rng = np.random.RandomState(3)
    X = rng.randn(30, 4)
    c1 = _gene_correlation(X, np.arange(30), np.arange(4), subsample_idx=np.arange(10))
    c2 = _gene_correlation(X, np.arange(10), np.arange(4), max_cells=100)
    np.testing.assert_array_equal(c1, c2)


def test_prune_keeps_higher_scoring_of_correlated_pair():
    corr = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    scores = np.array([0.2, 0.9, 0.5])
    kept, names, mapping = _prune_by_abs_corr_greedy(
        np.array(["g1", "g2", "g3"], dtype=object), scores, corr, thr=0.9, min_kept=1)
    assert "g2" in names and "g3" in names and "g1" not in names


def test_prune_force_keep_top_when_too_aggressive():
    corr = np.ones((4, 4))
    scores = np.array([0.1, 0.4, 0.9, 0.6])
    kept, names, _ = _prune_by_abs_corr_greedy(
        np.array([f"g{i}" for i in range(4)], dtype=object), scores, corr,
        thr=0.9, min_kept=3)
    assert kept.size == 3
    assert set(names.tolist()) == {"g2", "g3", "g1"}  # top-3 by score
