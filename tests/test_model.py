import numpy as np

from psospotter.metrics import _binary_auc
from psospotter.model import (
    _fit_logistic,
    _pick_C_for_target_sparsity,
    _stability_selection,
)


def test_fit_logistic_learns_separable_problem():
    rng = np.random.RandomState(1)
    n = 60
    y = np.repeat([0, 1], n // 2)
    X = rng.randn(n, 4)
    X[y == 1, 0] += 5
    clf = _fit_logistic(X, y, C=1, l1_ratio=0)
    proba = clf.predict_proba(X)[:, 1]
    assert proba.shape == (n,)
    assert _binary_auc(y, proba) > 0.95


def test_fit_logistic_is_reproducible():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 5)
    y = (X[:, 0] > 0).astype(int)
    a = _fit_logistic(X, y, C=1, l1_ratio=0.5, random_state=7).coef_
    b = _fit_logistic(X, y, C=1, l1_ratio=0.5, random_state=7).coef_
    np.testing.assert_array_equal(a, b)


def test_fit_logistic_fallback_succeeds():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = (X[:, 0] > 0).astype(int)
    clf = _fit_logistic(X, y, C=1, l1_ratio=0.5, fallback=True)
    assert clf.coef_.shape == (1, 3)


def test_pick_C_returns_value_from_grid():
    rng = np.random.RandomState(2)
    Xg = rng.randn(40, 12)
    y = np.repeat([0, 1], 20)
    Xg[y == 1, :3] += 3
    grid = (1e-3, 1e-2, 1e-1)
    C = _pick_C_for_target_sparsity(Xg, np.zeros((40, 2)), np.full(40, 0.5), y,
                                    np.arange(40), c_grid=grid, l1_ratio=0.9, target_nnz=3)
    assert C in grid


def test_stability_selection_ranks_signal_genes():
    rng = np.random.RandomState(3)
    n, G = 60, 10
    y = np.repeat([0, 1], n // 2)
    Xg = rng.randn(n, G)
    Xg[y == 1, :2] += 3  # genes 0,1 carry signal
    df = _stability_selection(Xg, np.zeros((n, 2)), np.full(n, 0.5), y, np.arange(n),
                              gene_names=np.array([f"g{i}" for i in range(G)], dtype=object),
                              C_sel=0.1, n_repeats=10, l1_ratio=0.9)
    assert df.shape[0] == G
    assert {"g0", "g1"} & set(df.head(4)["gene"])
    # sorted by selection_freq descending
    assert list(df["selection_freq"]) == sorted(df["selection_freq"], reverse=True)
