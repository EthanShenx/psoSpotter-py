import numpy as np
import pytest
from scipy import sparse

from psospotter.preprocess import (
    _compute_gene_mask,
    _normalize_log1p,
    _train_feature_moments,
    _zscore_features,
)


def test_normalize_log1p_scales_rows_then_log1p():
    m = sparse.csr_matrix(np.array([[0.0, 2.0], [1.0, 3.0]]))
    res = _normalize_log1p(m, target_sum=100).toarray()
    np.testing.assert_allclose(res[0], [np.log1p(0), np.log1p(100)], rtol=1e-6)
    np.testing.assert_allclose(res[1], [np.log1p(25), np.log1p(75)], rtol=1e-6)


def test_normalize_log1p_zero_library_row_unscaled():
    m = sparse.csr_matrix(np.array([[0.0, 0.0], [5.0, 5.0]]))
    res = _normalize_log1p(m, target_sum=100).toarray()
    np.testing.assert_array_equal(res[0], [0.0, 0.0])


def test_compute_gene_mask_thresholds():
    m = sparse.csr_matrix(np.array([[0, 0, 5], [0, 1, 1]]))
    mask = _compute_gene_mask(m, min_counts=1, min_cells_frac=0.01)
    np.testing.assert_array_equal(mask, [False, True, True])


def test_compute_gene_mask_single_cell():
    m = sparse.csr_matrix(np.array([[0, 3]]))
    mask = _compute_gene_mask(m, min_counts=1, min_cells_frac=0.01)
    np.testing.assert_array_equal(mask, [False, True])


def test_train_feature_moments_population_variance():
    m = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    mean, std = _train_feature_moments(m)
    np.testing.assert_allclose(mean, [2.0, 5.0])
    np.testing.assert_allclose(std, [np.sqrt(2 / 3)] * 2, rtol=1e-6)


def test_train_feature_moments_variance_floor():
    mean, std = _train_feature_moments(np.array([[7.0], [7.0], [7.0]]))
    np.testing.assert_allclose(std, [np.sqrt(1e-12)], rtol=1e-6)


def test_zscore_features_center_scale_clip():
    z = _zscore_features(np.array([[0.0], [10.0], [100.0]]),
                         np.array([10.0]), np.array([5.0]), z_clip=5.0)
    np.testing.assert_allclose(z.ravel(), [-2.0, 0.0, 5.0], rtol=1e-6)


def test_zscore_features_no_clip():
    z = _zscore_features(np.array([[100.0]]), np.array([0.0]), np.array([1.0]), z_clip=0)
    assert z[0, 0] == pytest.approx(100.0)
