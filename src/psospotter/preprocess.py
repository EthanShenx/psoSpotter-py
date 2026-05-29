"""Expression-matrix preprocessing.

In-memory replacements for the reference pipeline's on-disk ``zarr`` ``log1p``
layer and memmap z-score block. Convention: ``X`` is a cells-by-genes matrix
(rows = cells), matching the reference orientation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _normalize_log1p(X: sparse.spmatrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    """Library-size normalize and ``log1p``-transform a count matrix.

    Aim: reproduce ``_normalize_log1p_csr``. Each cell (row) is scaled so its
    counts sum to ``target_sum``, then ``log1p`` is applied (stored values cast
    to ``float32`` as in the reference). Cells with non-positive library size
    are left unscaled.

    Args:
        X: Cells-by-genes sparse matrix of raw counts.
        target_sum: Target total counts per cell after normalization.

    Returns:
        A :class:`scipy.sparse.csr_matrix` of normalized, ``log1p``-ed values.

    Examples:
        >>> from scipy import sparse
        >>> m = sparse.csr_matrix([[0.0, 2.0], [1.0, 3.0]])
        >>> _normalize_log1p(m, 100).toarray().round(4)
        array([[0.    , 4.6151],
               [3.2581, 4.3307]], dtype=float32)
    """
    X_csr = sparse.csr_matrix(X)
    lib = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float64)
    lib[lib <= 0] = 1.0
    scale = (float(target_sum) / lib).astype(np.float64)
    # Row scaling: equivalent to X_csr.multiply(scale[:, None]).
    X_norm = X_csr.multiply(scale[:, None]).tocsr()
    X_norm.data = np.log1p(X_norm.data).astype(np.float32, copy=False)
    return X_norm


def _compute_gene_mask(
    X: sparse.spmatrix, min_counts: float = 1, min_cells_frac: float = 0.01
) -> np.ndarray:
    """Compute a universal gene-keep mask from raw counts.

    Aim: reproduce the gene filter (``compute_gene_stats_on_rows`` + the
    ``gene_mask`` line). A gene is kept when its total raw count is at least
    ``min_counts`` and it is detected in at least
    ``max(1, int(n_cells * min_cells_frac))`` cells.

    Args:
        X: Cells-by-genes sparse matrix of raw counts (typically restricted to
            base/age-parsed cells).
        min_counts: Minimum summed raw count per gene.
        min_cells_frac: Minimum fraction of cells in which the gene is detected.

    Returns:
        A boolean array of length ``X.shape[1]``; ``True`` marks kept genes.

    Examples:
        >>> from scipy import sparse
        >>> m = sparse.csr_matrix([[0, 0, 5], [0, 1, 1]])
        >>> _compute_gene_mask(m, 1, 0.01).tolist()
        [False, True, True]
    """
    X_csr = sparse.csr_matrix(X)
    n_cells = X_csr.shape[0]
    gene_counts = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)
    gene_ncells = np.asarray((X_csr > 0).sum(axis=0)).ravel().astype(np.int64)
    min_cells = max(1, int(n_cells * float(min_cells_frac)))
    return (gene_counts >= float(min_counts)) & (gene_ncells >= min_cells)


def _train_feature_moments(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene mean and standard deviation on training rows.

    Aim: reproduce the train-only mean/std block used before z-scoring. Uses the
    population variance (divides by ``n``), floors variance at ``1e-12``, and
    casts to ``float32`` as in the reference.

    Args:
        X_train: Cells-by-genes dense array (training cells) of normalized
            ``log1p`` values.

    Returns:
        A tuple ``(mean, std)`` of ``float32`` arrays, each of length
        ``X_train.shape[1]``.

    Examples:
        >>> import numpy as np
        >>> mean, std = _train_feature_moments(np.array([[1.0], [2.0], [3.0]]))
        >>> float(mean[0])
        2.0
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    n_tr = float(Xt.shape[0])
    sum_x = Xt.sum(axis=0)
    sum_x2 = (Xt * Xt).sum(axis=0)
    mean = (sum_x / n_tr).astype(np.float32)
    var = (sum_x2 / n_tr) - (mean.astype(np.float64) ** 2)
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def _zscore_features(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray, z_clip: float = 5.0
) -> np.ndarray:
    """Z-score and clip features.

    Aim: reproduce the z-score + clip step applied when writing the reference
    memmap. Subtracts the per-gene mean, divides by the per-gene std, and (when
    ``z_clip > 0``) clips to ``[-z_clip, z_clip]``. Output is ``float32``.

    Args:
        X: Cells-by-genes dense array of normalized ``log1p`` values.
        mean: Per-gene means (length ``X.shape[1]``).
        std: Per-gene standard deviations (length ``X.shape[1]``).
        z_clip: Clipping bound; ``0`` or negative disables clipping.

    Returns:
        A ``float32`` array of z-scored (optionally clipped) values.

    Examples:
        >>> import numpy as np
        >>> _zscore_features(np.array([[0.0], [10.0], [100.0]]),
        ...                  np.array([10.0]), np.array([5.0]), 5.0).ravel().tolist()
        [-2.0, 0.0, 5.0]
    """
    Xb = (np.asarray(X, dtype=np.float64) - mean) / std
    if z_clip is not None and float(z_clip) > 0:
        np.clip(Xb, -float(z_clip), float(z_clip), out=Xb)
    return Xb.astype(np.float32)
