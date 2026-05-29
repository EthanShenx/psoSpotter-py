"""Feature-selection helpers: supervised effect scoring, gene-gene correlation,
and greedy correlation pruning.

Mirror ``supervised_effect_score_all_genes``, ``corr_on_train_subsample_subset``
and ``prune_by_abs_corr_greedy`` from the reference scripts. The reference reads
its matrix from a chunked ``zarr`` CSR store; here the equivalent dense in-memory
indexing is used, which is numerically identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _supervised_effect_score(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-8,
    min_per_class: int = 50,
) -> np.ndarray:
    """Per-gene supervised effect score (standardized mean difference).

    Aim: reproduce ``supervised_effect_score_all_genes``. For each gene,
    computes ``|mean1 - mean0|`` divided by the pooled standard deviation, using
    population variances.

    Args:
        X: Cells-by-genes dense array (training cells) of normalized ``log1p``
            values, in the same row order as ``y``.
        y: 0/1 class labels, one per row of ``X``.
        eps: Variance-stabilizing epsilon under the square root.
        min_per_class: Minimum cells required in each class (reference uses 50).

    Returns:
        A ``float64`` array of length ``X.shape[1]`` with per-gene effect scores.

    Raises:
        RuntimeError: If either class has fewer than ``min_per_class`` cells.

    Examples:
        >>> import numpy as np
        >>> X = np.array([[0.0], [0.0], [5.0], [5.0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> float(_supervised_effect_score(X, y, min_per_class=1)[0]) > 0
        True
    """
    y = np.asarray(y, dtype=np.int8)
    tr0 = np.where(y == 0)[0]
    tr1 = np.where(y == 1)[0]
    if tr0.size < min_per_class or tr1.size < min_per_class:
        raise RuntimeError("Too few train cells in one class for effect score.")

    X0 = np.asarray(X[tr0, :], dtype=np.float64)
    X1 = np.asarray(X[tr1, :], dtype=np.float64)
    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    var0 = (X0 * X0).mean(axis=0) - mean0 * mean0
    var1 = (X1 * X1).mean(axis=0) - mean1 * mean1
    var0[var0 < 0] = 0.0
    var1[var1 < 0] = 0.0

    denom = np.sqrt(0.5 * (var0 + var1) + float(eps))
    return (np.abs(mean1 - mean0) / denom).astype(np.float64)


def _gene_correlation(
    X: np.ndarray,
    rows: np.ndarray,
    gene_idx: np.ndarray,
    max_cells: int = 15000,
    random_state: int = 42,
    subsample_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Gene-gene correlation over a (sub-sampled) set of training cells.

    Aim: reproduce ``corr_on_train_subsample_subset``, with the same variance
    floor and clipping/NaN handling, including the ``float32`` output cast.

    Args:
        X: Cells-by-genes dense array of normalized ``log1p`` values.
        rows: Training row indices (0-based) to draw from.
        gene_idx: Gene column indices (0-based) to include.
        max_cells: Maximum cells to use; a subsample is drawn if exceeded.
        random_state: Seed for the subsample RNG (reference uses ``RANDOM_STATE``).
        subsample_idx: Optional explicit row indices (0-based, into the original
            matrix) overriding the random subsample, for exact parity.

    Returns:
        A ``float32`` correlation matrix of shape ``(len(gene_idx), len(gene_idx))``,
        clipped to ``[-1, 1]`` with NaNs set to 0 and a unit diagonal.

    Examples:
        >>> import numpy as np
        >>> X = np.random.RandomState(0).rand(20, 5)
        >>> _gene_correlation(X, np.arange(20), np.arange(5)).shape
        (5, 5)
    """
    rng = np.random.RandomState(random_state)
    rows = np.asarray(rows, dtype=np.int64)
    if subsample_idx is not None:
        sel = np.asarray(subsample_idx, dtype=np.int64)
    elif rows.size > max_cells:
        sel = rng.choice(rows, size=max_cells, replace=False)
    else:
        sel = rows
    sel = np.sort(sel)

    gene_idx = np.asarray(gene_idx, dtype=np.int64)
    Xb = np.asarray(X[sel, :][:, gene_idx], dtype=np.float64)
    n = float(sel.size)
    mean = Xb.sum(axis=0) / n
    mean2 = (Xb * Xb).sum(axis=0) / n
    var = mean2 - mean * mean
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var)

    Exy = (Xb.T @ Xb) / n
    cov = Exy - np.outer(mean, mean)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def _prune_by_abs_corr_greedy(
    gene_names: np.ndarray,
    scores: np.ndarray,
    corr: np.ndarray,
    thr: float = 0.90,
    min_kept: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Greedy correlation pruning of redundant genes.

    Aim: reproduce ``prune_by_abs_corr_greedy``. Walks genes in descending score
    order, keeping each not-yet-covered gene and marking every gene with
    ``|corr| >= thr`` to it as covered. Falls back to the top ``min_kept`` by
    score if too few survive.

    Args:
        gene_names: Gene names (length = number of genes).
        scores: Per-gene scores (same length as ``gene_names``).
        corr: Square correlation matrix from :func:`_gene_correlation`.
        thr: Absolute-correlation redundancy threshold.
        min_kept: Minimum number of genes to retain.

    Returns:
        A tuple ``(kept_pos, kept_names, mapping)`` where ``kept_pos`` is a
        ``int32`` array of 0-based retained positions (in selection order),
        ``kept_names`` is the corresponding names, and ``mapping`` is a list of
        per-representative dicts describing absorbed genes.

    Examples:
        >>> import numpy as np
        >>> corr = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        >>> kept, names, _ = _prune_by_abs_corr_greedy(
        ...     np.array(["g1", "g2", "g3"]), np.array([0.2, 0.9, 0.5]), corr,
        ...     thr=0.9, min_kept=1)
        >>> sorted(names.tolist())
        ['g2', 'g3']
    """
    gene_names = np.asarray(gene_names, dtype=object)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    order = np.argsort(scores)[::-1]
    covered = np.zeros(n, dtype=bool)
    abs_corr = np.abs(corr)
    thr = float(thr)

    kept_pos: list[int] = []
    mapping: list[dict[str, Any]] = []
    for pos in order:
        if covered[pos]:
            continue
        kept_pos.append(int(pos))
        m = abs_corr[pos, :] >= thr
        m[pos] = True
        covered[m] = True
        removed = np.where(m)[0]
        removed = removed[removed != pos]
        mapping.append(
            {
                "representative": str(gene_names[pos]),
                "rep_score": float(scores[pos]),
                "cluster_size": int(1 + removed.size),
                "removed_genes": ",".join(str(gene_names[j]) for j in removed),
            }
        )

    kept_arr = np.array(kept_pos, dtype=np.int32)
    if kept_arr.size < min_kept:
        kept_arr = np.argsort(scores)[::-1][: min(min_kept, n)].astype(np.int32)

    kept_names = gene_names[kept_arr]
    return kept_arr, kept_names, mapping
