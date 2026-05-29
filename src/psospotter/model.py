"""Logistic-model helpers: regularized fitting, sparsity calibration, and
stability selection.

Mirror the scikit-learn ``LogisticRegression`` usage of the reference
(``fit_logistic_with_fallback``, ``pick_C_for_target_sparsity``,
``stability_selection``). The subsampling RNGs use the same seed offsets as the
reference (``RANDOM_STATE + 13`` and ``RANDOM_STATE + 7``). Unlike the
single-species originals, every ``saga`` fit here is given ``random_state`` so
the model steps are reproducible (a deliberate, documented change).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _assemble_design(
    Xg: np.ndarray, cov_age: np.ndarray, cov_sex: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Assemble the ``[genes | age dummies | sex]`` design matrix for given rows.

    Aim: build the feature matrix fed to the logistic model, matching the
    reference column order.

    Args:
        Xg: Cells-by-genes (z-scored) array.
        cov_age: Cells-by-age-dummy array.
        cov_sex: Per-cell sex vector.
        rows: Row indices (0-based) to assemble.

    Returns:
        A ``float32`` design array with ``len(rows)`` rows.
    """
    Xg_r = np.asarray(Xg[rows, :], dtype=np.float32)
    Xa = cov_age[rows, :].astype(np.float32, copy=False)
    Xs = cov_sex[rows].reshape(-1, 1).astype(np.float32, copy=False)
    return np.concatenate([Xg_r, Xa, Xs], axis=1)


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    l1_ratio: float = 0.0,
    tol: float = 1e-4,
    random_state: int = 42,
    fallback: bool = False,
) -> LogisticRegression:
    """Fit a balanced elastic-net logistic model.

    Aim: reproduce the reference final/selection fits and
    ``fit_logistic_with_fallback``. Uses ``solver="saga"``,
    ``penalty="elasticnet"``, ``class_weight="balanced"``, ``max_iter=8000``.

    Args:
        X: Feature matrix (rows = samples).
        y: 0/1 labels.
        C: Inverse regularization strength.
        l1_ratio: Elastic-net mixing (``0`` = ridge-like).
        tol: Optimizer tolerance.
        random_state: Seed for the ``saga`` solver.
        fallback: If ``True``, fall back to ``liblinear`` + L2 on failure
            (cross-species behavior).

    Returns:
        The fitted :class:`sklearn.linear_model.LogisticRegression`.

    Examples:
        >>> import numpy as np
        >>> X = np.random.RandomState(0).rand(40, 3)
        >>> y = (X[:, 0] > 0.5).astype(int)
        >>> _fit_logistic(X, y).coef_.shape
        (1, 3)
    """

    def _make() -> LogisticRegression:
        return LogisticRegression(
            max_iter=8000,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=float(l1_ratio),
            C=float(C),
            class_weight="balanced",
            tol=tol,
            random_state=random_state,
        )

    if not fallback:
        clf = _make()
        clf.fit(X, y)
        return clf
    try:
        clf = _make()
        clf.fit(X, y)
        return clf
    except Exception:
        clf = LogisticRegression(
            max_iter=8000,
            solver="liblinear",
            penalty="l2",
            C=float(C),
            class_weight="balanced",
            tol=tol,
            random_state=random_state,
        )
        clf.fit(X, y)
        return clf


def _subsample_rows(
    tr_rows: np.ndarray,
    groups: np.ndarray | None,
    rng: np.random.RandomState,
    subsample_frac_groups: float,
) -> np.ndarray:
    """Draw a (group-aware or cell-level) training sub-sample.

    Aim: reproduce the sub-sampling logic shared by the calibration and
    stability-selection steps.

    Args:
        tr_rows: Training row indices (0-based).
        groups: Per-cell group identifiers, or ``None`` for cell sub-sampling.
        rng: The seeded RNG to draw with.
        subsample_frac_groups: Fraction of groups to keep when group-aware.

    Returns:
        The sub-sampled row indices (0-based).
    """
    if groups is not None:
        g_tr = groups[tr_rows]
        ug = pd.Series(g_tr).unique().astype(object)
        n_take = max(2, int(round(subsample_frac_groups * ug.size)))
        n_take = min(n_take, ug.size)
        take = rng.choice(ug, size=n_take, replace=False)
        mask = np.isin(g_tr, take)
        return tr_rows[mask]
    n_take = max(50, int(round(0.75 * tr_rows.size)))
    n_take = min(n_take, tr_rows.size)  # cap for small inputs (reference assumes large data)
    return rng.choice(tr_rows, size=n_take, replace=False)


def _pick_C_for_target_sparsity(
    Xg: np.ndarray,
    cov_age: np.ndarray,
    cov_sex: np.ndarray,
    y: np.ndarray,
    tr_rows: np.ndarray,
    groups: np.ndarray | None = None,
    c_grid: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    l1_ratio: float = 0.9,
    target_nnz: int = 250,
    subsample_frac_groups: float = 0.75,
    random_state: int = 42,
) -> float:
    """Pick the regularization strength giving a target gene sparsity.

    Aim: reproduce ``pick_C_for_target_sparsity``. Fits an elastic-net logistic
    model on a sub-sample over a grid of ``C`` and returns the ``C`` whose count
    of nonzero gene coefficients is closest to ``target_nnz``.

    Args:
        Xg: Cells-by-genes (z-scored) array.
        cov_age: Age dummy matrix.
        cov_sex: Sex vector.
        y: 0/1 labels (length = cells).
        tr_rows: Training row indices (0-based).
        groups: Optional per-cell groups for group-aware sub-sampling.
        c_grid: Candidate ``C`` values.
        l1_ratio: Elastic-net mixing for selection.
        target_nnz: Target nonzero gene-coefficient count.
        subsample_frac_groups: Fraction of groups to sub-sample.
        random_state: Base seed; the sub-sample RNG uses ``random_state + 13``.

    Returns:
        The chosen ``C`` value.

    Raises:
        RuntimeError: If the calibration sub-sample contains a single class.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> Xg = rng.rand(60, 8); y = (Xg[:, 0] > 0.5).astype(int)
        >>> _pick_C_for_target_sparsity(Xg, np.zeros((60, 2)), np.full(60, 0.5),
        ...     y, np.arange(60), c_grid=(1e-2, 1e-1), target_nnz=2) in (1e-2, 1e-1)
        True
    """
    rng = np.random.RandomState(random_state + 13)
    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    rows_sub = _subsample_rows(tr_rows, groups, rng, subsample_frac_groups)

    y_sub = y[rows_sub]
    if np.unique(y_sub).size < 2:
        raise RuntimeError("Calibration subsample has only one class; check split.")

    n_gene = Xg.shape[1]
    X = _assemble_design(Xg, cov_age, cov_sex, rows_sub)

    best_C = float(c_grid[-1])
    best_diff = float("inf")
    for C in c_grid:
        clf = _fit_logistic(X, y_sub, C=C, l1_ratio=l1_ratio, tol=1e-3,
                            random_state=random_state)
        beta = clf.coef_.ravel()[:n_gene]
        nnz = int(np.sum(np.abs(beta) > 1e-8))
        diff = abs(nnz - int(target_nnz))
        if diff < best_diff:
            best_diff = diff
            best_C = float(C)
    return best_C


def _stability_selection(
    Xg: np.ndarray,
    cov_age: np.ndarray,
    cov_sex: np.ndarray,
    y: np.ndarray,
    tr_rows: np.ndarray,
    gene_names: np.ndarray,
    C_sel: float,
    groups: np.ndarray | None = None,
    n_repeats: int = 50,
    l1_ratio: float = 0.9,
    subsample_frac_groups: float = 0.75,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stability selection via repeated sub-sampled elastic-net fits.

    Aim: reproduce ``stability_selection``. Repeatedly sub-samples training
    cells, fits an elastic-net logistic model, and accumulates per-gene
    selection frequency, sign consistency, and mean (absolute) coefficient.

    Args:
        Xg: Cells-by-genes (z-scored) array.
        cov_age: Age dummy matrix.
        cov_sex: Sex vector.
        y: 0/1 labels.
        tr_rows: Training row indices (0-based).
        gene_names: Gene names (length ``Xg.shape[1]``).
        C_sel: Regularization strength used for every repeat.
        groups: Optional per-cell groups for group-aware sub-sampling.
        n_repeats: Number of sub-sampling repeats.
        l1_ratio: Elastic-net mixing.
        subsample_frac_groups: Fraction of groups to sub-sample.
        random_state: Base seed; the sub-sample RNG uses ``random_state + 7``.

    Returns:
        A DataFrame (one row per gene) with ``gene``, ``selection_freq``,
        ``sign_consistency``, ``mean_beta``, ``mean_abs_beta``, sorted by
        descending selection frequency then mean absolute coefficient.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> Xg = rng.rand(60, 5); y = (Xg[:, 0] > 0.5).astype(int)
        >>> df = _stability_selection(Xg, np.zeros((60, 2)), np.full(60, 0.5),
        ...     y, np.arange(60), [f"g{i}" for i in range(5)], C_sel=0.1, n_repeats=3)
        >>> list(df.columns)[:2]
        ['gene', 'selection_freq']
    """
    rng = np.random.RandomState(random_state + 7)
    tr_rows = np.asarray(tr_rows, dtype=np.int64)
    G = Xg.shape[1]
    freq = np.zeros(G, dtype=np.int32)
    sign_sum = np.zeros(G, dtype=np.int32)
    beta_sum = np.zeros(G, dtype=np.float64)
    beta_abs_sum = np.zeros(G, dtype=np.float64)

    use_groups = groups
    if groups is not None:
        if pd.Series(groups[tr_rows]).unique().size < 4:
            use_groups = None  # fall back to cell sub-sampling, as in the reference

    for _ in range(int(n_repeats)):
        rows_sub = _subsample_rows(tr_rows, use_groups, rng, subsample_frac_groups)
        y_sub = y[rows_sub]
        if np.unique(y_sub).size < 2:
            continue
        X = _assemble_design(Xg, cov_age, cov_sex, rows_sub)
        clf = _fit_logistic(X, y_sub, C=C_sel, l1_ratio=l1_ratio, tol=1e-3,
                            random_state=random_state)
        beta = clf.coef_.ravel()[:G]
        sel = np.abs(beta) > 1e-8
        freq[sel] += 1
        sign_sum[sel] += np.where(beta[sel] > 0, 1, -1)
        beta_sum[sel] += beta[sel].astype(np.float64)
        beta_abs_sum[sel] += np.abs(beta[sel]).astype(np.float64)

    freq_f = freq.astype(np.float64) / float(n_repeats)
    denom = np.maximum(freq, 1)
    mean_beta = np.where(freq > 0, beta_sum / denom, 0.0)
    mean_abs_beta = np.where(freq > 0, beta_abs_sum / denom, 0.0)
    sign_consistency = np.where(freq > 0, np.abs(sign_sum) / denom, 0.0)

    df = pd.DataFrame(
        {
            "gene": np.asarray(gene_names, dtype=object),
            "selection_freq": freq_f,
            "sign_consistency": sign_consistency,
            "mean_beta": mean_beta,
            "mean_abs_beta": mean_abs_beta,
        }
    ).sort_values(["selection_freq", "mean_abs_beta"], ascending=[False, False])
    return df.reset_index(drop=True)
