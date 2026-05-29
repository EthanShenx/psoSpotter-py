"""Pipeline orchestration and public entry points.

:func:`fit_panel_model` mirrors the ``main()`` of ``human.py`` / ``mouse.py``;
:func:`fit_cross_species_panel` mirrors the ``main()`` of the
``*_train_*_test.py`` scripts. The reference scripts' on-disk I/O (h5ad strip,
zarr build, memmap, chunked CSR reads) is replaced by in-memory operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .config import PsoSpotterConfig
from .metadata import _build_covariates
from .metrics import _aggregate_sample_predictions, _binary_auc, _classification_report
from .model import (
    _fit_logistic,
    _pick_C_for_target_sparsity,
    _stability_selection,
)
from .ortholog import _load_ortholog_candidates
from .preprocess import (
    _compute_gene_mask,
    _normalize_log1p,
    _train_feature_moments,
    _zscore_features,
)
from .selection import (
    _gene_correlation,
    _prune_by_abs_corr_greedy,
    _supervised_effect_score,
)
from .splits import _stratified_group_split, _stratified_random_split


def _default_var_names(X: sparse.spmatrix, var_names: Any | None) -> np.ndarray:
    """Return gene names, defaulting to ``Gene0..Gene{n-1}`` when not supplied."""
    if var_names is None:
        return np.array([f"Gene{i}" for i in range(X.shape[1])], dtype=object)
    arr = np.asarray(var_names, dtype=object)
    if arr.size != X.shape[1]:
        raise ValueError("`var_names` length must equal the number of genes (columns of X).")
    return arr


def _prepare_species(
    X: sparse.spmatrix, obs: pd.DataFrame, var_names: np.ndarray, config: PsoSpotterConfig
) -> dict[str, Any]:
    """Resolve base cells, the kept-gene log1p matrix, and the group column.

    Aim: reproduce the base-cell selection (age parsing), universal gene mask,
    and normalized ``log1p`` matrix shared by both entry points.

    Args:
        X: Cells-by-genes raw counts.
        obs: Per-cell metadata aligned to ``X`` rows.
        var_names: Gene names (length = ``X.shape[1]``).
        config: Pipeline configuration.

    Returns:
        A dict with ``base`` indices, ``obs_base``, ``age_bin_base``, the kept
        ``X_log`` (CSR), ``var_names`` (kept), and ``group_col``.

    Raises:
        KeyError: If the age column is missing from ``obs``.
        RuntimeError: If no cells survive age parsing.
    """
    if config.age_col not in obs.columns:
        raise KeyError(f"Missing '{config.age_col}' in obs.")

    age_val = [config.age_parser(a) for a in obs[config.age_col].tolist()]
    age_bin = np.array([config.age_binner(v) for v in age_val], dtype=object)
    base = np.where(age_bin != None)[0].astype(np.int64)  # noqa: E711 (object array vs None)
    if base.size == 0:
        raise RuntimeError("No cells survived age parsing.")

    X_csr = sparse.csr_matrix(X)
    X_base = X_csr[base, :]
    obs_base = obs.iloc[base].reset_index(drop=True)
    age_bin_base = pd.Series(age_bin[base]).reset_index(drop=True)

    gene_mask = _compute_gene_mask(X_base, config.min_gene_counts, config.min_gene_cells_frac)
    X_log_full = _normalize_log1p(X_base, config.target_sum)
    X_log = X_log_full[:, gene_mask]
    var_kept = var_names[gene_mask]

    group_col = None
    for c in config.group_col_candidates:
        if c in obs.columns:
            group_col = c
            break

    return {
        "base": base,
        "obs_base": obs_base,
        "age_bin_base": age_bin_base,
        "X_log": X_log,
        "var_names": var_kept,
        "group_col": group_col,
    }


def _build_task(prep: dict[str, Any], config: PsoSpotterConfig) -> dict[str, Any]:
    """Map conditions to 0/1 labels and restrict to task cells.

    Aim: reproduce ``build_task_from_universal_base`` (in-memory). Selects cells
    whose condition maps via ``cond_map`` and assembles their dense feature
    matrix and covariate inputs.

    Args:
        prep: Output of :func:`_prepare_species`.
        config: Pipeline configuration.

    Returns:
        A dict with ``task_rows`` (into base), ``y``, ``age_bin``, ``sex``,
        ``groups`` (or ``None``), and the dense ``X_task`` (task cells x kept genes).

    Raises:
        KeyError: If the condition column is missing from ``obs``.
    """
    obs_base = prep["obs_base"]
    if config.cond_col not in obs_base.columns:
        raise KeyError(f"Missing '{config.cond_col}' in obs for task.")

    cond = obs_base[config.cond_col].astype(str)
    y_series = cond.map(config.cond_map)
    mask = ~y_series.isna()
    task_rows = np.where(mask.to_numpy())[0].astype(np.int64)
    y = y_series[mask].astype(np.int8).to_numpy()

    age_bin_task = prep["age_bin_base"][mask].reset_index(drop=True)
    sex_task = obs_base[config.sex_col][mask].reset_index(drop=True)
    groups_task = (
        obs_base[prep["group_col"]][mask].astype(object).reset_index(drop=True)
        if prep["group_col"] is not None
        else None
    )
    X_task = np.asarray(prep["X_log"][task_rows, :].todense())

    return {
        "task_rows": task_rows,
        "y": y,
        "age_bin": age_bin_task,
        "sex": sex_task,
        "groups": None if groups_task is None else groups_task.to_numpy(dtype=object),
        "X_task": X_task,
    }


def _fit_eval_panel(
    Xz: np.ndarray,
    age_mat: np.ndarray,
    sex_num: np.ndarray,
    y: np.ndarray,
    tr_i: np.ndarray,
    te_i: np.ndarray,
    panel_cols: np.ndarray,
    config: PsoSpotterConfig,
    y_train_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit the final model on ``[panel | age | sex]`` and evaluate train/test.

    Aim: reproduce the final-model fit and metric computation.

    Args:
        Xz: Z-scored gene matrix (cells x pruned genes).
        age_mat: Age dummy matrix.
        sex_num: Sex vector.
        y: 0/1 labels.
        tr_i: Train row indices.
        te_i: Test row indices.
        panel_cols: Column indices (into ``Xz``) of the selected panel.
        config: Pipeline configuration.
        y_train_override: Optional replacement training labels (for the
            label-shuffle control).

    Returns:
        A dict with the fitted ``model`` and a ``metrics`` sub-dict.
    """
    Xg = Xz[:, panel_cols]
    Xa_tr, Xa_te = age_mat[tr_i, :], age_mat[te_i, :]
    Xs_tr = sex_num[tr_i].reshape(-1, 1)
    Xs_te = sex_num[te_i].reshape(-1, 1)
    X_tr = np.concatenate([Xg[tr_i, :], Xa_tr, Xs_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([Xg[te_i, :], Xa_te, Xs_te], axis=1).astype(np.float32)

    y_tr = y[tr_i] if y_train_override is None else y_train_override
    y_te = y[te_i]
    clf = _fit_logistic(X_tr, y_tr, C=config.final_c, l1_ratio=config.final_l1_ratio,
                        tol=1e-4, random_state=config.random_state)

    proba_tr = clf.predict_proba(X_tr)[:, 1]
    proba_te = clf.predict_proba(X_te)[:, 1]
    pred_tr = clf.predict(X_tr)
    pred_te = clf.predict(X_te)

    metrics = {
        "train_acc": float(accuracy_score(y_tr, pred_tr)),
        "train_auc": _binary_auc(y_tr, proba_tr),
        "test_acc": float(accuracy_score(y_te, pred_te)),
        "test_auc": _binary_auc(y_te, proba_te),
        "train_report": _classification_report(y_tr, pred_tr),
        "test_report": _classification_report(y_te, pred_te),
        "test_proba": proba_te,
    }
    return {"model": clf, "metrics": metrics}


def _choose_sample_cv(
    groups: np.ndarray, y: np.ndarray, n_splits: int, random_state: int
) -> tuple[str, Any, int] | None:
    """Choose a sample-level CV splitter (LOSO for few groups, else group k-fold).

    Aim: reproduce ``choose_sample_cv_splitter``.

    Args:
        groups: Per-cell group identifiers.
        y: 0/1 labels.
        n_splits: Requested k-fold splits.
        random_state: Seed for the k-fold splitter.

    Returns:
        A tuple ``(mode, splitter, n_splits)`` or ``None`` if CV is infeasible.
    """
    df = pd.DataFrame(
        {"group": np.asarray(groups, dtype=object), "y": np.asarray(y, dtype=np.int8)})
    g2y = df.groupby("group")["y"].first()
    n_groups = int(g2y.shape[0])
    n_pos = int(np.sum(g2y.to_numpy(dtype=np.int8) == 1))
    n_neg = int(np.sum(g2y.to_numpy(dtype=np.int8) == 0))
    if min(n_pos, n_neg) < 2:
        return None
    if n_groups < 30:
        return ("loso", LeaveOneGroupOut(), n_groups)
    k = min(int(n_splits), min(n_pos, n_neg))
    if k < 2:
        return None
    return ("kfold", StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state), k)


def _run_sample_level_cv(
    Xg_panel: np.ndarray,
    age_mat: np.ndarray,
    sex_num: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: PsoSpotterConfig,
    n_splits: int = 5,
) -> dict[str, Any] | None:
    """Sample-level cross-validation with the fixed panel.

    Aim: reproduce ``run_sample_level_cv_fixed_panel``. Cross-validates over
    groups and aggregates predictions to the sample level.

    Args:
        Xg_panel: Z-scored panel-gene matrix (cells x panel).
        age_mat: Age dummy matrix.
        sex_num: Sex vector.
        y: 0/1 labels.
        groups: Per-cell group identifiers.
        config: Pipeline configuration.
        n_splits: K-fold splits for the group-k-fold path.

    Returns:
        A dict with ``mode``, ``predictions`` (DataFrame), ``acc``, ``auc`` and
        ``report``; or ``None`` if CV is infeasible.
    """
    choice = _choose_sample_cv(groups, y, n_splits, config.random_state)
    if choice is None:
        return None
    mode, splitter, _ = choice

    X_all = np.concatenate(
        [np.asarray(Xg_panel, dtype=np.float32), age_mat.astype(np.float32),
         sex_num.reshape(-1, 1).astype(np.float32)],
        axis=1,
    )
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups, dtype=object)

    if mode == "loso":
        split_iter = splitter.split(X_all, y, groups)
    else:
        split_iter = splitter.split(X_all, y, groups=groups)

    recs = []
    for tr_idx, te_idx in split_iter:
        if np.unique(y[tr_idx]).size < 2:
            continue
        clf = _fit_logistic(X_all[tr_idx], y[tr_idx], C=config.final_c,
                            l1_ratio=config.final_l1_ratio, tol=1e-4,
                            random_state=config.random_state)
        sc = clf.predict_proba(X_all[te_idx])[:, 1]
        recs.append(pd.DataFrame({"group": groups[te_idx], "y_true": y[te_idx], "cell_score": sc}))
    if not recs:
        return None

    cell = pd.concat(recs, ignore_index=True)
    samp = _aggregate_sample_predictions(
        cell["group"].to_numpy(dtype=object),
        cell["y_true"].to_numpy(dtype=np.int8),
        cell["cell_score"].to_numpy(dtype=np.float64),
    )
    return {
        "mode": mode,
        "predictions": samp,
        "acc": float(accuracy_score(samp["y_true"], samp["y_pred"])),
        "auc": _binary_auc(samp["y_true"].to_numpy(), samp["sample_score"].to_numpy()),
        "report": _classification_report(samp["y_true"].to_numpy(), samp["y_pred"].to_numpy()),
    }


def fit_panel_model(
    X: sparse.spmatrix,
    obs: pd.DataFrame,
    config: PsoSpotterConfig,
    var_names: Any | None = None,
) -> dict[str, Any]:
    """Fit a single-species biomarker panel model.

    Aim: run the full within-species pipeline (mirrors ``main()`` in
    ``human.py`` / ``mouse.py``): metadata parsing, gene filtering,
    normalization, supervised effect scoring, correlation pruning, z-scoring,
    sparsity calibration, stability selection, top-K panel construction, and a
    final regularized logistic fit with age/sex covariates, plus evaluation
    (and optional sample-level CV / label-shuffle control).

    Args:
        X: Cells-by-genes matrix of raw counts (a :mod:`scipy.sparse` matrix).
        obs: Per-cell metadata DataFrame aligned to ``X`` rows; must contain the
            age, sex and condition columns named in ``config``.
        config: A :class:`~psospotter.PsoSpotterConfig`.
        var_names: Optional gene names (length = ``X.shape[1]``); defaults to
            ``Gene0..Gene{n-1}``.

    Returns:
        A dict with ``panel`` (gene names), ``coefficients`` (DataFrame incl.
        intercept), ``metrics`` (train/test accuracy & AUC, reports, optional
        ``sample_cv`` / ``label_shuffle``), ``stability`` table, ``pruning``
        (kept names + map), ``calibration_C``, ``split`` indices, ``model``, and
        ``var_names``.

    Raises:
        RuntimeError: If too few task cells remain or panel selection fails.

    Examples:
        >>> from psospotter import simulate_data, psospotter_config, fit_panel_model
        >>> s = simulate_data(n_cells=200, n_genes=40, seed=1)
        >>> cfg = psospotter_config("human", min_pruned_genes=6, panel_k=6,
        ...     n_repeats=5, target_nnz_for_selection=6, effect_min_per_class=5)
        >>> res = fit_panel_model(s["X"], s["obs"], cfg, s["var_names"])
        >>> len(res["panel"])
        6
    """
    var_names = _default_var_names(X, var_names)
    prep = _prepare_species(X, obs, var_names, config)
    task = _build_task(prep, config)
    if task["task_rows"].size < 2:
        raise RuntimeError("Too few task cells after condition mapping.")

    y = task["y"]
    age_mat, sex_num = _build_covariates(task["age_bin"], task["sex"], config.age_bin_cols)

    if task["groups"] is not None and config.use_group_split:
        tr_i, te_i = _stratified_group_split(y, task["groups"], config.test_size,
                                             random_state=config.random_state)
        groups_arr = task["groups"]
    else:
        tr_i, te_i = _stratified_random_split(y, config.test_size, random_state=config.random_state)
        groups_arr = None

    X_task = task["X_task"]
    score_all = _supervised_effect_score(X_task[tr_i, :], y[tr_i], eps=config.cand_eps,
                                         min_per_class=config.effect_min_per_class)
    corr = _gene_correlation(X_task, tr_i, np.arange(X_task.shape[1]),
                             max_cells=config.max_corr_cells, random_state=config.random_state)
    kept_pos, kept_names, pmap = _prune_by_abs_corr_greedy(
        prep["var_names"], score_all, corr, thr=config.prune_abs_corr,
        min_kept=config.min_pruned_genes)

    X_pruned = X_task[:, kept_pos]
    mean, std = _train_feature_moments(X_pruned[tr_i, :])
    Xz = _zscore_features(X_pruned, mean, std, config.z_clip)

    C_sel = _pick_C_for_target_sparsity(
        Xz, age_mat, sex_num, y, tr_i, groups=groups_arr, c_grid=config.c_grid,
        l1_ratio=config.l1_ratio_sel, target_nnz=config.target_nnz_for_selection,
        subsample_frac_groups=config.subsample_frac_groups, random_state=config.random_state)
    df_stab = _stability_selection(
        Xz, age_mat, sex_num, y, tr_i, gene_names=kept_names, C_sel=C_sel,
        groups=groups_arr, n_repeats=config.n_repeats, l1_ratio=config.l1_ratio_sel,
        subsample_frac_groups=config.subsample_frac_groups, random_state=config.random_state)

    # NOTE (faithful to the reference): the panel genes are taken in stability
    # order for the coefficient labels, but the model columns are selected in
    # pruned order via np.isin; when the two orders differ the reference labels
    # the columns the same way, which we mirror exactly.
    panel_genes = df_stab.head(config.panel_k)["gene"].to_numpy(dtype=object)
    panel_mask = np.isin(kept_names.astype(object), panel_genes)
    panel_cols = np.where(panel_mask)[0].astype(np.int32)
    if panel_cols.size == 0:
        raise RuntimeError("Panel selection failed: no genes selected.")

    fit = _fit_eval_panel(Xz, age_mat, sex_num, y, tr_i, te_i, panel_cols, config)
    metrics = dict(fit["metrics"])

    metrics["sample_cv"] = None
    if config.run_sample_level_cv and groups_arr is not None:
        metrics["sample_cv"] = _run_sample_level_cv(
            Xz[:, panel_cols], age_mat, sex_num, y, groups_arr, config)

    metrics["label_shuffle"] = None
    if config.run_label_shuffle:
        rng = np.random.RandomState(config.label_shuffle_seed)
        y_tr_shuf = rng.permutation(y[tr_i])
        shuf = _fit_eval_panel(Xz, age_mat, sex_num, y, tr_i, te_i, panel_cols, config,
                               y_train_override=y_tr_shuf)
        metrics["label_shuffle"] = shuf["metrics"]

    beta = fit["model"].coef_.ravel().astype(float)
    intercept = float(fit["model"].intercept_.ravel()[0])
    feat_names = list(panel_genes.astype(str)) + list(config.age_bin_cols) + ["sex"]
    coef_df = pd.DataFrame({"feature": feat_names + ["Intercept"],
                            "beta": list(beta) + [intercept]})

    return {
        "panel": list(panel_genes.astype(str)),
        "coefficients": coef_df,
        "metrics": metrics,
        "stability": df_stab,
        "pruning": {"kept_names": kept_names, "map": pmap},
        "calibration_C": C_sel,
        "split": {"train": tr_i, "test": te_i},
        "model": fit["model"],
        "var_names": prep["var_names"],
    }


def fit_cross_species_panel(
    X_train: sparse.spmatrix,
    obs_train: pd.DataFrame,
    X_test: sparse.spmatrix,
    obs_test: pd.DataFrame,
    ortholog: pd.DataFrame,
    config_train: PsoSpotterConfig,
    config_test: PsoSpotterConfig,
    var_names_train: Any | None = None,
    var_names_test: Any | None = None,
    train_is_human: bool = True,
) -> dict[str, Any]:
    """Fit a cross-species panel model (train one species, test another).

    Aim: run the cross-species pipeline (mirrors the ``*_train_*_test.py``
    scripts). Selects 1:1 orthologs, builds the panel on the training species in
    ortholog-gene space, fits the final model, and validates it on the test
    species (mapping panel genes through the ortholog table and reusing the
    training z-score moments).

    Args:
        X_train, X_test: Cells-by-genes raw-count matrices per species.
        obs_train, obs_test: Per-cell metadata for each species.
        ortholog: DataFrame with ``human_gene``/``mouse_gene`` (+ optional
            ``orthology_type``) columns.
        config_train, config_test: Cross-mode configs for each species.
        var_names_train, var_names_test: Optional gene names per species.
        train_is_human: Whether the training species corresponds to the
            ``human_gene`` column of ``ortholog`` (set ``False`` for mouse-train).

    Returns:
        A dict with ``panel`` (DataFrame mapping train/test gene names),
        ``internal`` (held-out train-species metrics), ``external`` (test-species
        metrics), the fitted ``model``, and the z-score ``moments`` reused on the
        test species.

    Raises:
        RuntimeError: If too few ortholog genes remain or panel selection fails.

    Examples:
        >>> # See tests/test_pipeline.py for a small end-to-end example.
    """
    var_names_train = _default_var_names(X_train, var_names_train)
    var_names_test = _default_var_names(X_test, var_names_test)

    prep_tr = _prepare_species(X_train, obs_train, var_names_train, config_train)
    prep_te = _prepare_species(X_test, obs_test, var_names_test, config_test)

    orth = ortholog.copy()
    if not train_is_human:
        orth = orth.rename(columns={"human_gene": "mouse_gene", "mouse_gene": "human_gene"})
    _, human2mouse, candidate_pos = _load_ortholog_candidates(
        prep_tr["var_names"], prep_te["var_names"], orth, drop_mt_rp=config_train.drop_mt_rp)
    if candidate_pos.size < config_train.min_pruned_genes:
        raise RuntimeError("Too few ortholog genes remain after filtering.")

    task = _build_task(prep_tr, config_train)
    y = task["y"]
    age_mat, sex_num = _build_covariates(task["age_bin"], task["sex"], config_train.age_bin_cols)

    if task["groups"] is not None and config_train.use_group_split:
        tr_i, te_i = _stratified_group_split(y, task["groups"], config_train.test_size,
                                             random_state=config_train.random_state)
        groups_arr = task["groups"]
    else:
        tr_i, te_i = _stratified_random_split(y, config_train.test_size,
                                              random_state=config_train.random_state)
        groups_arr = None

    X_task = task["X_task"]
    score_all = _supervised_effect_score(X_task[tr_i, :], y[tr_i], eps=config_train.cand_eps,
                                         min_per_class=config_train.effect_min_per_class)
    cand_scores = score_all[candidate_pos]
    corr = _gene_correlation(X_task, tr_i, candidate_pos, max_cells=config_train.max_corr_cells,
                             random_state=config_train.random_state)
    pruned_local, pruned_names, _ = _prune_by_abs_corr_greedy(
        prep_tr["var_names"][candidate_pos], cand_scores, corr,
        thr=config_train.prune_abs_corr, min_kept=config_train.min_pruned_genes)
    pruned_global = candidate_pos[pruned_local]

    X_pruned = X_task[:, pruned_global]
    mean, std = _train_feature_moments(X_pruned[tr_i, :])
    Xz = _zscore_features(X_pruned, mean, std, config_train.z_clip)

    C_sel = _pick_C_for_target_sparsity(
        Xz, age_mat, sex_num, y, tr_i, groups=groups_arr, c_grid=config_train.c_grid,
        l1_ratio=config_train.l1_ratio_sel, target_nnz=config_train.target_nnz_for_selection,
        subsample_frac_groups=config_train.subsample_frac_groups,
        random_state=config_train.random_state)
    df_stab = _stability_selection(
        Xz, age_mat, sex_num, y, tr_i, gene_names=pruned_names, C_sel=C_sel,
        groups=groups_arr, n_repeats=config_train.n_repeats, l1_ratio=config_train.l1_ratio_sel,
        subsample_frac_groups=config_train.subsample_frac_groups,
        random_state=config_train.random_state)

    panel_train = df_stab.head(config_train.panel_k)["gene"].to_numpy(dtype=object)
    panel_mask = np.isin(pruned_names.astype(object), panel_train)
    panel_cols = np.where(panel_mask)[0].astype(np.int32)
    if panel_cols.size == 0:
        raise RuntimeError("Panel selection failed: no genes selected.")
    panel_test_genes = np.array([human2mouse[str(g)] for g in panel_train], dtype=object)

    fit = _fit_eval_panel(Xz, age_mat, sex_num, y, tr_i, te_i, panel_cols, config_train)

    # External validation on the test species.
    task_te = _build_task(prep_te, config_test)
    age_te, sex_te = _build_covariates(
        task_te["age_bin"], task_te["sex"], config_test.age_bin_cols)
    te_names = list(prep_te["var_names"])
    test_pos = np.array([te_names.index(g) for g in panel_test_genes], dtype=np.int64)
    panel_mean = mean[panel_cols]
    panel_std = std[panel_cols]

    Xg_ext = task_te["X_task"][:, test_pos]
    Xg_ext = _zscore_features(Xg_ext, panel_mean, panel_std, config_train.z_clip)
    X_ext = np.concatenate([Xg_ext, age_te.astype(np.float32),
                            sex_te.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)
    proba_ext = fit["model"].predict_proba(X_ext)[:, 1]
    pred_ext = (proba_ext >= 0.5).astype(np.int8)
    y_ext = task_te["y"]

    external = {
        "acc": float(accuracy_score(y_ext, pred_ext)),
        "auc": _binary_auc(y_ext, proba_ext),
        "report": _classification_report(y_ext, pred_ext),
        "proba": proba_ext,
    }

    return {
        "panel": pd.DataFrame({"train_gene": panel_train.astype(str),
                               "test_gene": panel_test_genes.astype(str)}),
        "internal": fit["metrics"],
        "external": external,
        "model": fit["model"],
        "moments": {"mean": panel_mean, "std": panel_std},
        "stability": df_stab,
        "split": {"train": tr_i, "test": te_i},
    }
