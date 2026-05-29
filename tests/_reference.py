"""Reference implementation of the original psoSpotter algorithm (in-memory).

This module re-derives the original algorithm directly from the reference
scripts (``human.py`` / ``mouse.py``), independently of the ``psospotter``
package, replacing the zarr/h5ad I/O with dense in-memory operations. It is used
only by the integration test to prove the refactored package produces identical
outputs.

The one intentional deviation from the literal originals: every ``saga``
``LogisticRegression`` is given ``random_state`` (the originals omit it, which
makes their model steps non-reproducible). The package does the same, so
end-to-end outputs match bit-for-bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ---- deterministic numeric helpers (copied from the reference scripts) ----


def ref_normalize_log1p(X: sparse.spmatrix, target_sum: float) -> sparse.csr_matrix:
    X_csr = sparse.csr_matrix(X)
    lib = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float64)
    lib[lib <= 0] = 1.0
    scale = (float(target_sum) / lib).astype(np.float64)
    X_norm = X_csr.multiply(scale[:, None]).tocsr()
    X_norm.data = np.log1p(X_norm.data).astype(np.float32, copy=False)
    return X_norm


def ref_gene_mask(X: sparse.spmatrix, min_counts: float, min_cells_frac: float) -> np.ndarray:
    X_csr = sparse.csr_matrix(X)
    n_cells = X_csr.shape[0]
    gene_counts = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)
    gene_ncells = np.asarray((X_csr > 0).sum(axis=0)).ravel().astype(np.int64)
    min_cells = max(1, int(n_cells * float(min_cells_frac)))
    return (gene_counts >= float(min_counts)) & (gene_ncells >= min_cells)


def ref_effect_score(X: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    tr0 = np.where(y == 0)[0]
    tr1 = np.where(y == 1)[0]
    X0 = X[tr0, :].astype(np.float64)
    X1 = X[tr1, :].astype(np.float64)
    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    var0 = (X0 * X0).mean(axis=0) - mean0 * mean0
    var1 = (X1 * X1).mean(axis=0) - mean1 * mean1
    var0[var0 < 0] = 0.0
    var1[var1 < 0] = 0.0
    denom = np.sqrt(0.5 * (var0 + var1) + float(eps))
    return (np.abs(mean1 - mean0) / denom).astype(np.float64)


def ref_corr(X: np.ndarray, rows: np.ndarray, gene_idx: np.ndarray, max_cells: int,
             random_state: int) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    rows = np.asarray(rows, dtype=np.int64)
    sel = rng.choice(rows, size=max_cells, replace=False) if rows.size > max_cells else rows
    sel = np.sort(sel)
    Xb = X[sel, :][:, gene_idx].astype(np.float64)
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


def ref_prune(gene_names: np.ndarray, scores: np.ndarray, corr: np.ndarray, thr: float,
              min_kept: int) -> np.ndarray:
    n = scores.size
    order = np.argsort(scores)[::-1]
    covered = np.zeros(n, dtype=bool)
    kept = []
    abs_corr = np.abs(corr)
    for pos in order:
        if covered[pos]:
            continue
        kept.append(int(pos))
        m = abs_corr[pos, :] >= float(thr)
        m[pos] = True
        covered[m] = True
    kept_arr = np.array(kept, dtype=np.int32)
    if kept_arr.size < min_kept:
        kept_arr = np.argsort(scores)[::-1][: min(min_kept, n)].astype(np.int32)
    return kept_arr


def ref_moments(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xt = X_tr.astype(np.float64)
    n = float(Xt.shape[0])
    mean = (Xt.sum(axis=0) / n).astype(np.float32)
    var = (((Xt * Xt).sum(axis=0)) / n) - (mean.astype(np.float64) ** 2)
    var[var < 1e-12] = 1e-12
    std = np.sqrt(var).astype(np.float32)
    return mean, std


def ref_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray, z_clip: float) -> np.ndarray:
    Xb = (X.astype(np.float64) - mean) / std
    if z_clip and z_clip > 0:
        np.clip(Xb, -z_clip, z_clip, out=Xb)
    return Xb.astype(np.float32)


def ref_group_split(y: np.ndarray, groups: np.ndarray, test_size: float,
                    random_state: int, max_tries: int = 300) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    idx_all = np.arange(y.size, dtype=np.int64)
    df = pd.DataFrame({"i": idx_all, "y": y, "g": g})
    g2y = df.groupby("g")["y"].first()
    g_list = g2y.index.to_numpy(dtype=object)
    g_y = g2y.to_numpy(dtype=np.int8)
    g0 = g_list[g_y == 0]
    g1 = g_list[g_y == 1]
    n_test0 = max(1, int(round(test_size * g0.size)))
    n_test1 = max(1, int(round(test_size * g1.size)))
    if g0.size > 1:
        n_test0 = min(n_test0, g0.size - 1)
    if g1.size > 1:
        n_test1 = min(n_test1, g1.size - 1)
    rng = np.random.RandomState(random_state)
    for _ in range(max_tries):
        te_g0 = rng.choice(g0, size=n_test0, replace=False)
        te_g1 = rng.choice(g1, size=n_test1, replace=False)
        te_groups = set(te_g0.tolist() + te_g1.tolist())
        te_mask = df["g"].isin(te_groups).to_numpy()
        te = df.loc[te_mask, "i"].to_numpy(dtype=np.int64)
        tr = df.loc[~te_mask, "i"].to_numpy(dtype=np.int64)
        if np.unique(y[tr]).size == 2 and np.unique(y[te]).size == 2:
            return tr, te
    raise RuntimeError("split failed")


def _ref_subsample(tr_rows: np.ndarray, groups: Any, rng: np.random.RandomState,
                   frac: float) -> np.ndarray:
    if groups is not None:
        g_tr = groups[tr_rows]
        ug = pd.Series(g_tr).unique().astype(object)
        n_take = max(2, int(round(frac * ug.size)))
        n_take = min(n_take, ug.size)
        take = rng.choice(ug, size=n_take, replace=False)
        return tr_rows[np.isin(g_tr, take)]
    n_take = max(50, int(round(0.75 * tr_rows.size)))
    n_take = min(n_take, tr_rows.size)
    return rng.choice(tr_rows, size=n_take, replace=False)


def _ref_fit(X: np.ndarray, y: np.ndarray, C: float, l1_ratio: float, tol: float,
             random_state: int) -> LogisticRegression:
    clf = LogisticRegression(max_iter=8000, solver="saga", penalty="elasticnet",
                             l1_ratio=l1_ratio, C=C, class_weight="balanced", tol=tol,
                             random_state=random_state)
    clf.fit(X, y)
    return clf


def _ref_design(Xg: np.ndarray, age: np.ndarray, sex: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [Xg[rows, :].astype(np.float32), age[rows, :].astype(np.float32),
         sex[rows].reshape(-1, 1).astype(np.float32)], axis=1
    )


def ref_pick_C(Xz: np.ndarray, age: np.ndarray, sex: np.ndarray, y: np.ndarray,
               tr: np.ndarray, groups: Any, c_grid: tuple[float, ...], l1_ratio: float,
               target_nnz: int, frac: float, random_state: int) -> float:
    rng = np.random.RandomState(random_state + 13)
    rows_sub = _ref_subsample(tr, groups, rng, frac)
    n_gene = Xz.shape[1]
    X = _ref_design(Xz, age, sex, rows_sub)
    y_sub = y[rows_sub]
    best_C, best_diff = float(c_grid[-1]), float("inf")
    for C in c_grid:
        clf = _ref_fit(X, y_sub, C, l1_ratio, 1e-3, random_state)
        nnz = int(np.sum(np.abs(clf.coef_.ravel()[:n_gene]) > 1e-8))
        if abs(nnz - target_nnz) < best_diff:
            best_diff = abs(nnz - target_nnz)
            best_C = float(C)
    return best_C


def ref_stability(Xz: np.ndarray, age: np.ndarray, sex: np.ndarray, y: np.ndarray,
                  tr: np.ndarray, gene_names: np.ndarray, C_sel: float, groups: Any,
                  n_repeats: int, l1_ratio: float, frac: float,
                  random_state: int) -> pd.DataFrame:
    rng = np.random.RandomState(random_state + 7)
    G = Xz.shape[1]
    freq = np.zeros(G, dtype=np.int32)
    sign_sum = np.zeros(G, dtype=np.int32)
    beta_sum = np.zeros(G, dtype=np.float64)
    beta_abs_sum = np.zeros(G, dtype=np.float64)
    use_groups = groups
    if groups is not None and pd.Series(groups[tr]).unique().size < 4:
        use_groups = None
    for _ in range(n_repeats):
        rows_sub = _ref_subsample(tr, use_groups, rng, frac)
        if np.unique(y[rows_sub]).size < 2:
            continue
        X = _ref_design(Xz, age, sex, rows_sub)
        clf = _ref_fit(X, y[rows_sub], C_sel, l1_ratio, 1e-3, random_state)
        beta = clf.coef_.ravel()[:G]
        sel = np.abs(beta) > 1e-8
        freq[sel] += 1
        sign_sum[sel] += np.where(beta[sel] > 0, 1, -1)
        beta_sum[sel] += beta[sel]
        beta_abs_sum[sel] += np.abs(beta[sel])
    denom = np.maximum(freq, 1)
    df = pd.DataFrame({
        "gene": np.asarray(gene_names, dtype=object),
        "selection_freq": freq.astype(np.float64) / n_repeats,
        "sign_consistency": np.where(freq > 0, np.abs(sign_sum) / denom, 0.0),
        "mean_beta": np.where(freq > 0, beta_sum / denom, 0.0),
        "mean_abs_beta": np.where(freq > 0, beta_abs_sum / denom, 0.0),
    }).sort_values(["selection_freq", "mean_abs_beta"], ascending=[False, False])
    return df.reset_index(drop=True)


def reference_fit_panel(X: sparse.spmatrix, obs: pd.DataFrame, var_names: np.ndarray,
                        cfg: Any) -> dict[str, Any]:
    """Run the original algorithm end-to-end on in-memory data.

    Mirrors the orchestration of ``human.py`` / ``mouse.py`` ``main()`` and
    returns every intermediate plus the final result, for comparison against
    :func:`psospotter.fit_panel_model`.
    """
    # base cells via age parsing (uses the config's parser/binner, which are
    # unit-tested separately)
    age_val = [cfg.age_parser(a) for a in obs[cfg.age_col].tolist()]
    age_bin = np.array([cfg.age_binner(v) for v in age_val], dtype=object)
    base = np.where(age_bin != None)[0].astype(np.int64)  # noqa: E711
    X_csr = sparse.csr_matrix(X)
    X_base = X_csr[base, :]
    obs_base = obs.iloc[base].reset_index(drop=True)
    age_bin_base = pd.Series(age_bin[base]).reset_index(drop=True)

    gene_mask = ref_gene_mask(X_base, cfg.min_gene_counts, cfg.min_gene_cells_frac)
    X_log = ref_normalize_log1p(X_base, cfg.target_sum)[:, gene_mask]
    var_kept = var_names[gene_mask]

    group_col = None
    for c in cfg.group_col_candidates:
        if c in obs.columns:
            group_col = c
            break

    cond = obs_base[cfg.cond_col].astype(str)
    y_series = cond.map(cfg.cond_map)
    mask = ~y_series.isna()
    task_rows = np.where(mask.to_numpy())[0].astype(np.int64)
    y = y_series[mask].astype(np.int8).to_numpy()
    age_bin_task = age_bin_base[mask].reset_index(drop=True)
    sex_task = obs_base[cfg.sex_col][mask].reset_index(drop=True)
    groups = (obs_base[group_col][mask].to_numpy(dtype=object) if group_col else None)
    X_task = np.asarray(X_log[task_rows, :].todense())

    age_dum = pd.get_dummies(age_bin_task, prefix="", prefix_sep="")
    for c in cfg.age_bin_cols:
        if c not in age_dum.columns:
            age_dum[c] = 0
    age_mat = age_dum[cfg.age_bin_cols].to_numpy(dtype=np.float32)
    sex_num = sex_task.apply(
        lambda v: 1.0 if str(v).strip().lower() in ("male", "m", "♂", "1")
        else (0.0 if str(v).strip().lower() in ("female", "f", "♀", "0") else 0.5)
    ).to_numpy(dtype=np.float32)

    if groups is not None and cfg.use_group_split:
        tr, te = ref_group_split(y, groups, cfg.test_size, cfg.random_state)
        groups_arr = groups
    else:
        # not exercised by the integration cases (all have groups)
        raise NotImplementedError

    score = ref_effect_score(X_task[tr, :], y[tr], cfg.cand_eps)
    corr = ref_corr(X_task, tr, np.arange(X_task.shape[1]), cfg.max_corr_cells, cfg.random_state)
    kept = ref_prune(var_kept, score, corr, cfg.prune_abs_corr, cfg.min_pruned_genes)
    kept_names = var_kept[kept]
    X_pruned = X_task[:, kept]
    mean, std = ref_moments(X_pruned[tr, :])
    Xz = ref_zscore(X_pruned, mean, std, cfg.z_clip)

    C_sel = ref_pick_C(Xz, age_mat, sex_num, y, tr, groups_arr, cfg.c_grid,
                       cfg.l1_ratio_sel, cfg.target_nnz_for_selection,
                       cfg.subsample_frac_groups, cfg.random_state)
    df_stab = ref_stability(Xz, age_mat, sex_num, y, tr, kept_names, C_sel, groups_arr,
                            cfg.n_repeats, cfg.l1_ratio_sel, cfg.subsample_frac_groups,
                            cfg.random_state)

    panel_genes = df_stab.head(cfg.panel_k)["gene"].to_numpy(dtype=object)
    panel_cols = np.where(np.isin(kept_names.astype(object), panel_genes))[0].astype(np.int32)

    Xg = Xz[:, panel_cols]
    X_tr = _ref_design(Xg, age_mat, sex_num, tr)
    X_te = _ref_design(Xg, age_mat, sex_num, te)
    clf = _ref_fit(X_tr, y[tr], cfg.final_c, cfg.final_l1_ratio, 1e-4, cfg.random_state)
    proba_te = clf.predict_proba(X_te)[:, 1]
    pred_te = clf.predict(X_te)

    beta = clf.coef_.ravel().astype(float)
    intercept = float(clf.intercept_.ravel()[0])

    return {
        "gene_mask": gene_mask,
        "var_names": var_kept,
        "X_log": np.asarray(X_log.todense()),
        "score": score,
        "corr": corr,
        "kept_pos": kept,
        "kept_names": kept_names,
        "Xz": Xz,
        "train": tr,
        "test": te,
        "C_sel": C_sel,
        "stability": df_stab,
        "panel": list(panel_genes.astype(str)),
        "beta": beta,
        "intercept": intercept,
        "test_auc": float(roc_auc_score(y[te], proba_te)),
        "test_acc": float(accuracy_score(y[te], pred_te)),
        "test_proba": proba_te,
    }
