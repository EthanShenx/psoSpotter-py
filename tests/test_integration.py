"""End-to-end parity: the refactored package must produce identical outputs to
an independent reference implementation of the original algorithm (``_reference``)
for 3+ input cases. Because both run the same NumPy/scikit-learn calls with the
same seeds, equality is exact (up to floating-point representation)."""

import numpy as np
import pytest

from _reference import reference_fit_panel
from psospotter import fit_panel_model, psospotter_config, simulate_data
from psospotter.pipeline import _build_task, _default_var_names, _prepare_species
from psospotter.preprocess import (
    _compute_gene_mask,
    _train_feature_moments,
    _zscore_features,
)
from psospotter.selection import (
    _gene_correlation,
    _prune_by_abs_corr_greedy,
    _supervised_effect_score,
)
from psospotter.splits import _stratified_group_split

CASES = [
    dict(n_cells=200, n_genes=40, n_groups=8, seed=1),
    dict(n_cells=240, n_genes=50, n_groups=10, seed=7),
    dict(n_cells=160, n_genes=36, n_groups=6, seed=13),
]


def _cfg():
    return psospotter_config(
        "human", min_pruned_genes=6, panel_k=6, n_repeats=8,
        target_nnz_for_selection=6, effect_min_per_class=5, max_corr_cells=10**6)


@pytest.mark.parametrize("case", CASES)
def test_end_to_end_identical_to_reference(case):
    s = simulate_data(species="human", **case)
    cfg = _cfg()
    pkg = fit_panel_model(s["X"], s["obs"], cfg, s["var_names"])
    ref = reference_fit_panel(s["X"], s["obs"], s["var_names"], cfg)

    np.testing.assert_array_equal(pkg["var_names"], ref["var_names"])
    np.testing.assert_array_equal(pkg["split"]["train"], ref["train"])
    np.testing.assert_array_equal(pkg["split"]["test"], ref["test"])
    assert pkg["calibration_C"] == ref["C_sel"]
    assert pkg["panel"] == ref["panel"]
    np.testing.assert_array_equal(
        pkg["stability"]["gene"].to_numpy(), ref["stability"]["gene"].to_numpy())
    np.testing.assert_allclose(
        pkg["stability"]["selection_freq"].to_numpy(),
        ref["stability"]["selection_freq"].to_numpy(), rtol=0, atol=0)

    # final model: identical coefficients, probabilities and metrics
    np.testing.assert_allclose(
        pkg["coefficients"]["beta"].to_numpy(),
        np.append(ref["beta"], ref["intercept"]), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(pkg["metrics"]["test_proba"], ref["test_proba"],
                               rtol=1e-9, atol=1e-9)
    assert pkg["metrics"]["test_auc"] == pytest.approx(ref["test_auc"], abs=1e-9)
    assert pkg["metrics"]["test_acc"] == pytest.approx(ref["test_acc"], abs=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_intermediate_steps_identical_to_reference(case):
    s = simulate_data(species="human", **case)
    cfg = _cfg()
    ref = reference_fit_panel(s["X"], s["obs"], s["var_names"], cfg)

    vn = _default_var_names(s["X"], s["var_names"])
    prep = _prepare_species(s["X"], s["obs"], vn, cfg)
    task = _build_task(prep, cfg)

    # gene mask (all ages valid here, so base = all cells) + kept names + matrix
    mask = _compute_gene_mask(s["X"], cfg.min_gene_counts, cfg.min_gene_cells_frac)
    np.testing.assert_array_equal(mask, ref["gene_mask"])
    np.testing.assert_array_equal(prep["var_names"], ref["var_names"])
    np.testing.assert_allclose(np.asarray(prep["X_log"].todense()), ref["X_log"],
                               rtol=0, atol=0)

    # split, effect score, correlation, pruning, z-score
    tr, _ = _stratified_group_split(task["y"], task["groups"], cfg.test_size,
                                    random_state=cfg.random_state)
    score = _supervised_effect_score(task["X_task"][tr], task["y"][tr],
                                     eps=cfg.cand_eps, min_per_class=cfg.effect_min_per_class)
    np.testing.assert_allclose(score, ref["score"], rtol=0, atol=0)

    corr = _gene_correlation(task["X_task"], tr, np.arange(task["X_task"].shape[1]),
                             max_cells=cfg.max_corr_cells, random_state=cfg.random_state)
    np.testing.assert_allclose(corr, ref["corr"], rtol=0, atol=0)

    kept, _, _ = _prune_by_abs_corr_greedy(prep["var_names"], score, corr,
                                           thr=cfg.prune_abs_corr,
                                           min_kept=cfg.min_pruned_genes)
    np.testing.assert_array_equal(kept, ref["kept_pos"])

    Xp = task["X_task"][:, kept]
    mean, std = _train_feature_moments(Xp[tr])
    Xz = _zscore_features(Xp, mean, std, cfg.z_clip)
    np.testing.assert_allclose(Xz, ref["Xz"], rtol=0, atol=0)
