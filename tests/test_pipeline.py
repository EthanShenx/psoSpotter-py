import numpy as np
import pandas as pd
import pytest

from psospotter import (
    PsoSpotterConfig,
    fit_cross_species_panel,
    fit_panel_model,
    psospotter_config,
    simulate_data,
)


def _small_cfg(species="human", mode="single", **kw):
    return psospotter_config(species, mode=mode, min_pruned_genes=6, panel_k=6,
                             n_repeats=5, target_nnz_for_selection=6,
                             effect_min_per_class=5, max_corr_cells=10**6,
                             c_grid=(1e-2, 1e-1, 1), **kw)


def test_config_presets_and_overrides():
    h = psospotter_config("human")
    assert h.age_bin_cols == ["Age_0_20", "Age_20_40", "Age_40_60", "Age_gt_60"]
    assert h.run_sample_level_cv is True
    m = psospotter_config("mouse", mode="cross", panel_k=11)
    assert m.age_bin_cols == ["Age_juvenile", "Age_young_adult", "Age_adult", "Age_aged"]
    assert m.panel_k == 11 and m.run_label_shuffle is True
    assert isinstance(m, PsoSpotterConfig)


def test_config_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unknown config override"):
        psospotter_config("human", not_a_field=1)


def test_fit_panel_model_end_to_end_recovers_signal():
    s = simulate_data(n_cells=240, n_genes=50, n_groups=8, signal_genes=8, seed=1)
    res = fit_panel_model(s["X"], s["obs"], _small_cfg(), s["var_names"])
    assert len(res["panel"]) == 6
    assert np.isfinite(res["metrics"]["test_auc"])
    assert res["metrics"]["test_auc"] > 0.8
    # coefficients = panel + age bins + sex + intercept
    assert len(res["coefficients"]) == 6 + 4 + 1 + 1
    assert res["coefficients"]["feature"].iloc[-1] == "Intercept"
    assert len(set(res["panel"]) & {f"Gene{i}" for i in range(8)}) > 0


def test_fit_panel_model_sample_level_cv_for_grouped_human():
    s = simulate_data(n_cells=200, n_genes=40, n_groups=8, seed=5)
    res = fit_panel_model(s["X"], s["obs"], _small_cfg(), s["var_names"])
    assert res["metrics"]["sample_cv"] is not None
    assert res["metrics"]["sample_cv"]["mode"] in ("loso", "kfold")


def test_mouse_label_shuffle_control_runs():
    s = simulate_data(n_cells=220, n_genes=40, n_groups=8, species="mouse", seed=9)
    res = fit_panel_model(s["X"], s["obs"], _small_cfg("mouse"), s["var_names"])
    assert res["metrics"]["label_shuffle"] is not None
    assert np.isfinite(res["metrics"]["label_shuffle"]["test_auc"])


def test_too_few_task_cells_raises():
    s = simulate_data(n_cells=40, n_genes=20, seed=2)
    # map no conditions -> empty task
    cfg = _small_cfg(cond_map={"NoSuchLabel": 1})
    with pytest.raises(RuntimeError):
        fit_panel_model(s["X"], s["obs"], cfg, s["var_names"])


def test_var_names_length_mismatch_raises():
    s = simulate_data(n_cells=40, n_genes=20, seed=2)
    with pytest.raises(ValueError):
        fit_panel_model(s["X"], s["obs"], _small_cfg(), np.array(["only_one"]))


def test_fit_cross_species_panel_end_to_end():
    h = simulate_data(n_cells=240, n_genes=50, n_groups=8, species="human", seed=10)
    m = simulate_data(n_cells=200, n_genes=50, n_groups=6, species="mouse", seed=11)
    m_var = np.array(["m" + g for g in h["var_names"]], dtype=object)
    orth = pd.DataFrame({"human_gene": h["var_names"].astype(str),
                         "mouse_gene": m_var.astype(str),
                         "orthology_type": "ortholog_one2one"})
    res = fit_cross_species_panel(
        h["X"], h["obs"], m["X"], m["obs"], orth,
        _small_cfg("human", mode="cross"), psospotter_config("mouse", mode="cross"),
        h["var_names"], m_var)
    assert len(res["panel"]) == 6
    assert (res["panel"]["test_gene"] == ("m" + res["panel"]["train_gene"])).all()
    assert np.isfinite(res["external"]["auc"])
    assert res["external"]["auc"] > 0.7
