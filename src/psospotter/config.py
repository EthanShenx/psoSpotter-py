"""Configuration for the psoSpotter pipeline.

:class:`PsoSpotterConfig` collects every tunable parameter (mirroring the
``USER EDIT BLOCK`` of each reference script); :func:`psospotter_config` builds
one with human/mouse and single/cross-species presets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .metadata import (
    _age_bin_human,
    _age_bin_mouse,
    _human_age_to_stage,
    _mouse_age_to_stage,
    _parse_human_age,
    _parse_postnatal_days,
)


@dataclass
class PsoSpotterConfig:
    """Tunable parameters for a psoSpotter run.

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        mode: ``"single"`` (within-species) or ``"cross"`` (shared life stages).
        age_col, sex_col, cond_col: obs column names.
        cond_map: Mapping from condition label to 0/1.
        age_bin_cols: Fixed order of age-dummy columns.
        age_parser: Callable mapping a raw age value to a numeric age or ``None``.
        age_binner: Callable mapping a numeric age to a stage label or ``None``.
        group_col_candidates: Candidate obs columns identifying donors/samples.
        target_sum: Library-size normalization target.
        min_gene_counts, min_gene_cells_frac: Gene-filter thresholds.
        test_size: Held-out fraction.
        use_group_split: Whether to use group-aware splitting when groups exist.
        max_corr_cells: Max cells for the correlation sub-sample.
        prune_abs_corr: Absolute-correlation pruning threshold.
        min_pruned_genes: Minimum genes to keep after pruning.
        n_repeats: Stability-selection repeats.
        subsample_frac_groups: Fraction of groups sub-sampled per repeat.
        c_grid: Candidate ``C`` values for sparsity calibration.
        l1_ratio_sel: Elastic-net mixing for selection.
        target_nnz_for_selection: Target nonzero gene count for calibration.
        panel_k: Final panel size.
        final_c, final_l1_ratio: Final-model regularization.
        z_clip: Z-score clipping bound.
        cand_eps: Effect-score epsilon.
        effect_min_per_class: Minimum cells per class for effect scoring.
        run_sample_level_cv: Whether to run sample-level CV (human default).
        run_label_shuffle: Whether to run the label-shuffle control (mouse default).
        label_shuffle_seed: Seed for the label-shuffle control.
        drop_mt_rp: Drop mitochondrial/ribosomal genes (cross-species).
        random_state: Base seed threaded through splits and model fits.
    """

    species: str
    mode: str
    cond_map: dict[str, int]
    age_bin_cols: list[str]
    age_parser: Callable[[Any], float | None]
    age_binner: Callable[[float | None], str | None]
    group_col_candidates: list[str]
    age_col: str = "Age"
    sex_col: str = "sex"
    cond_col: str = "condition"
    target_sum: float = 1e4
    min_gene_counts: float = 1
    min_gene_cells_frac: float = 0.01
    test_size: float = 0.2
    use_group_split: bool = True
    max_corr_cells: int = 15000
    prune_abs_corr: float = 0.90
    min_pruned_genes: int = 200
    n_repeats: int = 50
    subsample_frac_groups: float = 0.75
    c_grid: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    l1_ratio_sel: float = 0.9
    target_nnz_for_selection: int = 250
    panel_k: int = 50
    final_c: float = 1.0
    final_l1_ratio: float = 0.0
    z_clip: float = 5.0
    cand_eps: float = 1e-8
    effect_min_per_class: int = 50
    run_sample_level_cv: bool = False
    run_label_shuffle: bool = False
    label_shuffle_seed: int = 20260421
    drop_mt_rp: bool = True
    random_state: int = 42
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)


def psospotter_config(
    species: str = "human", mode: str = "single", **overrides: Any
) -> PsoSpotterConfig:
    """Build a :class:`PsoSpotterConfig` with species/mode presets.

    Aim: assemble pipeline parameters with sensible presets matching the
    reference scripts, applying any keyword ``overrides``.

    Args:
        species: ``"human"`` or ``"mouse"``.
        mode: ``"single"`` (uses calendar age bins) or ``"cross"`` (uses shared
            life-stage labels).
        **overrides: Any :class:`PsoSpotterConfig` field to override.

    Returns:
        A configured :class:`PsoSpotterConfig`.

    Raises:
        ValueError: If ``species`` or ``mode`` is not recognized, or an override
            names an unknown field.

    Examples:
        >>> cfg = psospotter_config("human")
        >>> cfg.age_bin_cols
        ['Age_0_20', 'Age_20_40', 'Age_40_60', 'Age_gt_60']
        >>> psospotter_config("mouse", mode="cross", panel_k=11).panel_k
        11
    """
    if species not in ("human", "mouse"):
        raise ValueError(f"Unknown species: {species!r}")
    if mode not in ("single", "cross"):
        raise ValueError(f"Unknown mode: {mode!r}")

    if mode == "cross":
        age_bin_cols = ["Age_juvenile", "Age_young_adult", "Age_adult", "Age_aged"]
        if species == "human":
            age_parser: Callable[[Any], float | None] = _parse_human_age
            age_binner: Callable[[float | None], str | None] = _human_age_to_stage
        else:
            age_parser = _parse_postnatal_days
            age_binner = _mouse_age_to_stage
    else:
        if species == "human":
            age_parser = _parse_human_age
            age_binner = _age_bin_human
            age_bin_cols = ["Age_0_20", "Age_20_40", "Age_40_60", "Age_gt_60"]
        else:
            age_parser = _parse_postnatal_days
            age_binner = _age_bin_mouse
            age_bin_cols = ["Age_0_6w", "Age_6w_6m", "Age_6m_1y", "Age_gt_1y"]

    if species == "human":
        cond_map = {"Healthy": 0, "psoriasis": 1}
        group_col_candidates = ["human_id", "Human", "human", "sample", "Sample",
                                "orig.ident", "donor", "Donor", "batch", "Batch",
                                "lane", "Lane"]
        n_repeats = 50
    else:
        cond_map = {"Healthy": 0, "IMQ-induced psoriasis": 1}
        group_col_candidates = ["mouse_id", "Mouse", "mouse", "sample", "Sample",
                                "orig.ident", "donor", "Donor", "batch", "Batch",
                                "lane", "Lane"]
        n_repeats = 100

    cfg = PsoSpotterConfig(
        species=species,
        mode=mode,
        cond_map=cond_map,
        age_bin_cols=age_bin_cols,
        age_parser=age_parser,
        age_binner=age_binner,
        group_col_candidates=group_col_candidates,
        n_repeats=n_repeats,
        run_sample_level_cv=(species == "human"),
        run_label_shuffle=(species == "mouse"),
    )

    valid = set(cfg.__dataclass_fields__)
    bad = set(overrides) - valid
    if bad:
        raise ValueError(f"Unknown config override(s): {sorted(bad)}")
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
