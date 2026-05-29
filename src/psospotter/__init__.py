"""psoSpotter: single-cell biomarker panel selection and psoriasis classification.

A Python port of the psoSpotter pipeline. Only the user-facing functions are
exported here; all implementation helpers are private (prefixed ``_``) inside
the submodules.

Example:
    >>> from psospotter import simulate_data, psospotter_config, fit_panel_model
    >>> data = simulate_data(seed=1)
    >>> cfg = psospotter_config("human")
    >>> result = fit_panel_model(data["X"], data["obs"], cfg, data["var_names"])
    >>> len(result["panel"]) <= cfg.panel_k
    True
"""

from __future__ import annotations

from .config import PsoSpotterConfig, psospotter_config
from .pipeline import fit_cross_species_panel, fit_panel_model
from .simulate import simulate_data

__all__ = [
    "fit_panel_model",
    "fit_cross_species_panel",
    "psospotter_config",
    "PsoSpotterConfig",
    "simulate_data",
]

__version__ = "0.1.0"
