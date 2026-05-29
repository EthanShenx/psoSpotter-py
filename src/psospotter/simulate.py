"""Synthetic data generation for examples and tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def simulate_data(
    n_cells: int = 200,
    n_genes: int = 60,
    n_groups: int = 8,
    signal_genes: int = 10,
    species: str = "human",
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate a small psoSpotter dataset.

    Aim: produce a tiny synthetic counts matrix and obs DataFrame with
    detectable between-condition signal, for documentation examples and tests.
    Groups are label-pure (each donor is entirely healthy or entirely disease).

    Args:
        n_cells: Number of cells (rows).
        n_genes: Number of genes (columns).
        n_groups: Number of donor groups.
        signal_genes: Number of leading genes carrying condition signal.
        species: ``"human"`` or ``"mouse"`` (controls the ``Age`` encoding).
        seed: Optional RNG seed.

    Returns:
        A dict with ``X`` (a :class:`scipy.sparse.csr_matrix` of counts),
        ``obs`` (a DataFrame with ``Age``, ``sex``, ``condition``, ``sample``),
        and ``var_names`` (gene names).

    Examples:
        >>> s = simulate_data(n_cells=50, n_genes=10, seed=1)
        >>> s["X"].shape
        (50, 10)
    """
    rng = np.random.RandomState(seed)
    grp = rng.randint(0, n_groups, size=n_cells)
    grp_cond = np.array([i % 2 for i in range(n_groups)], dtype=np.int8)
    y = grp_cond[grp]

    base = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float64)
    bump = rng.poisson(3.0, size=(n_cells, signal_genes)).astype(np.float64)
    base[:, :signal_genes] += y[:, None] * bump
    X = sparse.csr_matrix(base)
    var_names = np.array([f"Gene{j}" for j in range(n_genes)], dtype=object)

    if species == "human":
        age: Any = rng.randint(5, 80, size=n_cells)
        disease = "psoriasis"
    else:
        age = np.array([f"P{d}" for d in rng.randint(10, 400, size=n_cells)], dtype=object)
        disease = "IMQ-induced psoriasis"

    obs = pd.DataFrame(
        {
            "Age": age,
            "sex": np.where(rng.rand(n_cells) > 0.5, "M", "F"),
            "condition": np.where(y == 1, disease, "Healthy"),
            "sample": np.array([f"S{g}" for g in grp], dtype=object),
        }
    )
    return {"X": X, "obs": obs, "var_names": var_names}
