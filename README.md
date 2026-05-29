<div align="center">

<img src="logo/psoSpotter.png" alt="mitoSpotter Logo" width="180" />

# _mitoSpotter_

Single-cell RNA-seq biomarker-panel algorithm for sensitive psoriasis detection that selects a minimally viable gene panel in a cross-species manner.

</div>


## Installation

`psospotter` requires Python >= 3.9 and depends on `numpy`, `scipy`,
`scikit-learn` and `pandas` (installed automatically).

```sh
# from a local clone, run inside the psospotter-py/ directory:

# regular install
pip install .

# editable (development) install, plus the test/lint/type-check tools
pip install -e ".[dev]"
```

To run the checks after a development install:

```sh
pytest               # unit + integration tests
ruff check .         # lint
mypy src/psospotter  # type-check
```


## Pipeline

1. Parse age / sex / condition metadata into covariates.
2. Filter genes, library-size normalize and `log1p`-transform counts.
3. Score each gene by a supervised standardized mean difference.
4. Greedily prune correlated genes.
5. Z-score features (train moments) and clip.
6. Calibrate sparsity and run stability selection (elastic-net logistic).
7. Build the top-K gene panel; fit a final penalized logistic model with
   age/sex covariates.
8. Evaluate (cell-level metrics, optional sample-level CV / label-shuffle
   control), and — in cross-species mode — externally validate on the other
   species via the ortholog map.

## Quick start

```python
from psospotter import simulate_data, psospotter_config, fit_panel_model

# Tiny synthetic example
sim = simulate_data(n_cells=300, n_genes=80, seed=1)
cfg = psospotter_config("human")
res = fit_panel_model(sim["X"], sim["obs"], cfg, sim["var_names"])

res["panel"]               # selected gene panel
res["coefficients"]        # final model coefficients
res["metrics"]["test_auc"] # held-out cell-level ROC-AUC
```

`X` is a cells-by-genes matrix of **raw counts** (a `scipy.sparse` matrix) with
gene names supplied via `var_names`; `obs` is a per-cell metadata data frame with
the age, sex and condition columns named in the config.

## Data input

This package operates on **in-memory** matrices. The original's on-disk layer
(h5ad stripping, the two-pass zarr build, memmap, and the chunked CSR readers)
is intentionally not ported — load your `.h5ad`/zarr into a `scipy.sparse`
matrix first (e.g. via `anndata`/`scanpy`: `adata.X` is already cells × genes),
then call `fit_panel_model()` / `fit_cross_species_panel()`.

# Need help?

`psoSpotter` is a part of the ecosystem of [`scSAID`](https://skin-scsaid.com/). If you have any questions about `psoSpotter`, please don’t hesitate to
email Prof. Chaochen Wang (<chaochenwang@intl.zju.edu.cn>) and cc our active developer Mr. Yixiang Ren (<yixiangren99@gmail.com>) and Mr. Yuchen Shen (<coellearthx@gmail.com>). A more direct way is to raise a issue on GitHub.
