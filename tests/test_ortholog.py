import numpy as np
import pandas as pd
import pytest

from psospotter.ortholog import _keep_gene_pair, _load_ortholog_candidates


@pytest.mark.parametrize(
    "hg,mg,expected",
    [("KRT5", "Krt5", True), ("", "Krt5", False), (None, "Krt5", False),
     ("MT-CO1", "Co1", False), ("CO1", "mt-Co1", False),
     ("RPL13", "Rpl13", False), ("Gene", "Mrpl1", False)],
)
def test_keep_gene_pair(hg, mg, expected):
    assert _keep_gene_pair(hg, mg) is expected


def test_keep_gene_pair_disable_filter():
    assert _keep_gene_pair("MT-CO1", "mt-Co1", drop_mt_rp=False) is True


def test_load_ortholog_candidates_filters_to_present_one2one():
    hg = np.array(["KRT5", "MT-CO1", "COL1A1", "ABSENT"], dtype=object)
    mg = np.array(["Krt5", "Col1a1"], dtype=object)
    orth = pd.DataFrame({
        "human_gene": ["KRT5", "COL1A1", "MT-CO1", "ABSENT"],
        "mouse_gene": ["Krt5", "Col1a1", "mt-Co1", "Nope"],
        "orthology_type": ["ortholog_one2one"] * 3 + ["ortholog_one2many"],
    })
    _, h2m, pos = _load_ortholog_candidates(hg, mg, orth)
    assert set(h2m) == {"KRT5", "COL1A1"}
    assert h2m["KRT5"] == "Krt5"
    np.testing.assert_array_equal(np.sort(pos), [0, 2])  # 0-based positions in hg


def test_load_ortholog_candidates_missing_columns():
    with pytest.raises(ValueError, match="must contain columns"):
        _load_ortholog_candidates(np.array(["A"]), np.array(["a"]),
                                  pd.DataFrame({"x": [1]}))
