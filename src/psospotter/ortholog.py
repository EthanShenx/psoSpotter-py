"""Cross-species ortholog helpers.

Mirror ``keep_gene_pair`` and ``load_human_mouse_ortholog_candidates`` from the
cross-species reference scripts.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Mitochondrial / ribosomal name patterns (case-insensitive), matching the
# reference module-level regexes ``_MT_RE_H``, ``_MT_RE_M`` and ``_RP_RE``.
_MT_RE_H = re.compile(r"^MT-", re.IGNORECASE)
_MT_RE_M = re.compile(r"^mt-", re.IGNORECASE)
_RP_RE = re.compile(r"^(RPS|RPL|Mrps|Mrpl|Rps|Rpl)", re.IGNORECASE)


def _keep_gene_pair(hg: Any, mg: Any, drop_mt_rp: bool = True) -> bool:
    """Decide whether a human/mouse gene pair should be kept.

    Aim: reproduce ``keep_gene_pair``. Drops empty/``None`` pairs and, when
    ``drop_mt_rp`` is ``True``, mitochondrial and ribosomal genes on either side.

    Args:
        hg: Human gene symbol.
        mg: Mouse gene symbol.
        drop_mt_rp: Whether to drop mitochondrial/ribosomal genes.

    Returns:
        ``True`` if the pair should be kept, else ``False``.

    Examples:
        >>> _keep_gene_pair("KRT5", "Krt5")
        True
        >>> _keep_gene_pair("MT-CO1", "mt-Co1")
        False
    """
    if hg is None or mg is None:
        return False
    hg = str(hg)
    mg = str(mg)
    if hg == "" or mg == "":
        return False
    if drop_mt_rp:
        if _MT_RE_H.match(hg) or _MT_RE_M.match(mg):
            return False
        if _RP_RE.match(hg) or _RP_RE.match(mg):
            return False
    return True


def _load_ortholog_candidates(
    human_genes: np.ndarray,
    mouse_genes: np.ndarray,
    ortholog: pd.DataFrame,
    drop_mt_rp: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], np.ndarray]:
    """Build the usable 1:1 ortholog candidate set.

    Aim: reproduce ``load_human_mouse_ortholog_candidates``. Filters an ortholog
    table to one-to-one pairs that pass :func:`_keep_gene_pair` and are present
    in both gene universes, then reports the training-species gene positions.

    Args:
        human_genes: Training-species gene names (in column order).
        mouse_genes: Test-species gene names.
        ortholog: DataFrame with ``human_gene`` and ``mouse_gene`` columns (and
            an optional ``orthology_type`` column filtered to
            ``"ortholog_one2one"`` when present).
        drop_mt_rp: Passed to :func:`_keep_gene_pair`.

    Returns:
        A tuple ``(orth, human2mouse, candidate_pos)`` where ``orth`` is the
        filtered table, ``human2mouse`` maps training gene to test gene, and
        ``candidate_pos`` is a ``int32`` array of 0-based positions of usable
        training genes within ``human_genes``.

    Raises:
        ValueError: If the table lacks the required columns.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> orth = pd.DataFrame({"human_gene": ["KRT5"], "mouse_gene": ["Krt5"]})
        >>> _, h2m, pos = _load_ortholog_candidates(
        ...     np.array(["KRT5", "X"]), np.array(["Krt5"]), orth)
        >>> h2m["KRT5"], pos.tolist()
        ('Krt5', [0])
    """
    if not {"human_gene", "mouse_gene"}.issubset(ortholog.columns):
        raise ValueError("Ortholog table must contain columns: human_gene, mouse_gene")

    orth = ortholog.copy()
    orth["human_gene"] = orth["human_gene"].astype(str)
    orth["mouse_gene"] = orth["mouse_gene"].astype(str)

    if "orthology_type" in orth.columns:
        orth = orth[orth["orthology_type"] == "ortholog_one2one"].copy()

    keep = orth.apply(
        lambda r: _keep_gene_pair(r["human_gene"], r["mouse_gene"], drop_mt_rp), axis=1
    )
    orth = orth[keep].copy()
    human_set = set(np.asarray(human_genes).tolist())
    mouse_set = set(np.asarray(mouse_genes).tolist())
    orth = orth[orth["human_gene"].isin(human_set)].copy()
    orth = orth[orth["mouse_gene"].isin(mouse_set)].copy()
    orth = orth.drop_duplicates(subset=["human_gene", "mouse_gene"]).copy()

    human2mouse = dict(zip(orth["human_gene"], orth["mouse_gene"]))
    candidate_mask = np.isin(np.asarray(human_genes, dtype=object),
                             orth["human_gene"].to_numpy(dtype=object))
    candidate_pos = np.where(candidate_mask)[0].astype(np.int32)
    return orth, human2mouse, candidate_pos
