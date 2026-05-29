"""Metadata / covariate construction.

Faithful translations of the age/sex parsing and covariate helpers from the
reference psoSpotter scripts (``human.py``, ``mouse.py`` and the cross-species
variants). Every function here is deterministic and is matched bit-for-bit
against the reference in the parity tests.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

# Mouse postnatal-age label patterns (case-insensitive), matching the reference
# module-level regexes ``P_RE`` and ``E_RE``.
_P_RE = re.compile(r"^P\s*([0-9]+)", re.IGNORECASE)
_E_RE = re.compile(r"^E", re.IGNORECASE)


def _parse_human_age(age: Any) -> float | None:
    """Parse a human age value given in years.

    Aim: reproduce ``parse_human_age`` from ``human.py``.

    Args:
        age: A scalar value (numeric, string, or ``None``/``NaN``).

    Returns:
        The age in years as a ``float``, or ``None`` if the value is missing,
        blank, the strings ``"nan"``/``"none"`` (case-insensitive), non-numeric,
        or negative.

    Examples:
        >>> _parse_human_age("34")
        34.0
        >>> _parse_human_age(-3) is None
        True
    """
    if age is None:
        return None
    if isinstance(age, float) and math.isnan(age):
        return None
    s = str(age).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        val = float(s)
    except (ValueError, TypeError):
        return None
    if val < 0:
        return None
    return val


def _age_bin_human(age_years: float | None) -> str | None:
    """Bin a human age (years) into a stage label.

    Aim: reproduce ``age_bin_from_years`` from ``human.py``. Bins are
    left-closed: [0, 20), [20, 40), [40, 60), [60, inf).

    Args:
        age_years: Age in years, or ``None``.

    Returns:
        One of ``"Age_0_20"``, ``"Age_20_40"``, ``"Age_40_60"``,
        ``"Age_gt_60"``, or ``None`` when ``age_years`` is ``None``.

    Examples:
        >>> _age_bin_human(20)
        'Age_20_40'
    """
    if age_years is None:
        return None
    if age_years < 20:
        return "Age_0_20"
    if age_years < 40:
        return "Age_20_40"
    if age_years < 60:
        return "Age_40_60"
    return "Age_gt_60"


def _human_age_to_stage(age_years: float | None) -> str | None:
    """Map a human age (years) to a shared cross-species life stage.

    Aim: reproduce ``human_age_to_stage`` from the cross-species scripts.
    Cutoffs at 18, 40 and 65 years place human ages on the same axis as mouse.

    Args:
        age_years: Age in years, or ``None``.

    Returns:
        One of ``"Age_juvenile"``, ``"Age_young_adult"``, ``"Age_adult"``,
        ``"Age_aged"``, or ``None``.

    Examples:
        >>> _human_age_to_stage(70)
        'Age_aged'
    """
    if age_years is None:
        return None
    if age_years < 18:
        return "Age_juvenile"
    if age_years < 40:
        return "Age_young_adult"
    if age_years < 65:
        return "Age_adult"
    return "Age_aged"


def _parse_postnatal_days(age: Any) -> int | None:
    """Parse a mouse postnatal age (days) from a ``"P<days>"`` label.

    Aim: reproduce ``parse_postnatal_days`` from ``mouse.py``. Embryonic
    ("E...") labels are rejected.

    Args:
        age: A scalar value (string, numeric, or ``None``/``NaN``).

    Returns:
        Integer postnatal days, or ``None`` for missing, blank, embryonic, or
        non-matching values.

    Examples:
        >>> _parse_postnatal_days("P56")
        56
        >>> _parse_postnatal_days("E14.5") is None
        True
    """
    if age is None:
        return None
    if isinstance(age, float) and math.isnan(age):
        return None
    s = str(age).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    if _E_RE.match(s):
        return None
    m = _P_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None


def _age_bin_mouse(d: float | None) -> str | None:
    """Bin a mouse postnatal age (days) into a stage label.

    Aim: reproduce ``age_bin_from_days`` from ``mouse.py`` using week/month/year
    conversions of 7, 30.4375 and 365.25 days.

    Args:
        d: Postnatal days, or ``None``.

    Returns:
        One of ``"Age_0_6w"``, ``"Age_6w_6m"``, ``"Age_6m_1y"``,
        ``"Age_gt_1y"``, or ``None`` for ``None``/negative input.

    Examples:
        >>> _age_bin_mouse(42)
        'Age_6w_6m'
    """
    if d is None or d < 0:
        return None
    weeks = d / 7.0
    months = d / 30.4375
    years = d / 365.25
    if weeks < 6:
        return "Age_0_6w"
    if weeks >= 6 and months < 6:
        return "Age_6w_6m"
    if months >= 6 and years < 1:
        return "Age_6m_1y"
    if years >= 1:
        return "Age_gt_1y"
    return None


def _mouse_age_to_stage(d: float | None) -> str | None:
    """Map a mouse postnatal age (days) to a shared cross-species life stage.

    Aim: reproduce ``mouse_age_to_stage`` from the cross-species scripts,
    sharing labels with :func:`_human_age_to_stage`.

    Args:
        d: Postnatal days, or ``None``.

    Returns:
        One of ``"Age_juvenile"``, ``"Age_young_adult"``, ``"Age_adult"``,
        ``"Age_aged"``, or ``None``.

    Examples:
        >>> _mouse_age_to_stage(400)
        'Age_aged'
    """
    if d is None or d < 0:
        return None
    weeks = d / 7.0
    months = d / 30.4375
    years = d / 365.25
    if weeks < 6:
        return "Age_juvenile"
    if weeks >= 6 and months < 6:
        return "Age_young_adult"
    if months >= 6 and years < 1:
        return "Age_adult"
    return "Age_aged"


def _encode_sex(x: Any) -> float:
    """Encode a sex label as a numeric covariate.

    Aim: reproduce ``encode_sex`` from the reference scripts.

    Args:
        x: A scalar value (string, numeric, or ``None``/``NaN``).

    Returns:
        ``1.0`` for male (``"male"``, ``"m"``, the male symbol, ``"1"``),
        ``0.0`` for female (``"female"``, ``"f"``, the female symbol, ``"0"``),
        and ``0.5`` for unknown/missing.

    Examples:
        >>> _encode_sex("Male")
        1.0
        >>> _encode_sex(None)
        0.5
    """
    if x is None:
        return 0.5
    if isinstance(x, float) and math.isnan(x):
        return 0.5
    s = str(x).strip().lower()
    if s in ("male", "m", "♂", "1"):
        return 1.0
    if s in ("female", "f", "♀", "0"):
        return 0.0
    return 0.5


def _build_covariates(
    age_bin: pd.Series, sex: pd.Series, age_bin_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the one-hot age matrix and numeric sex vector.

    Aim: reproduce ``build_covariates`` from the reference scripts. Age labels
    are one-hot encoded into a fixed column order (missing levels become
    all-zero columns, matching ``pd.get_dummies`` followed by reindexing); sex
    is encoded via :func:`_encode_sex`.

    Args:
        age_bin: Per-cell age-stage labels (may contain ``None``/``NaN``).
        sex: Per-cell sex labels, same length as ``age_bin``.
        age_bin_cols: Required dummy column order.

    Returns:
        A tuple ``(age_matrix, sex_vector)`` where ``age_matrix`` is a
        ``float32`` array of shape ``(n_cells, len(age_bin_cols))`` and
        ``sex_vector`` is a ``float32`` array of length ``n_cells``.

    Raises:
        ValueError: If ``age_bin`` and ``sex`` differ in length.

    Examples:
        >>> import pandas as pd
        >>> age, sex = _build_covariates(
        ...     pd.Series(["Age_0_20", None]), pd.Series(["M", "F"]),
        ...     ["Age_0_20", "Age_20_40"])
        >>> age.tolist()
        [[1.0, 0.0], [0.0, 0.0]]
    """
    age_bin = pd.Series(age_bin).reset_index(drop=True)
    sex = pd.Series(sex).reset_index(drop=True)
    if len(age_bin) != len(sex):
        raise ValueError("`age_bin` and `sex` must have the same length.")

    age_dum = pd.get_dummies(age_bin, prefix="", prefix_sep="")
    for c in age_bin_cols:
        if c not in age_dum.columns:
            age_dum[c] = 0
    age_dum = age_dum[age_bin_cols]
    age_mat = age_dum.to_numpy(dtype=np.float32)
    sex_num = sex.apply(_encode_sex).to_numpy(dtype=np.float32)
    return age_mat, sex_num
