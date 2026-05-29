"""Evaluation metrics and sample-level aggregation.

Mirror ``safe_binary_auc``, ``make_classification_report`` and
``aggregate_sample_predictions`` from the reference. AUC and the classification
report delegate to scikit-learn, matching the reference exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC-AUC, safe for degenerate label sets.

    Aim: reproduce ``safe_binary_auc`` (delegates to
    :func:`sklearn.metrics.roc_auc_score`).

    Args:
        y_true: 0/1 true labels.
        y_score: Predicted positive-class scores.

    Returns:
        The ROC-AUC, or ``nan`` if ``y_true`` has fewer than two classes.

    Examples:
        >>> _binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.75
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Per-class precision/recall/F1 classification report.

    Aim: reproduce ``make_classification_report`` (target names ``Class0`` /
    ``Class1``, ``zero_division=0``), returned as a dictionary.

    Args:
        y_true: 0/1 true labels.
        y_pred: 0/1 predicted labels.

    Returns:
        The scikit-learn report dictionary (per-class precision/recall/f1/support
        plus ``accuracy``).

    Examples:
        >>> rep = _classification_report(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
        >>> round(rep["accuracy"], 2)
        0.75
    """
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Class0", "Class1"],
        output_dict=True,
        zero_division=0,
    )


def _aggregate_sample_predictions(
    groups: np.ndarray, y_true: np.ndarray, y_score: np.ndarray
) -> pd.DataFrame:
    """Aggregate cell-level scores to the sample (group) level.

    Aim: reproduce ``aggregate_sample_predictions``. Averages cell scores within
    each group, takes the group's single true label, counts cells, and
    thresholds the mean score at ``0.5``. Group order follows first appearance
    (``groupby(sort=False)``).

    Args:
        groups: Group identifiers, one per cell.
        y_true: 0/1 labels, one per cell.
        y_score: Cell-level scores.

    Returns:
        A DataFrame with columns ``group``, ``y_true``, ``sample_score``,
        ``n_cells``, ``y_pred``.

    Raises:
        RuntimeError: If any group contains more than one distinct label.

    Examples:
        >>> df = _aggregate_sample_predictions(
        ...     np.array(["a", "a", "b", "b"]), np.array([1, 1, 0, 0]),
        ...     np.array([0.6, 0.8, 0.2, 0.4]))
        >>> df["sample_score"].tolist()
        [0.7, 0.3]
    """
    df = pd.DataFrame(
        {
            "group": np.asarray(groups, dtype=object),
            "y_true": np.asarray(y_true, dtype=np.int8),
            "cell_score": np.asarray(y_score, dtype=np.float64),
        }
    )
    nu = df.groupby("group")["y_true"].nunique()
    bad = nu[nu > 1]
    if len(bad) > 0:
        raise RuntimeError(
            "Found mixed-label groups during sample aggregation. "
            f"Examples: {bad.index.tolist()[:10]}"
        )

    df_sample = (
        df.groupby("group", sort=False)
        .agg(
            y_true=("y_true", "first"),
            sample_score=("cell_score", "mean"),
            n_cells=("cell_score", "size"),
        )
        .reset_index()
    )
    df_sample["y_pred"] = (df_sample["sample_score"] >= 0.5).astype(np.int8)
    return df_sample
