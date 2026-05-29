"""Train/test split helpers.

Faithful reproductions of ``stratified_group_split_local`` and
``stratified_random_split_local``, using the same NumPy ``RandomState`` seeds and
draw order as the reference so splits are bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _stratified_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_tries: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified group-aware train/test split.

    Aim: reproduce ``stratified_group_split_local``. Whole groups are assigned
    to train or test so no group is split, and both folds contain both classes.
    Groups must be label-pure.

    Args:
        y: 0/1 labels, one per cell.
        groups: Group identifiers, same length as ``y``.
        test_size: Target fraction of groups (per class) held out.
        random_state: Seed for the group-selection RNG.
        max_tries: Maximum random attempts to find a valid split.

    Returns:
        A tuple ``(train_idx, test_idx)`` of 0-based ``int64`` row indices.

    Raises:
        RuntimeError: If any group is label-mixed, only one class exists at the
            group level, or no valid split is found within ``max_tries``.

    Examples:
        >>> import numpy as np
        >>> y = np.array([0, 0, 1, 1, 0, 1])
        >>> g = np.array(["a", "a", "b", "b", "c", "d"])
        >>> tr, te = _stratified_group_split(y, g, test_size=0.5)
        >>> set(tr).isdisjoint(set(te))
        True
    """
    y = np.asarray(y, dtype=np.int8)
    g = np.asarray(groups, dtype=object)
    idx_all = np.arange(y.size, dtype=np.int64)

    df = pd.DataFrame({"i": idx_all, "y": y, "g": g})
    nu = df.groupby("g")["y"].nunique()
    bad = nu[nu > 1]
    if len(bad) > 0:
        raise RuntimeError(
            f"Found mixed-label groups (nunique>1). Examples: {bad.index.tolist()[:10]}"
        )

    g2y = df.groupby("g")["y"].first()
    g_list = g2y.index.to_numpy(dtype=object)
    g_y = g2y.to_numpy(dtype=np.int8)
    g0 = g_list[g_y == 0]
    g1 = g_list[g_y == 1]
    if g0.size == 0 or g1.size == 0:
        raise RuntimeError("Only one class exists at group level.")

    n_test0 = max(1, int(round(test_size * g0.size)))
    n_test1 = max(1, int(round(test_size * g1.size)))
    if g0.size > 1:
        n_test0 = min(n_test0, g0.size - 1)
    if g1.size > 1:
        n_test1 = min(n_test1, g1.size - 1)

    rng = np.random.RandomState(random_state)
    for _ in range(max_tries):
        te_g0 = rng.choice(g0, size=n_test0, replace=False)
        te_g1 = rng.choice(g1, size=n_test1, replace=False)
        te_groups = set(te_g0.tolist() + te_g1.tolist())
        te_mask = df["g"].isin(te_groups).to_numpy()
        te = df.loc[te_mask, "i"].to_numpy(dtype=np.int64)
        tr = df.loc[~te_mask, "i"].to_numpy(dtype=np.int64)
        if np.unique(y[tr]).size == 2 and np.unique(y[te]).size == 2:
            return tr, te
    raise RuntimeError("Failed to find valid stratified group split.")


def _stratified_random_split(
    y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random train/test split (no groups).

    Aim: reproduce ``stratified_random_split_local``. Holds out ``test_size`` of
    each class independently, using the same shuffle sequence as the reference.

    Args:
        y: 0/1 labels.
        test_size: Target fraction of each class held out.
        random_state: Seed for the shuffle RNG.

    Returns:
        A tuple ``(train_idx, test_idx)`` of 0-based ``int64`` row indices.

    Examples:
        >>> import numpy as np
        >>> tr, te = _stratified_random_split(np.array([0, 0, 0, 1, 1, 1]), 0.34)
        >>> len(tr) + len(te)
        6
    """
    y = np.asarray(y, dtype=np.int8)
    idx = np.arange(y.size, dtype=np.int64)
    rng = np.random.RandomState(random_state)

    i0 = idx[y == 0]
    i1 = idx[y == 1]
    rng.shuffle(i0)
    rng.shuffle(i1)

    nte0 = max(1, int(round(test_size * i0.size)))
    nte1 = max(1, int(round(test_size * i1.size)))
    te = np.concatenate([i0[:nte0], i1[:nte1]])
    tr = np.concatenate([i0[nte0:], i1[nte1:]])
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te
