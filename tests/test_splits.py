import numpy as np
import pytest

from psospotter.splits import _stratified_group_split, _stratified_random_split


def test_random_split_holds_out_each_class_and_partitions():
    y = np.repeat([0, 1], 10)
    tr, te = _stratified_random_split(y, test_size=0.2, random_state=1)
    assert set(tr.tolist()) | set(te.tolist()) == set(range(20))
    assert len(set(tr.tolist()) & set(te.tolist())) == 0
    assert np.sum(y[te] == 0) == 2 and np.sum(y[te] == 1) == 2


def test_random_split_min_one_per_class():
    y = np.array([0, 0, 0, 1, 1, 1])
    tr, te = _stratified_random_split(y, test_size=0.01, random_state=1)
    assert np.sum(y[te] == 0) >= 1 and np.sum(y[te] == 1) >= 1


def test_random_split_is_seed_deterministic():
    y = np.repeat([0, 1], 10)
    a = _stratified_random_split(y, 0.2, random_state=42)
    b = _stratified_random_split(y, 0.2, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_group_split_keeps_groups_intact_both_classes():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    g = np.array(["a", "a", "b", "b", "c", "d", "e", "f"], dtype=object)
    tr, te = _stratified_group_split(y, g, test_size=0.5, random_state=2)
    assert set(g[tr]).isdisjoint(set(g[te]))
    assert np.unique(y[tr]).size == 2 and np.unique(y[te]).size == 2


def test_group_split_rejects_mixed_label_groups():
    y = np.array([0, 1, 0, 1])
    g = np.array(["a", "a", "b", "b"], dtype=object)
    with pytest.raises(RuntimeError, match="mixed-label"):
        _stratified_group_split(y, g)


def test_group_split_single_class_errors():
    y = np.array([0, 0, 0, 0])
    g = np.array(["a", "a", "b", "b"], dtype=object)
    with pytest.raises(RuntimeError):
        _stratified_group_split(y, g)
