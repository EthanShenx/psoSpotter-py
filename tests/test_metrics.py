import numpy as np
import pytest

from psospotter.metrics import (
    _aggregate_sample_predictions,
    _binary_auc,
    _classification_report,
)


def test_binary_auc_textbook_and_perfect():
    assert _binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert _binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_binary_auc_single_class_nan():
    assert np.isnan(_binary_auc(np.array([1, 1, 1]), np.array([0.2, 0.8, 0.5])))


def test_classification_report_values():
    rep = _classification_report(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert rep["accuracy"] == pytest.approx(0.75)
    assert rep["Class1"]["precision"] == pytest.approx(1.0)
    assert rep["Class1"]["recall"] == pytest.approx(0.5)
    assert rep["Class1"]["f1-score"] == pytest.approx(2 / 3)


def test_classification_report_zero_division():
    rep = _classification_report(np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert rep["Class1"]["precision"] == 0.0
    assert rep["Class1"]["f1-score"] == 0.0


def test_aggregate_sample_predictions_mean_threshold_order():
    df = _aggregate_sample_predictions(
        np.array(["a", "a", "b", "b"], dtype=object),
        np.array([1, 1, 0, 0]), np.array([0.6, 0.8, 0.2, 0.4]))
    assert df["group"].tolist() == ["a", "b"]  # first-appearance order
    np.testing.assert_allclose(df["sample_score"].to_numpy(), [0.7, 0.3])
    assert df["y_pred"].tolist() == [1, 0]
    assert df["n_cells"].tolist() == [2, 2]


def test_aggregate_sample_predictions_mixed_label_error():
    with pytest.raises(RuntimeError, match="mixed-label"):
        _aggregate_sample_predictions(np.array(["a", "a"], dtype=object),
                                      np.array([0, 1]), np.array([0.2, 0.8]))
