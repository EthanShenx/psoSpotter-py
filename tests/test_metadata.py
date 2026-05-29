import numpy as np
import pandas as pd
import pytest

from psospotter.metadata import (
    _age_bin_human,
    _age_bin_mouse,
    _build_covariates,
    _encode_sex,
    _human_age_to_stage,
    _mouse_age_to_stage,
    _parse_human_age,
    _parse_postnatal_days,
)


@pytest.mark.parametrize(
    "value,expected",
    [("34", 34.0), (57.5, 57.5), (0, 0.0), ("", None), ("nan", None),
     ("NoNe", None), ("abc", None), (-3, None), (None, None), (float("nan"), None)],
)
def test_parse_human_age(value, expected):
    assert _parse_human_age(value) == expected or (
        expected is None and _parse_human_age(value) is None)


@pytest.mark.parametrize(
    "yrs,expected",
    [(None, None), (0, "Age_0_20"), (19.999, "Age_0_20"), (20, "Age_20_40"),
     (40, "Age_40_60"), (60, "Age_gt_60"), (100, "Age_gt_60")],
)
def test_age_bin_human_boundaries(yrs, expected):
    assert _age_bin_human(yrs) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("P56", 56), ("p 7", 7), ("E14.5", None), ("adult", None), ("", None), (None, None)],
)
def test_parse_postnatal_days(value, expected):
    assert _parse_postnatal_days(value) == expected


@pytest.mark.parametrize(
    "days,expected",
    [(None, None), (-1, None), (41, "Age_0_6w"), (42, "Age_6w_6m"),
     (200, "Age_6m_1y"), (400, "Age_gt_1y")],
)
def test_age_bin_mouse_boundaries(days, expected):
    assert _age_bin_mouse(days) == expected


@pytest.mark.parametrize(
    "yrs,expected",
    [(10, "Age_juvenile"), (18, "Age_young_adult"), (40, "Age_adult"), (65, "Age_aged")],
)
def test_human_age_to_stage(yrs, expected):
    assert _human_age_to_stage(yrs) == expected


@pytest.mark.parametrize("days,expected", [(20, "Age_juvenile"), (400, "Age_aged")])
def test_mouse_age_to_stage(days, expected):
    assert _mouse_age_to_stage(days) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("Male", 1.0), ("m", 1.0), ("1", 1.0), ("♂", 1.0), ("Female", 0.0),
     ("F", 0.0), ("0", 0.0), ("♀", 0.0), ("unknown", 0.5), (None, 0.5),
     (float("nan"), 0.5)],
)
def test_encode_sex(value, expected):
    assert _encode_sex(value) == expected


def test_build_covariates_one_hot_fixed_order_with_na():
    cols = ["Age_0_20", "Age_20_40", "Age_40_60", "Age_gt_60"]
    age, sex = _build_covariates(
        pd.Series(["Age_0_20", "Age_40_60", None]), pd.Series(["M", "F", "x"]), cols)
    assert age.shape == (3, 4)
    np.testing.assert_array_equal(age[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(age[1], [0, 0, 1, 0])
    np.testing.assert_array_equal(age[2], [0, 0, 0, 0])  # NA -> all zero
    np.testing.assert_array_equal(sex, [1.0, 0.0, 0.5])


def test_build_covariates_length_mismatch():
    with pytest.raises(ValueError):
        _build_covariates(pd.Series(["a"]), pd.Series(["M", "F"]), ["a"])


def test_build_covariates_single_element():
    age, sex = _build_covariates(pd.Series(["Age_0_20"]), pd.Series(["M"]),
                                 ["Age_0_20", "Age_20_40"])
    assert age.shape == (1, 2)
    np.testing.assert_array_equal(age[0], [1, 0])
