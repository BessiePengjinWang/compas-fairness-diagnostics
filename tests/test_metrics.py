import numpy as np
import pytest

from compas_fairness.metrics import calibration_gap, p_rule


@pytest.mark.parametrize(
    "protected_rate, non_protected_rate, expected",
    [(0.3, 0.6, 50.0), (0.6, 0.3, 50.0), (0.5, 0.5, 100.0), (0.2, 0.8, 25.0)],
)
def test_p_rule_known_answers(protected_rate, non_protected_rate, expected):
    assert p_rule(protected_rate, non_protected_rate) == pytest.approx(expected)


def test_p_rule_never_exceeds_one_hundred():
    rates = np.linspace(0.05, 1.0, 20)
    for protected_rate in rates:
        for non_protected_rate in rates:
            assert p_rule(protected_rate, non_protected_rate) <= 100.0 + 1e-9


def test_calibration_gap_is_one_when_one_group_all_right_and_other_all_wrong():
    protected = np.array([1, 1, 0, 0])
    predictions = np.array([1, 1, 0, 0])
    truth = np.array([1, 1, 1, 1])
    assert calibration_gap(protected, predictions, truth) == pytest.approx(1.0)


def test_calibration_gap_known_partial_answer():
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    predictions = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    truth = np.ones(8, dtype=int)
    assert calibration_gap(protected, predictions, truth) == pytest.approx(0.25)


def test_calibration_gap_is_zero_when_groups_equally_accurate():
    protected = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 1, 0])
    truth = np.array([1, 1, 1, 1])
    assert calibration_gap(protected, predictions, truth) == pytest.approx(0.0)


def test_calibration_gap_does_not_depend_on_which_group_is_more_accurate():
    protected = np.array([1, 1, 0, 0])
    predictions = np.array([0, 0, 1, 1])
    truth = np.array([1, 1, 1, 1])
    assert calibration_gap(protected, predictions, truth) == pytest.approx(1.0)


@pytest.mark.parametrize("protected_rate, non_protected_rate", [(0.0, 0.5), (0.5, 0.0), (0.0, 0.0)])
def test_p_rule_is_not_a_number_when_a_group_has_no_positive_predictions(protected_rate, non_protected_rate):
    assert np.isnan(p_rule(protected_rate, non_protected_rate))
