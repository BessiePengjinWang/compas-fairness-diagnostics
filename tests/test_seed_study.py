import numpy as np
import pandas as pd
import pytest

from compas_fairness.data import load_compas
from compas_fairness.seed_study import (
    STAY_TREATMENTS,
    accuracy_points_lost_to_reach,
    missing_dates_sensitivity,
    part1_cost_table,
    part1_seed_table,
    summarize_over_seeds,
)
from compas_fairness.threshold_experiments import ThresholdExperiment
from tests.stubs import FixedProbabilityModel


def test_summarize_over_seeds_known_answer():
    table = pd.DataFrame({"seed": [0, 1, 2], "metric": [1.0, 2.0, 6.0]})
    summary = summarize_over_seeds(table)
    assert summary.loc["metric", "mean"] == pytest.approx(3.0)
    assert summary.loc["metric", "min"] == pytest.approx(1.0)
    assert summary.loc["metric", "max"] == pytest.approx(6.0)
    assert summary.loc["metric", "std"] == pytest.approx(2.6457513110645907)


def test_seed_table_has_one_row_per_seed_and_different_seeds_give_different_splits():
    table = part1_seed_table(load_compas(), seeds=(0, 1))
    assert table["seed"].tolist() == [0, 1]
    assert table["race_unconstrained_p_rule"].nunique() == 2
    assert table["race_gamma_0.15_p_rule"].between(0, 100).all()
    assert table["sex_unconstrained_accuracy"].between(0.5, 0.9).all()


def test_cost_is_zero_when_the_unconstrained_model_already_meets_the_target():
    experiment = ThresholdExperiment(
        name="stub",
        model=FixedProbabilityModel([0.9, 0.1, 0.9, 0.1]),
        X_train=pd.DataFrame({"f": [0.0, 0.0]}),
        X_test=pd.DataFrame({"f": [0.0, 0.0, 0.0, 0.0]}),
        y_train=pd.Series([1, 0]),
        y_test=pd.Series([1, 0, 1, 0]),
        sensitive_train=pd.Series([1, 0]),
        sensitive_test=pd.Series([1, 1, 0, 0]),
        out_of_fold_probabilities=np.array([0.9, 0.1]),
    )
    assert accuracy_points_lost_to_reach(experiment, 1, 101) == 0.0


def test_cost_table_reports_a_positive_cost_for_race_and_none_for_the_oldest_group():
    table = part1_cost_table(load_compas(), seeds=(0,))
    assert 0 < table.loc[0, "race"] < 20
    assert table.loc[0, "sex"] >= 0
    assert table.loc[0, "age_group_2"] <= 1


def test_missing_dates_sensitivity_has_the_expected_row_counts_for_each_treatment():
    table = missing_dates_sensitivity(load_compas(), seeds=(0,))
    assert table.loc[(STAY_TREATMENTS[0], "rows"), "mean"] == 5915
    assert table.loc[(STAY_TREATMENTS[1], "rows"), "mean"] == 6150
    assert table.loc[(STAY_TREATMENTS[2], "rows"), "mean"] == 6150
    assert table.xs("unconstrained_p_rule", level="metric")["mean"].between(30, 90).all()
