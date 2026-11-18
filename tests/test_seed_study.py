import pandas as pd
import pytest

from compas_fairness.data import load_compas
from compas_fairness.seed_study import part1_seed_table, summarize_over_seeds


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
