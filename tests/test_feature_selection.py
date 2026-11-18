import pandas as pd
import pytest

from compas_fairness.data import load_compas
from compas_fairness.feature_selection import (
    CANDIDATE_DROPS,
    NON_PROTECTED_FEATURES,
    build_part2_dataset,
    feature_removal_over_splits,
    final_comparison_over_splits,
    split_train_test_validation,
    summarize_feature_removal,
)


def test_summarize_feature_removal_known_answer():
    per_split = pd.DataFrame({
        "seed": [0, 1, 0, 1],
        "removed_feature": ["base", "base", "race", "race"],
        "accuracy": [0.60, 0.70, 0.65, 0.70],
        "accuracy_gap_between_races": [0.10, 0.20, 0.05, 0.25],
    })
    summary = summarize_feature_removal(per_split).set_index("removed_feature")
    assert summary.loc["race", "accuracy_mean"] == pytest.approx(0.675)
    assert summary.loc["race", "gap_mean"] == pytest.approx(0.15)
    assert summary.loc["race", "share_of_splits_gap_below_base"] == pytest.approx(0.5)
    assert summary.loc["race", "share_of_splits_accuracy_at_or_above_base"] == pytest.approx(1.0)
    assert summary.loc["base", "share_of_splits_gap_below_base"] == pytest.approx(0.0)


def test_three_way_split_is_disjoint_and_complete():
    dataset = build_part2_dataset(load_compas())
    train_set, test_set, validation_set = split_train_test_validation(dataset, 3)
    identifiers = [set(part["id"]) for part in (train_set, test_set, validation_set)]
    assert sum(len(part) for part in identifiers) == len(dataset)
    assert identifiers[0].isdisjoint(identifiers[1]) and identifiers[0].isdisjoint(identifiers[2]) and identifiers[1].isdisjoint(identifiers[2])


def test_part2_dataset_has_no_rows_without_jail_dates_and_race_is_not_a_shapley_feature():
    dataset = build_part2_dataset(load_compas())
    assert dataset[["c_jail_in", "c_jail_out"]].notna().all().all()
    assert "race" not in NON_PROTECTED_FEATURES


def test_over_splits_tables_have_the_expected_shape():
    dataset = build_part2_dataset(load_compas())
    per_split = feature_removal_over_splits(dataset, seeds=(0, 1))
    assert per_split["seed"].nunique() == 2
    assert per_split["removed_feature"].iloc[0] == "base"
    summary = summarize_feature_removal(per_split)
    assert summary["removed_feature"].tolist()[0] == "base"
    comparison = final_comparison_over_splits(dataset, seeds=(0, 1))
    assert len(comparison) == len(CANDIDATE_DROPS)
    assert comparison["validation_accuracy_mean"].between(0.5, 0.8).all()
