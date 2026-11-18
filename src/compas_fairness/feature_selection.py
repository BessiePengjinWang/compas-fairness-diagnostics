import pandas as pd
from sklearn.linear_model import LogisticRegression

from compas_fairness.config import RANDOM_STATE, TARGET
from compas_fairness.information import (
    accuracy_coefficient,
    discrimination_coefficient,
    shapley_values,
    values_for_all_subsets,
)
from compas_fairness.metrics import group_accuracy_gap
from compas_fairness.preprocessing import (
    add_length_of_stay,
    bin_priors_count,
    drop_rows_missing_jail_dates,
    encode_features,
)

CORE_FEATURES = ["c_charge_degree", "age_cat", "sex", "race", "length_of_stay"]
EXTENDED_FEATURES = ["priors_count"] + CORE_FEATURES
PROTECTED_ATTRIBUTE = "race"
NON_PROTECTED_FEATURES = [name for name in EXTENDED_FEATURES if name != PROTECTED_ATTRIBUTE]
CANDIDATE_DROPS = [
    ["length_of_stay", "sex"],
    ["length_of_stay", "priors_count"],
    ["length_of_stay", "age_cat"],
    ["length_of_stay", "race"],
    ["length_of_stay"],
]


def build_part2_dataset(data):
    prepared = encode_features(data)
    prepared = drop_rows_missing_jail_dates(prepared)
    prepared = add_length_of_stay(prepared)
    prepared = bin_priors_count(prepared)
    return prepared.drop_duplicates()


def build_race_recidivism_frame(data):
    frame = data[data["race"].isin(["Caucasian", "African-American"])].copy()
    frame["race"] = frame["race"].map({"Caucasian": 1, "African-American": 0})
    return frame


def split_train_test_validation(final_dataset, random_state=RANDOM_STATE):
    shuffled = final_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    row_count = len(shuffled)
    train_set = shuffled.iloc[: int(row_count * 5 / 7)]
    test_set = shuffled.iloc[int(row_count * 5 / 7): int(row_count * 6 / 7)]
    validation_set = shuffled.iloc[int(row_count * 6 / 7):]
    return train_set, test_set, validation_set


def core_versus_extended_accuracy(train_set, validation_set):
    train_labels = train_set[TARGET].to_numpy()
    validation_labels = validation_set[TARGET].to_numpy()
    classifier_with_priors = LogisticRegression().fit(train_set[EXTENDED_FEATURES], train_labels)
    classifier_without_priors = LogisticRegression().fit(train_set[CORE_FEATURES], train_labels)
    accuracy_with_priors = classifier_with_priors.score(validation_set[EXTENDED_FEATURES], validation_labels)
    accuracy_without_priors = classifier_without_priors.score(validation_set[CORE_FEATURES], validation_labels)
    return accuracy_with_priors, accuracy_without_priors


def shapley_score_table(train_set):
    outcome = train_set[TARGET].to_numpy().reshape(-1, 1)
    features = train_set[NON_PROTECTED_FEATURES].to_numpy()
    protected = train_set[PROTECTED_ATTRIBUTE].to_numpy().reshape(-1, 1)
    feature_count = len(NON_PROTECTED_FEATURES)

    accuracy_values = values_for_all_subsets(lambda subset: accuracy_coefficient(outcome, features, protected, subset), feature_count)
    discrimination_values = values_for_all_subsets(lambda subset: discrimination_coefficient(outcome, features, protected, subset), feature_count)

    return pd.DataFrame({
        "feature": NON_PROTECTED_FEATURES,
        "accuracy_coef": shapley_values(accuracy_values, feature_count),
        "discrimination_coef": shapley_values(discrimination_values, feature_count),
    })


def feature_removal_results(train_set, test_set, random_state=RANDOM_STATE):
    train_labels = train_set[TARGET].to_numpy()
    test_labels = test_set[TARGET].to_numpy()
    results = []
    for feature in ["base"] + EXTENDED_FEATURES:
        train_subset = train_set[EXTENDED_FEATURES] if feature == "base" else train_set[EXTENDED_FEATURES].drop(columns=[feature])
        test_subset = test_set[EXTENDED_FEATURES] if feature == "base" else test_set[EXTENDED_FEATURES].drop(columns=[feature])
        classifier = LogisticRegression(random_state=random_state).fit(train_subset, train_labels)
        predictions = classifier.predict(test_subset)
        results.append({
            "removed_feature": feature,
            "accuracy": classifier.score(test_subset, test_labels),
            "accuracy_gap_between_races": group_accuracy_gap(test_set["race"].to_numpy(), predictions, test_labels),
        })
    return pd.DataFrame(results)


def final_feature_set_comparison(train_set, validation_set, random_state=RANDOM_STATE):
    train_labels = train_set[TARGET].to_numpy()
    validation_labels = validation_set[TARGET].to_numpy()
    comparison = []
    for dropped in CANDIDATE_DROPS:
        train_subset = train_set[EXTENDED_FEATURES].drop(columns=dropped)
        validation_subset = validation_set[EXTENDED_FEATURES].drop(columns=dropped)
        classifier = LogisticRegression(random_state=random_state).fit(train_subset, train_labels)
        comparison.append({
            "dropped_features": ", ".join(dropped),
            "validation_accuracy": classifier.score(validation_subset, validation_labels),
        })
    return pd.DataFrame(comparison)


DEFAULT_SPLIT_SEEDS = tuple(range(20))


def feature_removal_over_splits(final_dataset, seeds=DEFAULT_SPLIT_SEEDS):
    collected = []
    for seed in seeds:
        train_set, test_set, _ = split_train_test_validation(final_dataset, seed)
        results = feature_removal_results(train_set, test_set)
        results["seed"] = seed
        collected.append(results)
    return pd.concat(collected, ignore_index=True)


def summarize_feature_removal(per_split_results):
    base = per_split_results[per_split_results["removed_feature"] == "base"].set_index("seed")
    rows = []
    for feature, group in per_split_results.groupby("removed_feature", sort=False):
        group = group.set_index("seed")
        rows.append({
            "removed_feature": feature,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_sd": group["accuracy"].std(),
            "gap_mean": group["accuracy_gap_between_races"].mean(),
            "gap_sd": group["accuracy_gap_between_races"].std(),
            "share_of_splits_gap_below_base": (group["accuracy_gap_between_races"] < base["accuracy_gap_between_races"]).mean(),
            "share_of_splits_accuracy_at_or_above_base": (group["accuracy"] >= base["accuracy"] - 1e-12).mean(),
        })
    return pd.DataFrame(rows)


def final_comparison_over_splits(final_dataset, seeds=DEFAULT_SPLIT_SEEDS):
    collected = []
    for seed in seeds:
        train_set, _, validation_set = split_train_test_validation(final_dataset, seed)
        results = final_feature_set_comparison(train_set, validation_set)
        results["seed"] = seed
        collected.append(results)
    per_split = pd.concat(collected, ignore_index=True)
    pivot = per_split.pivot(index="seed", columns="dropped_features", values="validation_accuracy")
    best_each_split = pivot.max(axis=1)
    summary = pd.DataFrame({
        "validation_accuracy_mean": pivot.mean(),
        "validation_accuracy_sd": pivot.std(),
        "share_of_splits_best_or_tied": (pivot.ge(best_each_split - 1e-12, axis=0)).mean(),
    })
    return summary.loc[[", ".join(drop) for drop in CANDIDATE_DROPS]]
