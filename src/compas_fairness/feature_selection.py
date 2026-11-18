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
