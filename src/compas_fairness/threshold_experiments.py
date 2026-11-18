from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from compas_fairness.config import RANDOM_STATE, TARGET
from compas_fairness.metrics import p_rule
from compas_fairness.preprocessing import (
    AGE_CATEGORY_MAPPING,
    add_length_of_stay,
    bin_priors_count,
    drop_rows_missing_jail_dates,
    encode_features,
)
from compas_fairness.thresholds import optimize_threshold_for_fairness

RACE_SOURCE_COLUMNS = [
    "age", "c_charge_degree", "age_cat", "sex",
    "priors_count", "is_recid", "c_jail_in", "c_jail_out",
]
RACE_MODEL_FEATURES = ["age_cat", "c_charge_degree", "sex", "priors_count", "length_of_stay"]


@dataclass
class ThresholdExperiment:
    name: str
    model: LogisticRegression
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_test: pd.Series


def build_race_experiment(data, random_state=RANDOM_STATE):
    race_data = encode_features(data[RACE_SOURCE_COLUMNS + [TARGET, "race"]])
    race_data = drop_rows_missing_jail_dates(race_data)
    race_data = add_length_of_stay(race_data)
    race_data = bin_priors_count(race_data)
    X, y = race_data[RACE_MODEL_FEATURES], race_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    return ThresholdExperiment("race", model, X_train, X_test, y_train, y_test, race_data.loc[X_test.index, "race"])


def build_sex_experiment(data, random_state=RANDOM_STATE):
    dummies = pd.get_dummies(data[["age", "priors_count", "sex"]])
    X = dummies.drop(columns=["sex_Female", "sex_Male"])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    return ThresholdExperiment("sex", model, X_train, X_test, y_train, y_test, data.loc[X_test.index, "sex"])


def build_age_experiment(data, random_state=RANDOM_STATE):
    data_age = data.assign(age_cat_mapped=data["age_cat"].map(AGE_CATEGORY_MAPPING))
    dummies = pd.get_dummies(data_age[["sex", "priors_count", "age_cat_mapped"]])
    X = dummies.drop(columns=["age_cat_mapped"])
    y = data_age[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    return ThresholdExperiment("age", model, X_train, X_test, y_train, y_test, data_age.loc[X_test.index, "age_cat_mapped"])


def protected_mask(experiment, group_value):
    return (experiment.sensitive_test == group_value).to_numpy()


def unconstrained_report(experiment, group_value):
    predictions = experiment.model.predict(experiment.X_test)
    protected = protected_mask(experiment, group_value)
    return {
        "accuracy": accuracy_score(experiment.y_test, predictions),
        "p_rule": p_rule(predictions[protected].mean(), predictions[~protected].mean()),
        "covariance": np.cov(protected, predictions)[0, 1],
    }


def optimize_for_group(experiment, group_value, gamma, n_thresholds=1001):
    return optimize_threshold_for_fairness(
        experiment.model,
        experiment.X_test,
        experiment.y_test,
        protected_mask(experiment, group_value),
        gamma=gamma,
        n_thresholds=n_thresholds,
    )


def sweep_gamma(experiment, group_value, gamma_values):
    rows = []
    for gamma_value in gamma_values:
        result = optimize_for_group(experiment, group_value, gamma_value)
        rows.append({"gamma": gamma_value, **result})
    return pd.DataFrame(rows)


def age_group_baselines(experiment):
    rows = []
    for age_group in AGE_CATEGORY_MAPPING.values():
        report = unconstrained_report(experiment, age_group)
        rows.append({"age_group": age_group, "p_rule": report["p_rule"], "covariance": report["covariance"]})
    return pd.DataFrame(rows)


def age_group_optimized(experiment, gamma, n_thresholds):
    rows = []
    for age_group in AGE_CATEGORY_MAPPING.values():
        result = optimize_for_group(experiment, age_group, gamma, n_thresholds)
        rows.append({"age_group": age_group, **result})
    return pd.DataFrame(rows)
