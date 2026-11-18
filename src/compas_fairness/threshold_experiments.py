from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split

from compas_fairness.config import RANDOM_STATE, TARGET
from compas_fairness.metrics import p_rule
from compas_fairness.preprocessing import (
    AGE_CATEGORY_MAPPING,
    add_length_of_stay,
    bin_priors_count,
    drop_rows_missing_jail_dates,
    encode_features,
)
from compas_fairness.thresholds import optimize_threshold_from_probabilities

RACE_SOURCE_COLUMNS = [
    "age", "c_charge_degree", "age_cat", "sex",
    "priors_count", "is_recid", "c_jail_in", "c_jail_out",
]
RACE_MODEL_FEATURES = ["age_cat", "c_charge_degree", "sex", "priors_count", "length_of_stay"]
CROSS_VALIDATION_FOLDS = 5


@dataclass
class ThresholdExperiment:
    name: str
    model: LogisticRegression
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_train: pd.Series
    sensitive_test: pd.Series
    out_of_fold_probabilities: np.ndarray


def compute_out_of_fold_probabilities(X_train, y_train):
    return cross_val_predict(
        LogisticRegression(solver="liblinear"),
        X_train,
        y_train,
        cv=CROSS_VALIDATION_FOLDS,
        method="predict_proba",
    )[:, 1]


def assemble_experiment(name, X, y, sensitive, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    return ThresholdExperiment(
        name=name,
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive.loc[X_train.index],
        sensitive_test=sensitive.loc[X_test.index],
        out_of_fold_probabilities=compute_out_of_fold_probabilities(X_train, y_train),
    )


def build_race_experiment(data, random_state=RANDOM_STATE):
    race_data = encode_features(data[RACE_SOURCE_COLUMNS + [TARGET, "race"]])
    race_data = drop_rows_missing_jail_dates(race_data)
    race_data = add_length_of_stay(race_data)
    race_data = bin_priors_count(race_data)
    return assemble_experiment("race", race_data[RACE_MODEL_FEATURES], race_data[TARGET], race_data["race"], random_state)


def build_sex_experiment(data, random_state=RANDOM_STATE):
    dummies = pd.get_dummies(data[["age", "priors_count", "sex"]])
    X = dummies.drop(columns=["sex_Female", "sex_Male"])
    return assemble_experiment("sex", X, data[TARGET], data["sex"], random_state)


def build_age_experiment(data, random_state=RANDOM_STATE):
    data_age = data.assign(age_cat_mapped=data["age_cat"].map(AGE_CATEGORY_MAPPING))
    dummies = pd.get_dummies(data_age[["sex", "priors_count", "age_cat_mapped"]])
    X = dummies.drop(columns=["age_cat_mapped"])
    return assemble_experiment("age", X, data_age[TARGET], data_age["age_cat_mapped"], random_state)


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


def select_threshold(experiment, group_value, gamma, n_thresholds=1001):
    protected_train = (experiment.sensitive_train == group_value).to_numpy()
    return optimize_threshold_from_probabilities(
        experiment.out_of_fold_probabilities,
        experiment.y_train.to_numpy(),
        protected_train,
        gamma=gamma,
        n_thresholds=n_thresholds,
    )


def evaluate_threshold(experiment, group_value, threshold):
    probabilities = experiment.model.predict_proba(experiment.X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    protected = protected_mask(experiment, group_value)
    return {
        "accuracy": accuracy_score(experiment.y_test, predictions),
        "p_rule": p_rule(predictions[protected].mean(), predictions[~protected].mean()),
        "covariance": np.cov(protected, predictions)[0, 1],
    }


def fair_threshold_report(experiment, group_value, gamma, n_thresholds=1001):
    selection = select_threshold(experiment, group_value, gamma, n_thresholds)
    test_metrics = evaluate_threshold(experiment, group_value, selection["threshold"])
    return {"threshold": selection["threshold"], "selection_p_rule": selection["p_rule"], **test_metrics}


def sweep_gamma(experiment, group_value, gamma_values):
    rows = []
    for gamma_value in gamma_values:
        rows.append({"gamma": gamma_value, **fair_threshold_report(experiment, group_value, gamma_value)})
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
        rows.append({"age_group": age_group, **fair_threshold_report(experiment, age_group, gamma, n_thresholds)})
    return pd.DataFrame(rows)
