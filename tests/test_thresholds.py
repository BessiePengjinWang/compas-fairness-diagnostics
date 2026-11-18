import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from compas_fairness.threshold_experiments import (
    ThresholdExperiment,
    compute_out_of_fold_probabilities,
    evaluate_threshold,
    fair_threshold_report,
)
from compas_fairness.thresholds import optimize_threshold_from_probabilities


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack([1 - self.probabilities, self.probabilities])

    def predict(self, features):
        return (self.probabilities > 0.5).astype(int)


def test_optimizer_picks_the_first_threshold_that_equalizes_rates_within_the_budget():
    probabilities = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    protected = np.array([False, False, True, True])
    best = optimize_threshold_from_probabilities(probabilities, labels, protected, gamma=0.5, n_thresholds=11)
    assert best["threshold"] == pytest.approx(0.0)
    assert best["p_rule"] == pytest.approx(100.0)
    assert best["accuracy"] == pytest.approx(0.5)


def test_optimizer_keeps_the_default_when_no_threshold_fits_the_accuracy_budget_and_is_fair():
    probabilities = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    protected = np.array([False, False, True, True])
    best = optimize_threshold_from_probabilities(probabilities, labels, protected, gamma=0.0, n_thresholds=11)
    assert best["threshold"] == pytest.approx(0.5)
    assert best["p_rule"] == pytest.approx(0.0)


def test_optimizer_does_not_crash_when_the_protected_group_gets_no_positive_prediction():
    probabilities = np.array([0.2, 0.2, 0.8, 0.8])
    labels = np.array([0, 0, 1, 1])
    protected = np.array([True, True, False, False])
    best = optimize_threshold_from_probabilities(probabilities, labels, protected, gamma=1.0, n_thresholds=11)
    assert best["p_rule"] == pytest.approx(100.0)


def test_evaluate_threshold_known_answer():
    experiment = ThresholdExperiment(
        name="stub",
        model=FixedProbabilityModel([0.9, 0.2, 0.8, 0.1]),
        X_train=pd.DataFrame({"f": [0.0]}),
        X_test=pd.DataFrame({"f": [0.0, 0.0, 0.0, 0.0]}),
        y_train=pd.Series([0]),
        y_test=pd.Series([1, 0, 0, 0]),
        sensitive_train=pd.Series([0]),
        sensitive_test=pd.Series([1, 1, 0, 0]),
        out_of_fold_probabilities=np.array([0.5]),
    )
    metrics = evaluate_threshold(experiment, 1, 0.5)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["p_rule"] == pytest.approx(100.0)
    assert metrics["covariance"] == pytest.approx(0.0)


def build_synthetic_experiment(test_labels):
    features, labels = make_classification(n_samples=600, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    groups = np.random.default_rng(1).integers(0, 2, size=600)
    frame = pd.DataFrame(features, columns=list("abcd"))
    train_index = frame.index[:400]
    test_index = frame.index[400:]
    X_train, X_test = frame.loc[train_index], frame.loc[test_index]
    y_train = pd.Series(labels[:400], index=train_index)
    y_test = pd.Series(test_labels(labels[400:]), index=test_index)
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    sensitive = pd.Series(groups, index=frame.index)
    return ThresholdExperiment(
        name="synthetic",
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive.loc[train_index],
        sensitive_test=sensitive.loc[test_index],
        out_of_fold_probabilities=compute_out_of_fold_probabilities(X_train, y_train),
    )


def test_selected_threshold_does_not_depend_on_test_labels():
    original = build_synthetic_experiment(lambda labels: labels)
    flipped = build_synthetic_experiment(lambda labels: 1 - labels)
    original_report = fair_threshold_report(original, 1, gamma=0.15, n_thresholds=101)
    flipped_report = fair_threshold_report(flipped, 1, gamma=0.15, n_thresholds=101)
    assert original_report["threshold"] == flipped_report["threshold"]
    assert original_report["selection_p_rule"] == flipped_report["selection_p_rule"]
    assert original_report["accuracy"] != flipped_report["accuracy"]
