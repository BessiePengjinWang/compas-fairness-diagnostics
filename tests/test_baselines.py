import numpy as np
import pandas as pd
import pytest

from compas_fairness.baselines import baseline_report
from compas_fairness.threshold_experiments import ThresholdExperiment


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack([1 - self.probabilities, self.probabilities])

    def predict(self, features):
        return (self.probabilities > 0.5).astype(int)


def build_stub_experiment():
    return ThresholdExperiment(
        name="stub",
        model=FixedProbabilityModel([0.9, 0.6, 0.4, 0.1]),
        X_train=pd.DataFrame({"f": [0.0, 0.0, 0.0]}, index=[10, 11, 12]),
        X_test=pd.DataFrame({"f": [0.0, 0.0, 0.0, 0.0]}, index=[0, 1, 2, 3]),
        y_train=pd.Series([1, 1, 0], index=[10, 11, 12]),
        y_test=pd.Series([1, 0, 1, 0], index=[0, 1, 2, 3]),
        sensitive_train=pd.Series([1, 0, 1], index=[10, 11, 12]),
        sensitive_test=pd.Series([1, 1, 0, 0], index=[0, 1, 2, 3]),
        out_of_fold_probabilities=np.array([0.5, 0.5, 0.5]),
    )


def test_baseline_report_known_answers():
    data = pd.DataFrame({"decile_score": [8, 3, 6, 2]}, index=[0, 1, 2, 3])
    report = baseline_report(build_stub_experiment(), data, 1).set_index("method")
    majority = report.loc["always predict the majority class"]
    assert majority["accuracy"] == pytest.approx(0.5)
    assert majority["auc"] == pytest.approx(0.5)
    compas = report.loc["COMPAS decile score >= 5"]
    assert compas["accuracy"] == pytest.approx(1.0)
    assert compas["p_rule"] == pytest.approx(100.0)
    assert compas["auc"] == pytest.approx(1.0)
    model = report.loc["logistic regression (project model)"]
    assert model["accuracy"] == pytest.approx(0.5)
    assert model["auc"] == pytest.approx(0.75)
