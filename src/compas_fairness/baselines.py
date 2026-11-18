import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from compas_fairness.metrics import p_rule
from compas_fairness.threshold_experiments import protected_mask

COMPAS_HIGH_RISK_CUTOFF = 5


def compas_scores_for_rows(data, row_index):
    return data["decile_score"].loc[row_index].to_numpy()


def baseline_row(method, predictions, scores, y_test, protected):
    protected_rate = predictions[protected].mean()
    non_protected_rate = predictions[~protected].mean()
    return {
        "method": method,
        "accuracy": accuracy_score(y_test, predictions),
        "p_rule": p_rule(protected_rate, non_protected_rate),
        "auc": roc_auc_score(y_test, scores),
    }


def baseline_report(experiment, data, group_value, cutoff=COMPAS_HIGH_RISK_CUTOFF):
    y_test = experiment.y_test.to_numpy()
    protected = protected_mask(experiment, group_value)
    majority_class = int(experiment.y_train.mean() >= 0.5)
    majority_predictions = np.full(len(y_test), majority_class)
    compas_scores = compas_scores_for_rows(data, experiment.X_test.index)
    model_probabilities = experiment.model.predict_proba(experiment.X_test)[:, 1]
    rows = [
        baseline_row("always predict the majority class", majority_predictions, np.full(len(y_test), 0.5), y_test, protected),
        baseline_row(f"COMPAS decile score >= {cutoff}", (compas_scores >= cutoff).astype(int), compas_scores, y_test, protected),
        baseline_row("logistic regression (project model)", experiment.model.predict(experiment.X_test), model_probabilities, y_test, protected),
    ]
    return pd.DataFrame(rows)


def majority_class_accuracy(experiment):
    majority_class = int(experiment.y_train.mean() >= 0.5)
    return accuracy_score(experiment.y_test.to_numpy(), np.full(len(experiment.y_test), majority_class))
