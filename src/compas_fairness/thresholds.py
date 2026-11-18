import numpy as np
from sklearn.metrics import accuracy_score

from compas_fairness.metrics import p_rule


def optimize_threshold_for_fairness(model, X_test, y_test, protected, gamma=0.0, n_thresholds=1001):
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    target_accuracy = baseline_accuracy * (1 - gamma)
    probabilities = model.predict_proba(X_test)[:, 1]

    best = {"threshold": 0.5, "accuracy": baseline_accuracy, "p_rule": 0.0, "covariance": float("inf")}

    for threshold in np.linspace(0, 1, n_thresholds):
        y_pred = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy < target_accuracy:
            continue

        protected_rate = y_pred[protected].mean()
        non_protected_rate = y_pred[~protected].mean()
        if non_protected_rate == 0:
            continue

        current_p_rule = p_rule(protected_rate, non_protected_rate)
        current_covariance = np.cov(protected, y_pred)[0, 1]

        is_better = current_p_rule > best["p_rule"] or (
            current_p_rule == best["p_rule"] and abs(current_covariance) < abs(best["covariance"])
        )
        if is_better:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "p_rule": current_p_rule,
                "covariance": current_covariance,
            }

    return best
