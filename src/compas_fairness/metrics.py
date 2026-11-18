def p_rule(protected_rate, non_protected_rate):
    return min(protected_rate / non_protected_rate, non_protected_rate / protected_rate) * 100


def calibration_gap(protected, y_pred, y_true):
    accuracies = []
    for group in (1, 0):
        mask = protected == group
        accuracies.append((y_pred[mask] == y_true[mask]).mean())
    return abs(accuracies[0] - accuracies[1])
