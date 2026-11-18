import itertools
import math

import numpy as np
import pandas as pd

from compas_fairness.partial_information import shared_information


def as_integer_matrix(values):
    if isinstance(values, pd.DataFrame):
        columns = [pd.factorize(values[name])[0] for name in values.columns]
    elif isinstance(values, pd.Series):
        columns = [pd.factorize(values)[0]]
    else:
        array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        columns = [pd.factorize(array[:, index])[0] for index in range(array.shape[1])]
    if not columns:
        return np.zeros((0, 0), dtype=int)
    return np.column_stack(columns).astype(int)


def joint_entropy(values):
    matrix = as_integer_matrix(values)
    if matrix.shape[1] == 0:
        return 0.0
    _, counts = np.unique(matrix, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def stack_columns(*parts):
    matrices = [as_integer_matrix(part) for part in parts]
    row_count = max(matrix.shape[0] for matrix in matrices)
    matrices = [matrix if matrix.shape[1] > 0 else np.zeros((row_count, 0), dtype=int) for matrix in matrices]
    return np.hstack(matrices)


def mutual_information(first, second):
    value = joint_entropy(first) + joint_entropy(second) - joint_entropy(stack_columns(first, second))
    return max(value, 0.0)


def conditional_mutual_information(first, second, given):
    value = (
        joint_entropy(stack_columns(first, given))
        + joint_entropy(stack_columns(second, given))
        - joint_entropy(given)
        - joint_entropy(stack_columns(first, second, given))
    )
    return max(value, 0.0)


def accuracy_coefficient(outcome, features, protected, subset):
    if len(subset) == 0:
        return 0.0
    complement = [index for index in range(features.shape[1]) if index not in subset]
    conditioning = stack_columns(features[:, complement], protected)
    return conditional_mutual_information(outcome, features[:, list(subset)], conditioning)


def discrimination_coefficient(outcome, features, protected, subset):
    if len(subset) == 0:
        return 0.0
    subset_values = features[:, list(subset)]
    shared = shared_information(np.asarray(outcome).reshape(-1), np.asarray(protected).reshape(-1), subset_values)
    return (
        shared
        * mutual_information(subset_values, protected)
        * conditional_mutual_information(subset_values, protected, outcome)
    )


def values_for_all_subsets(characteristic_function, feature_count):
    values = {}
    for size in range(feature_count + 1):
        for subset in itertools.combinations(range(feature_count), size):
            values[subset] = characteristic_function(subset)
    return values


def shapley_values(subset_values, feature_count):
    result = np.zeros(feature_count)
    for index in range(feature_count):
        others = [other for other in range(feature_count) if other != index]
        for size in range(feature_count):
            weight = math.factorial(size) * math.factorial(feature_count - size - 1) / math.factorial(feature_count)
            for subset in itertools.combinations(others, size):
                with_feature = tuple(sorted(subset + (index,)))
                result[index] += weight * (subset_values[with_feature] - subset_values[subset])
    return result


def fairness_utility_score(accuracy_coefficients, discrimination_coefficients, alpha):
    return accuracy_coefficients - alpha * discrimination_coefficients
