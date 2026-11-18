import itertools
import math

import numpy as np
from sklearn.metrics import mutual_info_score


def conditional_mutual_info(df, target_col, feature_cols, condition_cols):
    if not condition_cols:
        return mutual_info_score(df[target_col], df[feature_cols])
    joint_cols = [feature_cols] + condition_cols if isinstance(feature_cols, str) else feature_cols + condition_cols
    joint = df[joint_cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    condition_only = df[condition_cols].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    return mutual_info_score(df[target_col], joint) - mutual_info_score(df[target_col], condition_only)


def shared_information(df, target_col, feature_cols, protected_col):
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    mi_target_protected = mutual_info_score(df[target_col], df[protected_col])
    joint = df[feature_cols + [protected_col]].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    mi_target_joint = mutual_info_score(df[target_col], joint)
    unique_info = mi_target_protected - mi_target_joint
    return max(mi_target_protected - unique_info, 0)


def unique_values_per_column(array):
    return [np.unique(array[:, col]).tolist() for col in range(array.shape[1])]


def unique_information(array_1, array_2):
    combined = np.concatenate((array_1, array_2), axis=1)
    total_info = 0.0
    for combo in itertools.product(*unique_values_per_column(combined)):
        joint_rate = np.mean((combined == combo).all(axis=1))
        rate_1 = np.mean((array_1 == combo[: array_1.shape[1]]).all(axis=1))
        rate_2 = np.mean((array_2 == combo[array_1.shape[1] :]).all(axis=1))
        if joint_rate > 0 and rate_1 > 0 and rate_2 > 0:
            total_info += abs(joint_rate * np.log(joint_rate / rate_1) / rate_1)
    return total_info


def unique_information_conditional(array_1, array_2, condition):
    combined = np.concatenate((array_1, array_2, condition), axis=1)
    total_info = 0.0
    for combo in itertools.product(*unique_values_per_column(combined)):
        joint_mask = (combined == combo).all(axis=1)
        mask_1 = (array_1 == combo[: array_1.shape[1]]).all(axis=1)
        mask_2_given_condition = (combined[:, array_1.shape[1] :] == combo[array_1.shape[1] :]).all(axis=1)

        joint_rate = np.mean(joint_mask)
        rate_1 = np.mean(mask_1)
        rate_2 = np.mean(mask_2_given_condition)
        rate_1_given_2 = np.mean(mask_1[mask_2_given_condition]) if np.any(mask_2_given_condition) else 0

        if joint_rate > 0 and rate_1 > 0 and rate_2 > 0 and rate_1_given_2 > 0:
            total_info += abs(joint_rate * np.log(joint_rate / rate_2) / rate_1_given_2)
    return total_info


def accuracy_coef(outcome, feature_subset, complement, protected):
    condition = np.concatenate((complement, protected), axis=1)
    return unique_information_conditional(outcome, feature_subset, condition)


def discrimination_coef(outcome, feature_subset, protected):
    feature_and_protected = np.concatenate((feature_subset, protected), axis=1)
    return (
        unique_information(outcome, feature_and_protected)
        * unique_information(feature_subset, protected)
        * unique_information_conditional(feature_subset, protected, outcome)
    )


def marginal_accuracy_coef(outcome, features, protected, target_feature_idx):
    n_features = features.shape[1]
    value = 0.0
    other_indices = [i for i in range(n_features) if i != target_feature_idx]
    for subset_size in range(n_features):
        for subset in itertools.combinations(other_indices, subset_size):
            weight = math.factorial(len(subset)) * math.factorial(n_features - len(subset) - 1) / math.factorial(n_features)
            with_feature = list(subset) + [target_feature_idx]
            complement_with = [i for i in range(n_features) if i not in with_feature]
            complement_without = [i for i in range(n_features) if i not in subset]
            v_with = accuracy_coef(outcome, features[:, with_feature], features[:, complement_with], protected)
            v_without = accuracy_coef(outcome, features[:, subset], features[:, complement_without], protected)
            value += weight * (v_with - v_without)
    return value


def marginal_discrimination_coef(outcome, features, protected, target_feature_idx):
    n_features = features.shape[1]
    value = 0.0
    other_indices = [i for i in range(n_features) if i != target_feature_idx]
    for subset_size in range(n_features):
        for subset in itertools.combinations(other_indices, subset_size):
            weight = math.factorial(len(subset)) * math.factorial(n_features - len(subset) - 1) / math.factorial(n_features)
            with_feature = list(subset) + [target_feature_idx]
            v_with = discrimination_coef(outcome, features[:, with_feature], protected)
            v_without = discrimination_coef(outcome, features[:, subset], protected)
            value += weight * (v_with - v_without)
    return value


def fairness_utility_score(accuracy_coefs, discrimination_coefs, alpha):
    return accuracy_coefs - alpha * discrimination_coefs
