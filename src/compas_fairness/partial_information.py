import numpy as np
from scipy.optimize import minimize

PROBABILITY_FLOOR = 1e-12


def as_two_dimensional(values):
    array = np.asarray(values)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def probability_table(outcome, protected, features):
    outcome_index = np.unique(np.asarray(outcome), return_inverse=True)[1].reshape(-1)
    protected_values, protected_index = np.unique(np.asarray(protected), return_inverse=True)
    protected_index = protected_index.reshape(-1)
    if len(protected_values) > 2:
        raise ValueError("the protected attribute must have at most two values")
    feature_array = as_two_dimensional(features)
    if feature_array.shape[1] == 0:
        state_index = np.zeros(len(outcome_index), dtype=int)
    else:
        state_index = np.unique(feature_array, axis=0, return_inverse=True)[1].reshape(-1)
    table = np.zeros((outcome_index.max() + 1, 2, state_index.max() + 1))
    np.add.at(table, (outcome_index, protected_index, state_index), 1.0)
    return table / table.sum()


def xlogx(values):
    safe = np.where(values > 0, values, 1.0)
    return np.where(values > 0, values * np.log(safe), 0.0)


def safe_log(values):
    return np.log(np.maximum(values, 1e-300))


def outcome_protected_mutual_information(table):
    joint = table.sum(axis=2)
    outcome_mass = joint.sum(axis=1, keepdims=True)
    group_mass = joint.sum(axis=0, keepdims=True)
    independent = outcome_mass * group_mass
    mask = joint > 0
    return float((joint[mask] * np.log(joint[mask] / independent[mask])).sum())


def outcome_entropy_given_state(table):
    outcome_state_mass = table.sum(axis=1)
    state_mass = outcome_state_mass.sum(axis=0, keepdims=True)
    mask = outcome_state_mass > 0
    return float(-(outcome_state_mass[mask] * np.log(outcome_state_mass[mask] / np.broadcast_to(state_mass, outcome_state_mass.shape)[mask])).sum())


def solve_minimum_conditional_information(table):
    outcome_state_mass = table.sum(axis=1)
    outcome_group_mass = table.sum(axis=2)
    outcome_mass = outcome_state_mass.sum(axis=1)
    active = outcome_state_mass > 0
    active_count = int(active.sum())

    def expand(flat):
        grid = np.zeros_like(outcome_state_mass)
        grid[active] = flat
        return grid

    def objective(flat):
        group_one = expand(flat)
        group_zero = outcome_state_mass - group_one
        return float(
            xlogx(group_one).sum() + xlogx(group_zero).sum()
            - xlogx(group_one.sum(axis=0)).sum() - xlogx(group_zero.sum(axis=0)).sum()
        )

    def gradient(flat):
        group_one = expand(flat)
        group_zero = outcome_state_mass - group_one
        state_one = group_one.sum(axis=0)[None, :]
        state_zero = group_zero.sum(axis=0)[None, :]
        full = safe_log(group_one) - safe_log(group_zero) - safe_log(state_one) + safe_log(state_zero)
        return full[active]

    row_index = np.nonzero(active)[0]
    equality_matrix = np.zeros((outcome_state_mass.shape[0], active_count))
    equality_matrix[row_index, np.arange(active_count)] = 1.0
    equality_target = outcome_group_mass[:, 1]

    initial_grid = outcome_state_mass * (outcome_group_mass[:, 1] / np.where(outcome_mass > 0, outcome_mass, 1.0))[:, None]
    masses = outcome_state_mass[active]
    floors = np.minimum(PROBABILITY_FLOOR, masses / 4)
    bounds = list(zip(floors, masses - floors))
    start = np.clip(initial_grid[active], floors, masses - floors)

    constraints = [{"type": "eq", "fun": lambda flat: equality_matrix @ flat - equality_target, "jac": lambda flat: equality_matrix}]
    result = minimize(objective, start, jac=gradient, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-15})

    group_one = expand(result.x)
    coupling = np.zeros_like(table)
    coupling[:, 1, :] = group_one
    coupling[:, 0, :] = outcome_state_mass - group_one
    unique_information = max(outcome_entropy_given_state(table) + objective(result.x), 0.0)
    return unique_information, coupling


def unique_information_of_protected(outcome, protected, features):
    table = probability_table(outcome, protected, features)
    if np.unique(np.asarray(protected)).size < 2 or table.shape[0] < 2:
        return 0.0
    return solve_minimum_conditional_information(table)[0]


def shared_information(outcome, protected, features):
    table = probability_table(outcome, protected, features)
    if np.unique(np.asarray(protected)).size < 2 or table.shape[0] < 2:
        return 0.0
    mutual_information = outcome_protected_mutual_information(table)
    unique_information = solve_minimum_conditional_information(table)[0]
    return max(mutual_information - unique_information, 0.0)
