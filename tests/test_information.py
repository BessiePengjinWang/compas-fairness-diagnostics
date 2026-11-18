import itertools
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score

from compas_fairness.information import (
    accuracy_coefficient,
    as_integer_matrix,
    conditional_mutual_information,
    discrimination_coefficient,
    fairness_utility_score,
    mutual_information,
    permutation_null_conditional_information,
    shapley_values,
    values_for_all_subsets,
)

COINS = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "copy_of_a": [0, 0, 1, 1]})


def random_dataset(seed, row_count=500):
    generator = np.random.default_rng(seed)
    protected = generator.integers(0, 2, size=row_count)
    features = np.column_stack([
        generator.integers(0, 3, size=row_count),
        (protected + generator.integers(0, 2, size=row_count)) % 2,
        generator.integers(0, 2, size=row_count),
    ])
    outcome = ((features[:, 0] + features[:, 1] + generator.integers(0, 2, size=row_count)) > 2).astype(int)
    return outcome.reshape(-1, 1), features, protected.reshape(-1, 1)


def test_mutual_information_known_answers():
    assert mutual_information(COINS["a"], COINS["b"]) == pytest.approx(0.0, abs=1e-12)
    assert mutual_information(COINS["a"], COINS["copy_of_a"]) == pytest.approx(math.log(2), abs=1e-12)
    xor = pd.DataFrame({"r1": [0, 0, 1, 1], "r2": [0, 1, 0, 1]})
    xor["t"] = xor["r1"] ^ xor["r2"]
    assert mutual_information(xor["t"], xor["r1"]) == pytest.approx(0.0, abs=1e-12)
    assert mutual_information(xor["t"], xor[["r1", "r2"]]) == pytest.approx(math.log(2), abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_mutual_information_matches_scikit_learn(seed):
    generator = np.random.default_rng(seed)
    first = generator.integers(0, 4, size=300)
    second = (first + generator.integers(0, 3, size=300)) % 5
    assert mutual_information(first, second) == pytest.approx(mutual_info_score(first, second), abs=1e-10)


def test_conditional_mutual_information_known_answers():
    constant = np.zeros(4, dtype=int)
    assert conditional_mutual_information(COINS["a"], COINS["b"], constant) == pytest.approx(0.0, abs=1e-12)
    independent_condition = pd.Series([0, 1, 0, 1])
    assert conditional_mutual_information(COINS["a"], COINS["copy_of_a"], independent_condition) == pytest.approx(math.log(2), abs=1e-12)


def test_conditional_mutual_information_is_zero_for_conditionally_independent_variables():
    rows = [(z, x, y) for z in (0, 1) for x in (0, 1) for y in (0, 1)]
    frame = pd.DataFrame(rows, columns=["z", "x", "y"])
    assert conditional_mutual_information(frame["x"], frame["y"], frame["z"]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_chain_rule_holds(seed):
    outcome, features, _ = random_dataset(seed)
    left = mutual_information(outcome, features[:, :2])
    right = mutual_information(outcome, features[:, [1]]) + conditional_mutual_information(outcome, features[:, [0]], features[:, [1]])
    assert left == pytest.approx(right, abs=1e-10)


def test_as_integer_matrix_accepts_strings_and_data_frames():
    frame = pd.DataFrame({"sex": ["Male", "Female", "Male"], "age": [30, 40, 30]})
    matrix = as_integer_matrix(frame)
    assert matrix.shape == (3, 2)
    assert matrix[0, 0] == matrix[2, 0] != matrix[1, 0]


def test_accuracy_coefficient_is_zero_exactly_when_the_subset_is_not_a_direct_cause():
    rows = [(x0, x1, a) for x0 in (0, 1) for x1 in (0, 1) for a in (0, 1)]
    frame = pd.DataFrame(rows, columns=["x0", "x1", "a"])
    outcome = frame[["x0"]].to_numpy()
    features = frame[["x0", "x1"]].to_numpy()
    protected = frame[["a"]].to_numpy()
    assert accuracy_coefficient(outcome, features, protected, (1,)) == pytest.approx(0.0, abs=1e-12)
    assert accuracy_coefficient(outcome, features, protected, (0,)) == pytest.approx(math.log(2), abs=1e-12)
    assert accuracy_coefficient(outcome, features, protected, ()) == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_accuracy_and_discrimination_coefficients_are_non_negative_and_monotone(seed):
    outcome, features, protected = random_dataset(seed)
    for coefficient in (accuracy_coefficient, discrimination_coefficient):
        values = values_for_all_subsets(lambda subset: coefficient(outcome, features, protected, subset), features.shape[1])
        for subset, value in values.items():
            assert value >= -1e-12
            for extra in range(features.shape[1]):
                if extra not in subset:
                    larger = tuple(sorted(subset + (extra,)))
                    assert values[larger] >= value - 1e-7


def test_discrimination_coefficient_is_zero_when_the_subset_is_independent_of_the_protected_attribute():
    rows = [(x, a, y) for x in (0, 1, 2) for a in (0, 1) for y in (0, 1)]
    frame = pd.DataFrame(rows, columns=["x", "a", "y"])
    value = discrimination_coefficient(frame[["y"]].to_numpy(), frame[["x"]].to_numpy(), frame[["a"]].to_numpy(), (0,))
    assert value == pytest.approx(0.0, abs=1e-12)


def test_shapley_values_of_an_additive_game_equal_the_weights():
    weights = [0.5, 2.0, 1.5]
    values = values_for_all_subsets(lambda subset: sum(weights[index] for index in subset), 3)
    assert shapley_values(values, 3) == pytest.approx(weights)


def test_shapley_values_are_symmetric_for_interchangeable_players():
    values = values_for_all_subsets(lambda subset: len(subset) ** 2, 4)
    result = shapley_values(values, 4)
    assert result == pytest.approx([4.0, 4.0, 4.0, 4.0])


@pytest.mark.parametrize("seed", range(3))
def test_shapley_values_satisfy_efficiency_and_non_negativity(seed):
    outcome, features, protected = random_dataset(seed)
    feature_count = features.shape[1]
    for coefficient in (accuracy_coefficient, discrimination_coefficient):
        values = values_for_all_subsets(lambda subset: coefficient(outcome, features, protected, subset), feature_count)
        result = shapley_values(values, feature_count)
        assert result.sum() == pytest.approx(values[tuple(range(feature_count))], abs=1e-10)
        assert (result >= -1e-10).all()


def test_fairness_utility_score_rewards_accuracy_and_penalizes_discrimination():
    accuracy = np.array([0.04, 0.02])
    discrimination = np.array([0.01, 0.0])
    assert fairness_utility_score(accuracy, discrimination, alpha=2.0) == pytest.approx([0.02, 0.02])


def test_permutation_null_shows_the_upward_bias_of_the_estimator_with_many_small_strata():
    generator = np.random.default_rng(0)
    row_count = 300
    first = generator.integers(0, 2, size=row_count)
    second = generator.integers(0, 2, size=row_count)
    given = np.repeat(np.arange(50), 6)
    observed = conditional_mutual_information(first, second, given)
    null_values = permutation_null_conditional_information(first, second, given, permutations=100, seed=1)
    assert null_values.mean() > 0.05
    assert abs(observed - null_values.mean()) < 3 * null_values.std() + 1e-12


def test_permutation_null_is_far_below_the_observed_value_for_a_real_dependence():
    generator = np.random.default_rng(2)
    first = generator.integers(0, 2, size=400)
    second = first.copy()
    given = generator.integers(0, 3, size=400)
    observed = conditional_mutual_information(first, second, given)
    null_values = permutation_null_conditional_information(first, second, given, permutations=100, seed=3)
    assert observed > null_values.max()


def test_permutation_null_never_moves_values_across_strata():
    first = np.array([0, 1, 0, 1, 0, 1])
    second = np.array([0, 0, 0, 1, 1, 1])
    given = np.array([0, 0, 0, 1, 1, 1])
    null_values = permutation_null_conditional_information(first, second, given, permutations=20, seed=0)
    assert (null_values >= 0).all()
    constant_second = np.zeros(6, dtype=int)
    assert permutation_null_conditional_information(first, constant_second, given, permutations=5, seed=0) == pytest.approx(np.zeros(5))
