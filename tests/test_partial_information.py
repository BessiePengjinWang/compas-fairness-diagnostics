import math

import numpy as np
import pytest

from compas_fairness.partial_information import (
    outcome_protected_mutual_information,
    probability_table,
    shared_information,
    solve_minimum_conditional_information,
    unique_information_of_protected,
)

BITS_PER_NAT = 1 / math.log(2)
AND_OUTCOME = np.array([0, 0, 0, 1])
AND_PROTECTED = np.array([0, 0, 1, 1])
AND_STATE = np.array([[0], [1], [0], [1]])


def test_and_gate_has_published_shared_information_of_0_311_bits_and_no_unique_information():
    shared_bits = shared_information(AND_OUTCOME, AND_PROTECTED, AND_STATE) * BITS_PER_NAT
    unique_bits = unique_information_of_protected(AND_OUTCOME, AND_PROTECTED, AND_STATE) * BITS_PER_NAT
    assert shared_bits == pytest.approx(0.311278, abs=1e-5)
    assert unique_bits == pytest.approx(0.0, abs=1e-6)


def test_identical_copies_share_all_their_information():
    copy = np.array([0, 1, 0, 1])
    assert shared_information(copy, copy, copy.reshape(-1, 1)) == pytest.approx(math.log(2), abs=1e-6)


def test_information_only_in_the_protected_attribute_is_unique():
    outcome = np.array([0, 0, 1, 1])
    independent_state = np.array([[0], [1], [0], [1]])
    assert unique_information_of_protected(outcome, outcome, independent_state) == pytest.approx(math.log(2), abs=1e-6)
    assert shared_information(outcome, outcome, independent_state) == pytest.approx(0.0, abs=1e-6)


def test_xor_has_no_shared_and_no_unique_information():
    xor_outcome = np.array([0, 1, 1, 0])
    assert shared_information(xor_outcome, AND_PROTECTED, AND_STATE) == pytest.approx(0.0, abs=1e-6)
    assert unique_information_of_protected(xor_outcome, AND_PROTECTED, AND_STATE) == pytest.approx(0.0, abs=1e-6)


def random_table(seed, outcome_states, state_count):
    generator = np.random.default_rng(seed)
    table = generator.dirichlet(np.ones(outcome_states * 2 * state_count)).reshape(outcome_states, 2, state_count)
    return table


@pytest.mark.parametrize("seed", range(5))
def test_optimal_coupling_keeps_the_required_marginals(seed):
    table = random_table(seed, 2, 4)
    _, coupling = solve_minimum_conditional_information(table)
    assert np.allclose(coupling.sum(axis=2), table.sum(axis=2), atol=1e-8)
    assert np.allclose(coupling.sum(axis=1), table.sum(axis=1), atol=1e-8)
    assert (coupling >= -1e-12).all()


def conditional_information_of_coupling(coupling):
    state_mass = coupling.sum(axis=(0, 1))
    outcome_state = coupling.sum(axis=1)
    group_state = coupling.sum(axis=0)
    total = 0.0
    for t in range(coupling.shape[0]):
        for a in range(2):
            for x in range(coupling.shape[2]):
                value = coupling[t, a, x]
                if value > 0:
                    total += value * math.log(value * state_mass[x] / (outcome_state[t, x] * group_state[a, x]))
    return total


@pytest.mark.parametrize("seed", range(5))
def test_solver_matches_brute_force_grid_search_on_binary_tables(seed):
    table = random_table(seed, 2, 2)
    solver_value, _ = solve_minimum_conditional_information(table)
    outcome_state = table.sum(axis=1)
    group_mass = table.sum(axis=2)[:, 1]
    ranges = []
    for t in range(2):
        low = max(0.0, group_mass[t] - outcome_state[t, 1])
        high = min(outcome_state[t, 0], group_mass[t])
        ranges.append(np.linspace(low, high, 201))
    best = float("inf")
    for first in ranges[0]:
        for second in ranges[1]:
            one = np.array([[first, group_mass[0] - first], [second, group_mass[1] - second]])
            zero = outcome_state - one
            coupling = np.stack([zero, one], axis=1)
            if (coupling < -1e-15).any():
                continue
            best = min(best, conditional_information_of_coupling(np.maximum(coupling, 0.0)))
    assert solver_value == pytest.approx(best, abs=2e-5)


@pytest.mark.parametrize("seed", range(5))
def test_shared_plus_unique_equals_mutual_information_with_the_outcome(seed):
    generator = np.random.default_rng(seed)
    outcome = generator.integers(0, 2, size=400)
    protected = (outcome + generator.integers(0, 2, size=400) > 1).astype(int)
    state = np.column_stack([generator.integers(0, 2, size=400), (protected + generator.integers(0, 2, size=400)) % 3])
    table = probability_table(outcome, protected, state)
    total = shared_information(outcome, protected, state) + unique_information_of_protected(outcome, protected, state)
    assert total == pytest.approx(outcome_protected_mutual_information(table), abs=1e-7)


def test_protected_attribute_with_one_value_has_no_shared_or_unique_information():
    outcome = np.array([0, 1, 0, 1])
    protected = np.zeros(4, dtype=int)
    assert shared_information(outcome, protected, AND_STATE) == 0.0
    assert unique_information_of_protected(outcome, protected, AND_STATE) == 0.0
