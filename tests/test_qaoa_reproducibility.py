"""Reproducibility tests for the QAOA parameter optimizer.

The optimizer must draw its initial variational parameters from a seeded
generator so repeated runs on the same instance return bit-identical
(gamma, beta, energy) tuples, and must leave the global NumPy RNG untouched.
"""

import numpy as np
import pytest

from ising_hamiltonian_converter import IsingHamiltonian
from QAOA_solver import DEFAULT_QAOA_SEED, QAOASolver


def _reference_model() -> IsingHamiltonian:
    return IsingHamiltonian(
        h={0: 0.5, 1: -0.8, 2: 0.3},
        J={(0, 1): -1.2, (1, 2): 0.9},
        offset=2.0,
    )


def test_optimize_parameters_is_reproducible_for_fixed_seed():
    solver = QAOASolver(_reference_model(), p=2)
    gamma_a, beta_a, energy_a = solver.optimize_parameters(max_iter=12)
    gamma_b, beta_b, energy_b = solver.optimize_parameters(max_iter=12)
    assert np.array_equal(gamma_a, gamma_b)
    assert np.array_equal(beta_a, beta_b)
    assert energy_a == energy_b


def test_optimize_parameters_reproducible_across_fresh_solver_instances():
    gamma_a, beta_a, energy_a = QAOASolver(_reference_model(), p=2).optimize_parameters(max_iter=12)
    gamma_b, beta_b, energy_b = QAOASolver(_reference_model(), p=2).optimize_parameters(max_iter=12)
    assert np.array_equal(gamma_a, gamma_b)
    assert np.array_equal(beta_a, beta_b)
    assert energy_a == energy_b


def test_default_seed_is_the_documented_constant():
    solver = QAOASolver(_reference_model(), p=2)
    gamma_a, beta_a, energy_a = solver.optimize_parameters(max_iter=12)
    gamma_b, beta_b, energy_b = solver.optimize_parameters(max_iter=12, seed=DEFAULT_QAOA_SEED)
    assert np.array_equal(gamma_a, gamma_b)
    assert np.array_equal(beta_a, beta_b)
    assert energy_a == energy_b


def test_seed_override_changes_the_optimization_trajectory():
    gamma_a, beta_a, energy_a = QAOASolver(_reference_model(), p=2).optimize_parameters(max_iter=12)
    gamma_b, beta_b, energy_b = QAOASolver(_reference_model(), p=2).optimize_parameters(max_iter=12, seed=777)
    seeded_changed = not (np.array_equal(gamma_a, gamma_b) and np.array_equal(beta_a, beta_b))
    assert seeded_changed


def test_seeded_optimization_leaves_global_numpy_rng_untouched():
    np.random.seed(0)
    expected_state = np.random.get_state()
    QAOASolver(_reference_model(), p=2).optimize_parameters(max_iter=4)
    actual_state = np.random.get_state()
    assert np.array_equal(actual_state[1], expected_state[1])
    assert actual_state[0] == expected_state[0]
    assert actual_state[2] == expected_state[2]
    assert actual_state[3] == expected_state[3]
    assert actual_state[4] == expected_state[4]


def test_cobyla_optimizer_is_deterministic_for_fixed_seed():
    gamma_a, beta_a, energy_a = QAOASolver(_reference_model(), p=2).optimize_parameters_cobyla(max_iter=30)
    gamma_b, beta_b, energy_b = QAOASolver(_reference_model(), p=2).optimize_parameters_cobyla(max_iter=30)
    assert np.array_equal(gamma_a, gamma_b)
    assert np.array_equal(beta_a, beta_b)
    assert energy_a == energy_b
