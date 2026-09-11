"""Multi-depth (p >= 1) QAOA support tests.

optimize_parameters must work for any depth p >= 1 with a parameter vector of
length 2p (p cost angles gamma + p mixer angles beta), and higher depth must
improve or match the p=1 expectation energy on a fixed instance with a fixed
seed. sweep_qaoa_depth must run the COBYLA p-sweep and select the best depth.
"""

import numpy as np
import pytest

from ising_hamiltonian_converter import IsingHamiltonian
from QAOA_solver import DEFAULT_QAOA_SEED, QAOASolver, sweep_qaoa_depth


def _reference_model() -> IsingHamiltonian:
    return IsingHamiltonian(
        h={0: 0.5, 1: -0.8, 2: 0.3},
        J={(0, 1): -1.2, (1, 2): 0.9},
        offset=2.0,
    )


def test_invalid_depth_rejected():
    with pytest.raises(ValueError):
        QAOASolver(_reference_model(), p=0)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_parameter_vector_length_is_two_p(p):
    solver = QAOASolver(_reference_model(), p=p)
    gamma, beta, _energy = solver.optimize_parameters(max_iter=8, seed=DEFAULT_QAOA_SEED)
    assert len(gamma) == p
    assert len(beta) == p
    assert solver.p == p


def test_p2_improves_or_matches_p1_energy_on_fixed_instance():
    solver_p1 = QAOASolver(_reference_model(), p=1)
    _g1, _b1, energy_p1 = solver_p1.optimize_parameters(max_iter=25, seed=DEFAULT_QAOA_SEED)
    solver_p2 = QAOASolver(_reference_model(), p=2)
    _g2, _b2, energy_p2 = solver_p2.optimize_parameters(max_iter=25, seed=DEFAULT_QAOA_SEED)
    assert len(_g2) == 2 and len(_b2) == 2
    assert energy_p2 <= energy_p1 + 1e-9


def test_cobyla_p2_improves_or_matches_p1_energy():
    solver_p1 = QAOASolver(_reference_model(), p=1)
    _g1, _b1, energy_p1 = solver_p1.optimize_parameters_cobyla(max_iter=60, seed=DEFAULT_QAOA_SEED)
    solver_p2 = QAOASolver(_reference_model(), p=2)
    _g2, _b2, energy_p2 = solver_p2.optimize_parameters_cobyla(max_iter=60, seed=DEFAULT_QAOA_SEED)
    assert energy_p2 <= energy_p1 + 1e-9


def test_sweep_qaoa_depth_returns_all_depths_and_best_selection():
    sweep = sweep_qaoa_depth(_reference_model(), p_values=(1, 2, 3), budget_per_p=40, seed=DEFAULT_QAOA_SEED)
    assert [entry["p"] for entry in sweep["results"]] == [1, 2, 3]
    best_ratio = max(entry["approx_ratio"] for entry in sweep["results"])
    assert sweep["best"]["approx_ratio"] == pytest.approx(best_ratio)
    for entry in sweep["results"]:
        assert len(entry["gamma"]) == entry["p"]
        assert len(entry["beta"]) == entry["p"]
    # basis-state energies are shared across depths (identical min/max)
    solver_p1 = QAOASolver(_reference_model(), p=1)
    assert sweep["e_min"] == pytest.approx(solver_p1.e_min)
    assert sweep["e_max"] == pytest.approx(solver_p1.e_max)


def test_sweep_qaoa_depth_reproducible():
    sweep_a = sweep_qaoa_depth(_reference_model(), p_values=(1, 2), budget_per_p=30, seed=DEFAULT_QAOA_SEED)
    sweep_b = sweep_qaoa_depth(_reference_model(), p_values=(1, 2), budget_per_p=30, seed=DEFAULT_QAOA_SEED)
    assert sweep_a["best"]["p"] == sweep_b["best"]["p"]
    assert sweep_a["best"]["energy"] == pytest.approx(sweep_b["best"]["energy"])
    assert sweep_a["best"]["gamma"] == sweep_b["best"]["gamma"]
    assert sweep_a["best"]["beta"] == sweep_b["best"]["beta"]
