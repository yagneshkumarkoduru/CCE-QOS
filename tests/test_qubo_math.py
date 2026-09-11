"""Mathematical property tests for the QUBO generator and CP-SAT scheduler.

These tests pin the corrected one-hot QUBO algebra (diagonal -penalty,
symmetric off-diagonal +penalty) and the CP-SAT scheduler's resource model.
"""

import numpy as np
import pytest

from implementations.v1_exact_cpsat_solver.ortools_cpsat_engine import ExactCPSATScheduler
from implementations.v2_cce_qubo_apr_engine.qubo_hamiltonian_generator import CCEQUBOGenerator


def _group_energy(Q: np.ndarray, bits: np.ndarray) -> float:
    return float(bits @ Q @ bits)


def test_one_hot_all_zeros_energy_beats_valid_assignments():
    """Critical property: the all-zeros vector must have HIGHER energy than any
    valid one-hot assignment, otherwise the solver can 'cheat' by scheduling
    nothing (the exact bug this regression test pins)."""
    penalty = 50.0
    gen = CCEQUBOGenerator(num_tasks=1, num_slots=2, num_banks=1)
    Q = gen.build_hamiltonian({"lambda_one_hot": penalty})

    zero = np.array([0, 0])
    valids = [np.array([1, 0]), np.array([0, 1])]
    invalids = [np.array([1, 1])]

    e_zero = _group_energy(Q, zero)
    e_valid = [_group_energy(Q, v) for v in valids]
    e_invalid = [_group_energy(Q, v) for v in invalids]

    # All-zeros strictly worse than every valid one-hot assignment.
    assert all(e_zero > ev for ev in e_valid)
    # Any multi-selection is also strictly worse than every valid assignment.
    assert all(ei > ev for ei in e_invalid for ev in e_valid)
    # Exact energies for a 2-variable group: valid = -penalty, zeros = 0,
    # double = 0 (both 50 higher than a valid assignment).
    assert e_zero == pytest.approx(0.0)
    assert e_valid[0] == pytest.approx(-penalty)
    assert e_valid[1] == pytest.approx(-penalty)
    assert e_invalid[0] == pytest.approx(0.0)


def test_one_hot_symmetric_assignments_equal_energy():
    """Valid assignments that are permutations of each other share energy."""
    penalty = 50.0
    gen = CCEQUBOGenerator(num_tasks=1, num_slots=3, num_banks=2)
    Q = gen.build_hamiltonian({"lambda_one_hot": penalty})
    n = gen.num_vars

    # Every valid one-hot selection within the group must have equal energy.
    base = None
    for idx in range(n):
        bits = np.zeros(n)
        bits[idx] = 1.0
        e = _group_energy(Q, bits)
        if base is None:
            base = e
        assert e == pytest.approx(base)

    # And the Q matrix must be symmetric.
    assert np.allclose(Q, Q.T)


def test_one_hot_multi_task_independent_groups():
    """Two tasks each with a private single slot: scheduling both is valid and
    strictly better than leaving a task unassigned (penalty is per task)."""
    penalty = 50.0
    gen = CCEQUBOGenerator(num_tasks=2, num_slots=1, num_banks=1)
    Q = gen.build_hamiltonian({"lambda_one_hot": penalty})

    both = np.array([1, 1])   # both tasks scheduled once: valid
    only_a = np.array([1, 0])  # task 1 unassigned: one-hot violated
    zero = np.array([0, 0])

    assert _group_energy(Q, only_a) == pytest.approx(-penalty)
    assert _group_energy(Q, zero) == pytest.approx(0.0)
    assert _group_energy(Q, both) == pytest.approx(-2.0 * penalty)
    assert _group_energy(Q, both) < _group_energy(Q, only_a) < _group_energy(Q, zero)
    # No cross-task coupling: unrelated groups do not interfere.
    assert Q[0, 1] == pytest.approx(0.0)


def test_cpsat_serializes_tasks_exceeding_single_bank():
    """With a single SRAM bank, tasks cannot overlap: the exact scheduler must
    serialize them (makespan at least the sum of durations)."""
    tasks = [
        {"id": "a", "duration": 3, "memory_kb": 16},
        {"id": "b", "duration": 4, "memory_kb": 16},
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=8, num_banks=1)
    result = scheduler.solve(tasks, [])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["makespan"] >= 7
    assert set(result["banks"].values()) == {0}


def test_cpsat_uses_two_banks_in_parallel():
    """With two SRAM banks, independent tasks run concurrently: makespan is
    strictly less than the serialized sum of durations."""
    tasks = [
        {"id": "a", "duration": 3, "memory_kb": 16},
        {"id": "b", "duration": 4, "memory_kb": 16},
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=8, num_banks=2)
    result = scheduler.solve(tasks, [])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["makespan"] < 7  # strictly better than serialization
    assert result["banks"]["a"] != result["banks"]["b"]


def test_cpsat_respects_dag_precedence():
    """A dependency chain forces ordering regardless of bank availability."""
    tasks = [
        {"id": "a", "duration": 2, "memory_kb": 8},
        {"id": "b", "duration": 3, "memory_kb": 8},
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=64, num_banks=4)
    result = scheduler.solve(tasks, [("a", "b")])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["schedule"]["b"] >= result["schedule"]["a"] + 2
