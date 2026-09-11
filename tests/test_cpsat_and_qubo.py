"""Tests for the corrected CP-SAT scheduler and QUBO generator.
These ensure resource constraints and penalty calculations behave as expected.
"""

import numpy as np
import pytest
from implementations.v1_exact_cpsat_solver.ortools_cpsat_engine import ExactCPSATScheduler
from implementations.v2_cce_qubo_apr_engine.qubo_hamiltonian_generator import CCEQUBOGenerator

def test_cpsat_no_overlap_single_bank():
    tasks = [
        {"id": "a", "duration": 3, "memory_kb": 8},
        {"id": "b", "duration": 4, "memory_kb": 8},
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=8, num_banks=1)
    result = scheduler.solve(tasks, [])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["makespan"] >= 7
    assert result["banks"] == {"a": 0, "b": 0}

def test_cpsat_overlap_multiple_banks():
    tasks = [
        {"id": "a", "duration": 3, "memory_kb": 8},
        {"id": "b", "duration": 4, "memory_kb": 8},
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=8, num_banks=2)
    result = scheduler.solve(tasks, [])
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    assert result["makespan"] >= 4
    assert result["banks"]["a"] != result["banks"]["b"]

def test_qubo_one_hot_penalty():
    # Single task with two slots: selecting both slots (invalid) must cost more
    # than selecting exactly one (valid).
    generator = CCEQUBOGenerator(num_tasks=1, num_slots=2, num_banks=1)
    Q = generator.build_hamiltonian({"lambda_one_hot": 50.0})
    single = np.array([1, 0])   # correctly selects one slot
    double = np.array([1, 1])   # violates one-hot (selects two slots)
    # Corrected one-hot algebra: valid = -penalty, invalid double = 0.
    assert np.isclose(single @ Q @ single, -50.0)
    assert np.isclose(double @ Q @ double, 0.0)
    diff = double @ Q @ double - single @ Q @ single
    assert np.isclose(diff, 50.0)
    # The all-zeros assignment must also be penalized relative to valid ones.
    zero = np.array([0, 0])
    assert np.isclose(zero @ Q @ zero, 0.0)
    assert zero @ Q @ zero > single @ Q @ single

    # Two tasks, each with its own single slot: both scheduled is the valid
    # one-hot assignment (energy -50 per task); unassigned tasks cost more.
    generator = CCEQUBOGenerator(num_tasks=2, num_slots=1, num_banks=1)
    Q = generator.build_hamiltonian({"lambda_one_hot": 50.0})
    single = np.array([1, 0])   # task 1 left unassigned: one-hot violated
    double = np.array([1, 1])   # each task scheduled exactly once: valid
    assert np.isclose(single @ Q @ single, -50.0)
    assert np.isclose(double @ Q @ double, -100.0)
    assert double @ Q @ double < single @ Q @ single
    # Unrelated tasks do not couple: cross-group off-diagonal is zero.
    assert np.isclose(Q[0, 1], 0.0)
