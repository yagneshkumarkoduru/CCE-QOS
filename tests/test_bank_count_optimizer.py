"""Tests for the Bayesian bank-count optimiser."""

from __future__ import annotations

import pytest

pytest.importorskip("skopt", reason="scikit-optimize is required for the autotuning tests")

from implementations.v1_exact_cpsat_solver.ortools_cpsat_engine import ExactCPSATScheduler
from autotuning.bank_count_optimizer import BayesianOptimizer, optimize


def _makespan_for_banks(num_banks: float) -> float:
    """Makespan of an eight-task independent workload for a given bank count.

    Every task needs an entire bank for its full duration, so with ``b`` banks
    at most ``b`` tasks overlap and the theoretical minimum makespan is the
    sum of durations divided by the bank count (ceiling).
    """
    tasks = [
        {"id": f"t{i}", "duration": 3, "memory_kb": 8}
        for i in range(8)
    ]
    scheduler = ExactCPSATScheduler(sram_capacity_kb=8, num_banks=int(num_banks))
    result = scheduler.solve(tasks, [])
    return float(result["makespan"])


def test_optimizer_minimises_simple_quadratic() -> None:
    def objective(x: float) -> float:
        return (x - 3) ** 2

    best_point, best_value = optimize(objective, [(0.0, 6.0)], n_calls=20, random_state=7)
    assert best_value < 0.5
    assert 2.0 <= best_point[0] <= 4.0


def test_bank_count_search_reaches_best_makespan() -> None:
    best_point, best_value = optimize(
        _makespan_for_banks,
        [(1, 8)],
        n_calls=8,
        random_state=42,
    )
    # Eight independent 3-cycle tasks never finish sooner than 3 cycles on
    # eight banks, and single-bank execution needs 24 cycles.
    assert best_value == pytest.approx(3.0)
    assert int(best_point[0]) == 8


def test_optimizer_records_every_evaluation() -> None:
    opt = BayesianOptimizer([(0, 4)], random_state=0)
    for _ in range(5):
        point = opt.ask()
        value = float(point[0]) ** 2
        opt.tell(point, value)
    assert len(opt.evaluated_points) == 5
    assert len(opt.evaluated_values) == 5
    assert min(opt.evaluated_values) == min(float(p[0]) ** 2 for p in opt.evaluated_points)


def test_optimizer_requires_evaluations_before_best() -> None:
    opt = BayesianOptimizer([(1, 8)], random_state=1)
    with pytest.raises(RuntimeError):
        opt.best()


def test_invalid_call_budget_rejected() -> None:
    with pytest.raises(ValueError):
        optimize(lambda x: x, [(1, 8)], n_calls=0)
