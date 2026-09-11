"""Bayesian hyper-parameter optimiser for CCE-QOS scheduling workloads.

This module wraps scikit-optimize to provide a small, dependency-light search
driver.  Its primary use case in this repository is selecting the SRAM bank
count for the exact CP-SAT scheduler: more banks allow more parallelism but
cost area and routing resources, so the interesting engineering question is
the smallest bank count that already reaches the minimum achievable makespan.

The optimiser is deterministic given ``random_state`` and always minimises the
objective.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Tuple

import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real

__all__ = ["SearchDimension", "BayesianOptimizer", "optimize"]


def SearchDimension(low: int, high: int) -> Tuple[int, int]:
    """Return an integer search dimension as a ``(low, high)`` tuple.

    Kept as a tiny helper so call sites read naturally::

        optimize(objective, [SearchDimension(1, 8)])
    """
    return (int(low), int(high))


def _to_skopt_dimension(spec: Sequence) -> List:
    """Convert ``(low, high)`` tuples or value lists into skopt dimensions."""
    if isinstance(spec, tuple):
        low, high = spec
        if isinstance(low, int) and isinstance(high, int):
            return Integer(low, high)
        return Real(float(low), float(high))
    if isinstance(spec, list):
        return Categorical(spec)
    raise TypeError(f"Unsupported search dimension: {spec!r}")


class BayesianOptimizer:
    """Deterministic Bayesian optimiser over mixed integer/real/categorical spaces.

    Parameters
    ----------
    dimensions:
        One spec per free parameter: an ``(low, high)`` tuple for numeric
        ranges, or a list of values for a categorical parameter.
    n_initial_points:
        Number of uniformly random evaluations before the surrogate model
        takes over (skopt default: 10).
    random_state:
        Seed controlling both initial sampling and the surrogate; identical
        seeds reproduce identical search traces.
    """

    def __init__(
        self,
        dimensions: List[Sequence],
        n_initial_points: int = 6,
        random_state: int = 42,
    ) -> None:
        self.dimensions = [_to_skopt_dimension(spec) for spec in dimensions]
        self._optimizer = Optimizer(
            self.dimensions,
            base_estimator="GP",
            n_initial_points=n_initial_points,
            random_state=random_state,
        )
        self.evaluated_points: List[List] = []
        self.evaluated_values: List[float] = []

    def ask(self) -> List:
        return self._optimizer.ask()

    def tell(self, point: List, value: float) -> None:
        self._optimizer.tell(point, float(value))
        self.evaluated_points.append(list(point))
        self.evaluated_values.append(float(value))

    def best(self) -> Tuple[List, float]:
        """Return the best ``(point, value)`` seen so far (lowest value wins)."""
        if not self.evaluated_values:
            raise RuntimeError("No evaluations recorded yet.")
        idx = int(np.argmin(self.evaluated_values))
        return self.evaluated_points[idx], self.evaluated_values[idx]


def optimize(
    func,
    dimensions: List[Sequence],
    n_calls: int = 15,
    random_state: int = 42,
):
    """Minimise ``func`` over ``dimensions`` with a fixed evaluation budget.

    Returns ``(best_point, best_value)``.  ``func`` receives one argument per
    dimension and must return a scalar.
    """
    if n_calls < 1:
        raise ValueError("n_calls must be positive")
    opt = BayesianOptimizer(dimensions, random_state=random_state)
    for _ in range(n_calls):
        point = opt.ask()
        value = func(*point)
        opt.tell(point, value)
    return opt.best()
