from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from core_types import HardwareConfig, OperatorGraph
from energy_model import EnergyModel
from qubo_types import QUBOData


@dataclass(slots=True)
class ProblemSpec:
    """
    Quantum-backend input contract.
    """

    graph: OperatorGraph
    hardware: HardwareConfig
    alpha: Dict[str, float]
    beta: Dict[str, float]
    gamma: Dict[str, float]
    penalties: Dict[str, float]


def build_qubo(problem_spec: ProblemSpec) -> QUBOData:
    """
    Build the canonical CCE-QUBO consumed by classical/quantum solvers.
    """
    model = EnergyModel(
        graph=problem_spec.graph,
        hw=problem_spec.hardware,
        alpha=problem_spec.alpha,
        beta=problem_spec.beta,
        gamma=problem_spec.gamma,
        initial_penalties=problem_spec.penalties,
    )
    return model.build_qubo()


def qubo_energy(qubo_data: QUBOData, bits: Sequence[int]) -> float:
    """
    Evaluate E(z) = const + sum_i a_i z_i + sum_{i<=j} b_ij z_i z_j
    for a binary assignment `bits`.
    """
    if len(bits) != qubo_data.num_variables:
        raise ValueError("Bitstring length does not match QUBO variable count.")

    energy = float(qubo_data.constant)
    for idx, coeff in qubo_data.linear.items():
        energy += float(coeff) * int(bits[idx])
    for (i, j), coeff in qubo_data.quadratic.items():
        energy += float(coeff) * int(bits[i]) * int(bits[j])
    return energy


def _decode_schedule_projection(qubo_data: QUBOData, bits: Sequence[int]) -> List[int]:
    """
    Recover a coarse schedule order from active x_{i,t,r} variables:
    sort by time, then op id; keep first occurrence per op.
    """
    active_x: List[tuple[int, int, str]] = []
    for idx, meta in qubo_data.var_metadata.items():
        if int(bits[idx]) != 1:
            continue
        if meta.get("kind") != "x":
            continue
        op = int(meta["op"])
        t = int(meta["t"])
        r = str(meta["r"])
        active_x.append((t, op, r))

    active_x.sort(key=lambda item: (item[0], item[1], item[2]))
    seen = set()
    order: List[int] = []
    for _, op, _ in active_x:
        if op in seen:
            continue
        seen.add(op)
        order.append(op)
    return order


def run_qaoa_statevector(
    qubo_data: QUBOData,
    num_samples: int = 48,
    p_layers: int = 2,
    max_iter: int = 25,
    seed: int = 101,
    max_qubits: int = 12,
    p_sweep: bool = True,
    p_max: int = 3,
    cobyla_budget: int = 60,
) -> List[Dict[str, Any]]:
    """
    Exact statevector QAOA candidate generation via QAOA_solver.py
    (feasible only for small problem sizes: 2^num_qubits statevector).

    Converts the QUBO to an Ising Hamiltonian and optimizes the variational
    parameters. With p_sweep enabled (default) the optimization runs for every
    depth p = 1..p_max with a COBYLA budget per depth (cobyla_budget
    evaluations) and the depth with the best ground-state approximation ratio
    is used; with p_sweep disabled a single p_layers-depth gradient descent
    run is used. Sampling then takes the highest-probability computational
    basis states as candidate schedules.
    Returns candidates in the same dict shape as run_qaoa_stub with backend
    "qaoa_statevector". Returns [] if the problem is too large.
    """
    n = qubo_data.num_variables
    if n <= 0 or n > max_qubits:
        return []

    import numpy as np

    from ising_hamiltonian_converter import convert_qubo_to_ising
    from QAOA_solver import QAOASolver, sweep_qaoa_depth

    ising = convert_qubo_to_ising(qubo_data)

    p_sweep_summary: List[Dict[str, Any]] = []
    if p_sweep:
        # Seeded depth sweep p = 1..p_max, COBYLA budget per depth; the depth
        # with the best approximation ratio supplies the sampled ansatz.
        sweep = sweep_qaoa_depth(
            ising,
            p_values=range(1, max(1, int(p_max)) + 1),
            budget_per_p=int(cobyla_budget),
            seed=int(seed),
            max_qubits=int(max_qubits),
        )
        best_solver = sweep["best"]["solver"]
        opt_gamma = np.asarray(sweep["best"]["gamma"], dtype=float)
        opt_beta = np.asarray(sweep["best"]["beta"], dtype=float)
        p_sweep_summary = [
            {k: entry[k] for k in ("p", "energy", "approx_ratio")} for entry in sweep["results"]
        ]
    else:
        best_solver = QAOASolver(ising, p=int(p_layers), max_qubits=int(max_qubits))
        # QAOA_solver draws initial parameters from its own seeded generator;
        # snapshot and restore the global NumPy state so callers that share the
        # global RNG are unaffected either way.
        rng_state = np.random.get_state()
        try:
            opt_gamma, opt_beta, _best_energy = best_solver.optimize_parameters(
                max_iter=int(max_iter), seed=int(seed)
            )
        finally:
            np.random.set_state(rng_state)

    psi = best_solver.evaluate_statevector(opt_gamma, opt_beta)
    probs = np.abs(psi) ** 2
    top_states = np.argsort(probs)[::-1][: max(1, num_samples)]

    # var_map: original variable id -> qubit index. QAOASolver's spin mapping
    # is s = 1 - 2*bit with x = (1 - s)/2, hence x_i == bit_i.
    results: List[Dict[str, Any]] = []
    for rank, state in enumerate(top_states, start=1):
        bits = [0] * n
        for var_id, q_idx in best_solver.var_map.items():
            idx = int(var_id)
            if 0 <= idx < n:
                bits[idx] = int((int(state) >> int(q_idx)) & 1)
        active = [idx for idx, b in enumerate(bits) if b == 1]
        results.append(
            {
                "rank": rank,
                "energy": qubo_energy(qubo_data, bits),
                "bitstring": "".join("1" if b else "0" for b in bits),
                "active_variables": active,
                "schedule_projection": _decode_schedule_projection(qubo_data, bits),
                "backend": "qaoa_statevector",
                "qaoa_params": {
                    "gamma": [float(x) for x in np.atleast_1d(opt_gamma)],
                    "beta": [float(x) for x in np.atleast_1d(opt_beta)],
                    "layers": int(best_solver.p),
                    "qubits": int(best_solver.num_qubits),
                    "optimizer": "cobyla_p_sweep" if p_sweep else "gradient_descent",
                    "p_sweep": p_sweep_summary,
                },
            }
        )
    return results


def run_qaoa_stub(
    qubo_data: QUBOData,
    num_samples: int = 64,
    num_steps: int = 220,
    seed: int = 101,
) -> List[Dict[str, Any]]:
    """
    Deterministic classical multi-start bit-flip local search on the QUBO
    energy, exposed through a quantum-backend-compatible candidate interface.

    Used as the fallback for large QUBOs where exact statevector QAOA is
    infeasible (2^n states). Labeled backend "local_search_fallback" so
    outputs never misrepresent this as a quantum result.
    """
    rng = random.Random(seed)
    n = qubo_data.num_variables
    if n <= 0:
        return []

    starts = max(4, min(num_samples, 16))
    pool: List[List[int]] = [[rng.randint(0, 1) for _ in range(n)] for _ in range(starts)]
    pool_energy = [qubo_energy(qubo_data, bits) for bits in pool]

    best_seen: Dict[str, float] = {}
    for walk_idx in range(starts):
        bits = list(pool[walk_idx])
        energy = pool_energy[walk_idx]

        for step in range(max(1, num_steps)):
            temp = max(0.01, 1.5 * (1.0 - step / max(1, num_steps)))
            k = rng.randint(1, 3)
            flip_indices = rng.sample(range(n), k=k)

            proposal = list(bits)
            for fidx in flip_indices:
                proposal[fidx] = 1 - proposal[fidx]

            p_energy = qubo_energy(qubo_data, proposal)
            delta = p_energy - energy
            accept = delta <= 0.0 or rng.random() < pow(2.718281828, -delta / temp)
            if accept:
                bits = proposal
                energy = p_energy

        bitstring = "".join("1" if b else "0" for b in bits)
        prev = best_seen.get(bitstring)
        if prev is None or energy < prev:
            best_seen[bitstring] = energy

    ranked = sorted(best_seen.items(), key=lambda item: item[1])[: max(1, num_samples)]
    results: List[Dict[str, Any]] = []
    for rank, (bitstring, energy) in enumerate(ranked, start=1):
        bits = [1 if ch == "1" else 0 for ch in bitstring]
        active = [idx for idx, b in enumerate(bits) if b == 1]
        results.append(
            {
                "rank": rank,
                "energy": float(energy),
                "bitstring": bitstring,
                "active_variables": active,
                "schedule_projection": _decode_schedule_projection(qubo_data, bits),
                "backend": "local_search_fallback",
            }
        )
    return results


class QuantumInterface:
    """
    Thin OO wrapper around the required functional API.
    """

    def __init__(self, seed: int = 101) -> None:
        self.seed = int(seed)

    def build_qubo(self, problem_spec: ProblemSpec) -> QUBOData:
        return build_qubo(problem_spec)

    def run_qaoa_statevector(
        self,
        qubo_data: QUBOData,
        num_samples: int = 48,
        p_layers: int = 2,
        max_iter: int = 25,
        p_sweep: bool = True,
        p_max: int = 3,
        cobyla_budget: int = 60,
    ) -> List[Dict[str, Any]]:
        return run_qaoa_statevector(
            qubo_data=qubo_data,
            num_samples=num_samples,
            p_layers=p_layers,
            max_iter=max_iter,
            seed=self.seed,
            p_sweep=p_sweep,
            p_max=p_max,
            cobyla_budget=cobyla_budget,
        )

    def run_qaoa_stub(
        self,
        qubo_data: QUBOData,
        num_samples: int = 64,
        num_steps: int = 220,
    ) -> List[Dict[str, Any]]:
        return run_qaoa_stub(
            qubo_data=qubo_data,
            num_samples=num_samples,
            num_steps=num_steps,
            seed=self.seed,
        )
