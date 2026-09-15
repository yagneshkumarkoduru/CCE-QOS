"""Braket QAOA circuit parity tests (local simulation only, no AWS calls).

The Braket circuit must reproduce the solver's exact statevector
expectation; the measurement bitstring convention (qubit 0 leftmost,
verified against local and managed SV1) must match the solver's
basis-state indexing; and the legacy OpenQASM export must include the
linear cost terms.
"""

import numpy as np
import pytest

pytest.importorskip("braket")

from braket.circuits import Circuit  # noqa: E402
from braket.devices import LocalSimulator  # noqa: E402

from QAOA_solver import DEFAULT_QAOA_SEED, QAOASolver  # noqa: E402
from braket_experiments.circuits import (  # noqa: E402
    bitstring_energy,
    build_qaoa_circuit,
    energy_stats,
    openqasm3,
    solver_index_from_bitstring,
    solver_ordered_probabilities,
)
from braket_experiments.instances import (  # noqa: E402
    build_random_ising,
    reference_chain_model,
)


def _fitted_solver(model, p, max_iter=40):
    solver = QAOASolver(model, p=p)
    gamma, beta, _energy = solver.optimize_parameters_cobyla(
        max_iter=max_iter, seed=DEFAULT_QAOA_SEED
    )
    return solver, gamma, beta


def _braket_probabilities_in_solver_order(solver, gamma, beta):
    circuit = build_qaoa_circuit(solver, gamma, beta).state_vector()
    result = LocalSimulator("braket_sv").run(circuit).result()
    amplitudes = np.asarray(result.values).reshape(-1)
    return solver_ordered_probabilities(amplitudes, solver.num_qubits)


def test_measurement_bit_order_convention_is_qubit_zero_leftmost():
    """Lock the empirically verified convention (local and managed SV1)."""

    counts = (
        LocalSimulator()
        .run(Circuit().x(0).i(1), shots=20)
        .result()
        .measurement_counts
    )
    assert dict(counts) == {"10": 20}
    assert solver_index_from_bitstring("10") == 1
    assert solver_index_from_bitstring("01") == 2


@pytest.mark.parametrize("p", [1, 2])
def test_circuit_statevector_matches_solver_expectation(p):
    model = reference_chain_model()
    solver, gamma, beta = _fitted_solver(model, p)
    probabilities = _braket_probabilities_in_solver_order(solver, gamma, beta)
    expectation = float(np.sum(probabilities * solver.basis_energies))
    assert expectation == pytest.approx(
        solver.compute_expectation(gamma, beta), abs=1e-8
    )


def test_circuit_statevector_matches_on_larger_instance():
    model = build_random_ising()
    solver, gamma, beta = _fitted_solver(model, p=1, max_iter=30)
    probabilities = _braket_probabilities_in_solver_order(solver, gamma, beta)
    expectation = float(np.sum(probabilities * solver.basis_energies))
    assert expectation == pytest.approx(
        solver.compute_expectation(gamma, beta), abs=1e-8
    )


def test_bitstring_energy_matches_basis_energies():
    solver = QAOASolver(reference_chain_model(), p=1)
    for state in range(solver.dim):
        bits = format(state, f"0{solver.num_qubits}b")
        braket_bits = bits[::-1]
        assert bitstring_energy(solver, braket_bits) == pytest.approx(
            solver.basis_energies[state]
        )


def test_measurement_counts_reconstruct_expectation():
    model = reference_chain_model()
    solver, gamma, beta = _fitted_solver(model, p=2)
    circuit = build_qaoa_circuit(solver, gamma, beta)
    counts = LocalSimulator().run(circuit, shots=20000).result().measurement_counts
    stats = energy_stats(solver, counts)
    assert stats["mean_energy"] == pytest.approx(
        solver.compute_expectation(gamma, beta), abs=0.08
    )
    assert stats["best_energy"] >= solver.e_min - 1e-9


def test_openqasm3_export_is_braket_openqasm3():
    solver = QAOASolver(reference_chain_model(), p=1)
    source = openqasm3(solver, [0.3], [0.4])
    assert source.startswith("OPENQASM 3.0")


def test_to_openqasm_includes_linear_cost_terms():
    solver = QAOASolver(reference_chain_model(), p=1)
    qasm = solver.to_openqasm(np.array([0.3]), np.array([0.4]))
    # 3 linear rz terms (one per variable with nonzero h) plus 2 edge rz terms
    assert qasm.count("rz(") == 5
