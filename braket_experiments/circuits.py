"""QAOA circuit construction and energy evaluation for Braket execution.

The circuit mirrors the solver's unitary exactly:

    |psi> = prod_l e^{-i beta_l H_M} e^{-i gamma_l H_C} |+>^n
    H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
    H_M = sum_i X_i

The cost layer therefore contains single-qubit Z rotations for the linear
h terms (which the legacy OpenQASM 2 export omitted), CX-RZ-CX blocks for
the quadratic J terms, and single-qubit X rotations for the mixer.

Measurement convention: Braket returns the highest-index qubit as the
leftmost character, so the solver's basis-state integer is exactly
``int(bitstring, 2)``.
"""

from __future__ import annotations

from braket.circuits import Circuit

from QAOA_solver import QAOASolver


def build_qaoa_circuit(solver: QAOASolver, gamma, beta) -> Circuit:
    """Build the exact QAOA circuit for the solver's mapped qubits."""

    gamma = [float(value) for value in gamma]
    beta = [float(value) for value in beta]
    if len(gamma) != solver.p or len(beta) != solver.p:
        raise ValueError("gamma and beta must each have length p")

    circuit = Circuit()
    for qubit in range(solver.num_qubits):
        circuit.h(qubit)

    for layer in range(solver.p):
        for variable, qubit in solver.var_map.items():
            coefficient = float(solver.h.get(variable, 0.0))
            if abs(coefficient) > 0.0:
                circuit.rz(qubit, 2.0 * gamma[layer] * coefficient)
        for (u, v), coefficient in solver.J.items():
            if u in solver.var_map and v in solver.var_map:
                qubit_u = solver.var_map[u]
                qubit_v = solver.var_map[v]
                angle = 2.0 * float(coefficient) * gamma[layer]
                circuit.cnot(qubit_u, qubit_v)
                circuit.rz(qubit_v, angle)
                circuit.cnot(qubit_u, qubit_v)
        for qubit in range(solver.num_qubits):
            circuit.rx(qubit, 2.0 * beta[layer])

    return circuit


def solver_index_from_bitstring(bitstring: str) -> int:
    """Map a Braket measurement string to the solver's basis-state integer.

    Convention verified empirically on this SDK (local ``braket_sv`` and
    managed SV1, task arn .../81df2400 recorded in the ledger): qubit 0 is
    the leftmost character, so X on qubit 0 of a two-qubit circuit returns
    ``'10'``. The solver indexes basis states with qubit q at bit q, so the
    string is reversed before parsing.
    """

    return int(bitstring[::-1], 2)


def bitstring_energy(solver: QAOASolver, bitstring: str) -> float:
    """Energy of one measured bitstring under the solver's basis energies."""

    index = solver_index_from_bitstring(bitstring)
    if not 0 <= index < solver.dim:
        raise ValueError(f"bitstring {bitstring!r} outside the basis space")
    return float(solver.basis_energies[index])


def energy_stats(solver: QAOASolver, counts) -> dict:
    """Energy statistics of a measured count distribution."""

    total = sum(int(count) for count in counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    mean_energy = 0.0
    best_energy = float("inf")
    ground_shots = 0
    for bitstring, count in counts.items():
        energy = bitstring_energy(solver, bitstring)
        mean_energy += energy * int(count)
        best_energy = min(best_energy, energy)
        if abs(energy - solver.e_min) < 1e-9:
            ground_shots += int(count)
    mean_energy /= total
    return {
        "shots": total,
        "mean_energy": mean_energy,
        "best_energy": best_energy,
        "ground_state_probability": ground_shots / total,
        "approx_ratio_mean": solver.approximation_ratio(mean_energy),
        "approx_ratio_best": solver.approximation_ratio(best_energy),
        "unique_bitstrings": len(counts),
    }


def openqasm3(solver: QAOASolver, gamma, beta) -> str:
    """OpenQASM 3 source for the circuit (Braket serialization)."""

    return build_qaoa_circuit(solver, gamma, beta).to_ir().source


def solver_ordered_probabilities(braket_amplitudes, num_qubits: int):
    """Reorder Braket statevector amplitudes into the solver's basis order.

    Braket indexes the statevector with qubit 0 as the most significant
    bit; the solver uses qubit q at bit q. The permutation is a bit
    reversal within ``num_qubits`` bits.
    """

    import numpy as np

    amplitudes = np.asarray(braket_amplitudes, dtype=np.complex128).reshape(-1)
    dim = 1 << num_qubits
    if amplitudes.shape != (dim,):
        raise ValueError("amplitude vector does not match the qubit count")
    ordered = np.zeros(dim, dtype=np.complex128)
    for index, amplitude in enumerate(amplitudes):
        reversed_bits = int(format(index, f"0{num_qubits}b")[::-1], 2)
        ordered[reversed_bits] = amplitude
    return np.abs(ordered) ** 2
