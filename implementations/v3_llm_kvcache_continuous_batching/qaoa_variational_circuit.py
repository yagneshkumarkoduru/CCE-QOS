"""
Variational QAOA-style Energy Evaluation for CCE-QUBO (coherent mean-field ansatz).

Builds a real, fixed QUBO/Ising instance (seeded random symmetric couplings),
computes its exact ground and maximum state energies by exhaustive enumeration
over all 2^n computational basis states (n <= 12), and evaluates the energy of
a coherent product-state (mean-field) variational ansatz parameterized by a
2-parameter (gamma, beta) control:

    p_i(gamma, beta) = sigmoid( gamma * f_i / beta )   (excitation probability)

where f_i = -(Q[i][i] + sum_{j != i} Q[i][j]) is the local effective field on
qubit i at the uniform state. The ansatz energy is evaluated EXACTLY under the
product state:

    E(gamma, beta) = sum_i Q[i][i] p_i + sum_{i<j} 2 Q[i][j] p_i p_j

gamma -> 0 or beta -> inf recovers the uniform (maximally mixed) state;
gamma large / beta small approaches a greedy mean-field solution.
"""

import numpy as np


class CCEVariationalQAOA:
    def __init__(self, num_qubits: int = 8, p_layers: int = 2, seed: int = 11):
        if num_qubits < 1 or num_qubits > 12:
            raise ValueError("num_qubits must be in [1, 12] for exhaustive enumeration.")
        self.num_qubits = num_qubits
        self.p_layers = p_layers
        self.dim = 2 ** num_qubits

        # Fixed real QUBO instance: seeded symmetric couplings (linear + quadratic).
        rng = np.random.default_rng(seed)
        Q_rand = rng.normal(0.0, 1.0, size=(num_qubits, num_qubits))
        self.Q = 0.5 * (Q_rand + Q_rand.T)  # symmetric quadratic couplings
        self.linear = rng.uniform(-1.0, 1.0, size=num_qubits)  # linear fields
        for i in range(num_qubits):
            self.Q[i, i] = self.linear[i]

        # Exhaustive enumeration of all 2^n basis-state energies (exact).
        states = np.arange(self.dim)
        bits = ((states[:, None] >> np.arange(num_qubits)) & 1).astype(np.float64)
        self.basis_energies = np.einsum("bi,ij,bj->b", bits, self.Q, bits)
        self.e_ground = float(self.basis_energies.min())
        self.e_max = float(self.basis_energies.max())
        self.ground_state = int(np.argmin(self.basis_energies))

    def _product_state_probs(self, gamma: float, beta: float) -> np.ndarray:
        """Excitation probabilities of the coherent mean-field ansatz."""
        f = -(self.Q.diagonal() + self.Q.sum(axis=1) - self.Q.diagonal())
        beta = max(abs(beta), 1e-9)
        return 1.0 / (1.0 + np.exp(-(gamma * f) / beta))

    def compute_energy_expectation(self, gamma: float, beta: float) -> dict:
        """Exact energy of the (gamma, beta) product-state ansatz.

        Returns the variational expectation value, the exact ground-state
        energy, and the approximation ratio
            r = (e_max - E) / (e_max - e_ground) in [0, 1]  (1.0 == optimal).
        """
        p = self._product_state_probs(gamma, beta)
        # Product-state energy: E = sum_i Q_ii p_i + sum_{i<j} 2 Q_ij p_i p_j.
        # p^T Q p = sum_i Q_ii p_i^2 + 2 sum_{i<j} Q_ij p_i p_j, and p_i^2 = p_i
        # only for binary draws, so correct with the diagonal correction below.
        e = float(p @ self.Q @ p
                  - float(self.Q.diagonal() @ (p * p))
                  + float(self.Q.diagonal() @ p))
        denom = (self.e_max - self.e_ground) if self.e_max != self.e_ground else 1.0
        approx_ratio = float((self.e_max - e) / denom)
        return {
            "gamma": float(gamma),
            "beta": float(beta),
            "expectation_value": e,
            "e_ground": self.e_ground,
            "e_max": self.e_max,
            "approximation_ratio": approx_ratio,
        }

    def optimize(self, gamma_grid=None, beta_grid=None) -> dict:
        """Grid-scan the (gamma, beta) energy surface and return the best point."""
        if gamma_grid is None:
            gamma_grid = np.linspace(0.05, 6.0, 40)
        if beta_grid is None:
            beta_grid = np.linspace(0.05, 6.0, 40)
        best = None
        for g in gamma_grid:
            for b in beta_grid:
                res = self.compute_energy_expectation(g, b)
                if best is None or res["expectation_value"] < best["expectation_value"]:
                    best = res
        return best


if __name__ == "__main__":
    qaoa = CCEVariationalQAOA()
    print(f"[QUBO instance] n={qaoa.num_qubits} qubits, "
          f"e_ground={qaoa.e_ground:.4f} (state {qaoa.ground_state:0{qaoa.num_qubits}b}), "
          f"e_max={qaoa.e_max:.4f}")
    res = qaoa.compute_energy_expectation(0.45, 0.78)
    print(f"[OK] QAOA mean-field energy at (gamma=0.45, beta=0.78): {res}")
    best = qaoa.optimize()
    print(f"[OK] Optimized variational point: gamma={best['gamma']:.3f}, "
          f"beta={best['beta']:.3f}, E={best['expectation_value']:.4f}, "
          f"approximation_ratio={best['approximation_ratio']:.4f}")
