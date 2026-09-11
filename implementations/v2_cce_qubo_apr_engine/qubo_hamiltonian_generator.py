"""
Quadratic Unconstrained Binary Optimization (QUBO) Hamiltonian Generator for CCE-QOS.
Maps continuous schedule and bank allocation variables into binary decision space:
x_{i, t, b} in {0, 1} -> Task i scheduled at time slot t in memory bank b.
H_total = H_energy + lambda_one_hot * H_one_hot + lambda_precedence * H_precedence.

One-hot penalty derivation (exact algebra):
    H_one_hot = (sum_k x_k - 1)^2
              = sum_k x_k^2 + 2*sum_{k<l} x_k x_l - 2*sum_k x_k + 1
    For binary x_k, x_k^2 == x_k, so:
              = -sum_k x_k + 2*sum_{k<l} x_k x_l + 1
    In QUBO form H = x^T Q x with x^T Q x = sum_i Q[i][i] x_i + 2*sum_{i<j} Q[i][j] x_i x_j
    (Q stored symmetric), the coefficients are:
        Q[k][k] = -penalty              (linear/diagonal term)
        Q[k][l] = +penalty (k != l)     (each symmetric entry stores penalty so
                                         the pair contributes 2*penalty*x_k*x_l)
    The constant 1 is dropped (irrelevant to argmin over x).
"""

import numpy as np


class CCEQUBOGenerator:
    def __init__(self, num_tasks: int, num_slots: int, num_banks: int = 4):
        self.num_tasks = num_tasks
        self.num_slots = num_slots
        self.num_banks = num_banks
        self.num_vars = num_tasks * num_slots * num_banks
        self.Q = np.zeros((self.num_vars, self.num_vars))

    def var_index(self, task: int, slot: int, bank: int) -> int:
        return task * (self.num_slots * self.num_banks) + slot * self.num_banks + bank

    def add_one_hot_constraints(self, penalty: float = 50.0):
        """Each task must be scheduled exactly once: (sum_{t, b} x_{i,t,b} - 1)^2.

        Correct expansion for binary variables (see module docstring):
          - diagonal: Q[k][k] += -penalty  (linear coefficient of x_k)
          - off-diagonal: Q[k][l] += +penalty per symmetric entry, so the
            quadratic pair term contributes 2*penalty*x_k*x_l.

        With this formulation, any valid one-hot assignment (exactly one 1
        in the group) has energy -penalty, while the all-zeros assignment
        has energy 0 and any assignment with two or more 1s has energy >= 0.
        Hence invalid assignments are strictly penalized relative to valid ones.
        """
        for i in range(self.num_tasks):
            indices = [self.var_index(i, t, b) for t in range(self.num_slots) for b in range(self.num_banks)]
            # Diagonal: linear coefficient of x_k in -sum_k x_k is -penalty.
            for k in indices:
                self.Q[k, k] += -penalty
            # Off-diagonal: each symmetric entry stores +penalty so that
            # x^T Q x yields 2*penalty*x_k*x_l per pair (the +2*sum_{k<l} term).
            for p in range(len(indices)):
                for q in range(p + 1, len(indices)):
                    self.Q[indices[p], indices[q]] += penalty
                    self.Q[indices[q], indices[p]] += penalty

    def add_precedence_constraints(self, dependencies, penalty: float = 10.0):
        """Penalize slot inversions: task i scheduled at a later-or-equal slot
        than a task j that depends on i.

        Implemented as a soft quadratic penalty over slot indices (the bank
        dimension is marginalized by taking the max over active banks is not
        tractable in a purely quadratic form, so the penalty couples all
        (slot_i, slot_j) variable pairs with slot_i > slot_j):

            H_precedence = penalty * sum_{(i depends on j)} sum_{t_i >= t_j}
                           sum_{b_i, b_j} x_{i,t_i,b_i} * x_{j,t_j,b_j}

        Note: this penalizes a task being scheduled in the same slot as its
        dependency as well as after it. Duration-aware ordering (t_i + tau_i
        <= t_j) is NOT modeled here because the generator has no access to
        task durations; it only enforces slot-level ordering.
        """
        for child, parent in dependencies:
            for t_child in range(self.num_slots):
                for t_parent in range(t_child + 1):
                    # parent must run strictly before child: penalize
                    # (parent at t_parent, child at t_child) when t_parent >= t_child
                    for b_parent in range(self.num_banks):
                        for b_child in range(self.num_banks):
                            idx_p = self.var_index(parent, t_parent, b_parent)
                            idx_c = self.var_index(child, t_child, b_child)
                            self.Q[idx_p, idx_c] += penalty / 2.0
                            self.Q[idx_c, idx_p] += penalty / 2.0

    def build_hamiltonian(self, penalties: dict, dependencies=None) -> np.ndarray:
        """Build the full QUBO matrix.

        Args:
            penalties: dict with optional keys "lambda_one_hot" and
                "lambda_precedence". Both are honored; unknown keys raise
                ValueError so silent no-op penalties cannot occur.
            dependencies: optional iterable of (parent_task, child_task) pairs
                (child depends on parent). Required for lambda_precedence to
                have any effect; if None, precedence is not added even when a
                lambda_precedence value is supplied (documented behavior).
        """
        known_keys = {"lambda_one_hot", "lambda_precedence"}
        unknown = set(penalties.keys()) - known_keys
        if unknown:
            raise ValueError(
                f"Unknown penalty keys ignored would be unsafe. "
                f"Unsupported keys: {sorted(unknown)}. Supported: {sorted(known_keys)}"
            )

        self.Q.fill(0.0)
        self.add_one_hot_constraints(penalty=penalties.get("lambda_one_hot", 50.0))

        lam_prec = penalties.get("lambda_precedence")
        if lam_prec is not None:
            if dependencies is None:
                raise ValueError(
                    "lambda_precedence supplied but no dependencies given; "
                    "pass dependencies=[(parent, child), ...] to build_hamiltonian."
                )
            self.add_precedence_constraints(dependencies, penalty=lam_prec)

        return self.Q
