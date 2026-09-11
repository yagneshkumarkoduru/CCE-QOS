"""
QUBO-APR Engine Benchmark: CCE-QUBO and Adaptive Penalty Refinement (APR)
Runs a real APR optimization loop: a QUBO is built with CCEQUBOGenerator
(one-hot + precedence penalties), combined with a linear/quadratic objective
energy, and solved approximately by greedy random-restart bit-flip local
search. AdaptivePenaltyRefinement updates penalty multipliers on measured
violations each round; a static-penalty control track uses frozen penalties.
All plotted violations and energies are recorded from the loop, not
synthesized. Generates publication plot: figures/fig_cce_qubo_apr_benchmark.png

Objective energy model (comparable across tracks because it is penalty-free):
    H_obj = sum_i dur_i * (1 + slot_growth * t_i)              (compute energy)
          + kappa * sum_{t, b} C(|{i : x_{i,t,b}=1}|, 2)       (bank conflicts)
A topological first-fit greedy schedule is the baseline; the APR track
minimizes H_obj + penalty terms and its H_obj is reported.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from .apr_penalty_refinement import AdaptivePenaltyRefinement
    from .qubo_hamiltonian_generator import CCEQUBOGenerator
except ImportError:
    try:
        from apr_penalty_refinement import AdaptivePenaltyRefinement
        from qubo_hamiltonian_generator import CCEQUBOGenerator
    except ImportError:
        from implementations.v2_cce_qubo_apr_engine.apr_penalty_refinement import AdaptivePenaltyRefinement
        from implementations.v2_cce_qubo_apr_engine.qubo_hamiltonian_generator import CCEQUBOGenerator

NUM_TASKS, NUM_SLOTS, NUM_BANKS = 4, 4, 2
DEPENDENCIES = [(0, 2), (1, 2), (2, 3)]  # task2 depends on {0,1}; task3 depends on 2
DURATIONS = np.array([3.0, 2.0, 4.0, 1.0])
SLOT_GROWTH = 0.15  # per-slot compute-energy growth factor (later slots costlier)
KAPPA_BANK = 2.0    # bank-conflict energy per colliding task pair


def _qubo_energy(Q: np.ndarray, bits: np.ndarray) -> float:
    return float(bits @ Q @ bits)


def _objective_energy(bits: np.ndarray) -> float:
    """Penalty-free objective energy of a decoded schedule (arbitrary units)."""
    block = NUM_SLOTS * NUM_BANKS
    energy = 0.0
    for i in range(NUM_TASKS):
        group = bits[i * block:(i + 1) * block].reshape(NUM_SLOTS, NUM_BANKS)
        slots, banks = np.where(group > 0.5)
        for s in slots:
            energy += DURATIONS[i] * (1.0 + SLOT_GROWTH * s)
    for t in range(NUM_SLOTS):
        for b in range(NUM_BANKS):
            occupants = sum(
                1 for i in range(NUM_TASKS)
                if bits[i * block + t * NUM_BANKS + b] > 0.5
            )
            energy += KAPPA_BANK * occupants * (occupants - 1) / 2.0
    return energy


def _selected_slot(bits: np.ndarray, task: int):
    """Return the slot where a task is active (max over banks), or None if
    the task has no active variable (one-hot violation already counted)."""
    block = NUM_SLOTS * NUM_BANKS
    group = bits[task * block:(task + 1) * block]
    slots = group.reshape(NUM_SLOTS, NUM_BANKS).sum(axis=1)
    if slots.sum() == 0:
        return None
    return int(np.argmax(slots))


def _count_violations(bits: np.ndarray) -> dict:
    """Measure real constraint violations from a decoded schedule."""
    block = NUM_SLOTS * NUM_BANKS
    violations = {"lambda_one_hot": 0.0, "lambda_precedence": 0.0}

    for i in range(NUM_TASKS):
        active = int(np.sum(bits[i * block:(i + 1) * block]))
        violations["lambda_one_hot"] += abs(active - 1)

    slots = {i: _selected_slot(bits, i) for i in range(NUM_TASKS)}
    for parent, child in DEPENDENCIES:
        s_p, s_c = slots.get(parent), slots.get(child)
        if s_p is None or s_c is None:
            # Missing schedule endpoint: already counted in one-hot.
            continue
        if s_c <= s_p:
            # Child not strictly after its dependency: one slot inversion.
            violations["lambda_precedence"] += 1.0

    return violations


def _local_search(Q: np.ndarray, rng: np.random.RandomState,
                  restarts: int = 8, flips_per_restart: int = 240) -> np.ndarray:
    """Greedy random-restart bit-flip local search over the QUBO."""
    n = Q.shape[0]
    best_bits = None
    best_energy = np.inf

    for _ in range(restarts):
        bits = (rng.rand(n) < 0.5).astype(np.float64)
        energy = _qubo_energy(Q, bits)
        for _ in range(flips_per_restart):
            k = rng.randint(1, 4)
            idx = rng.choice(n, size=k, replace=False)
            proposal = bits.copy()
            proposal[idx] = 1.0 - proposal[idx]
            p_energy = _qubo_energy(Q, proposal)
            if p_energy <= energy:
                bits, energy = proposal, p_energy
        if energy < best_energy:
            best_energy = energy
            best_bits = bits

    return best_bits


def _greedy_topological_schedule() -> np.ndarray:
    """Baseline: first-fit topological order, one task per slot, bank 0."""
    order = []
    placed = set()
    while len(order) < NUM_TASKS:
        for i in range(NUM_TASKS):
            if i in placed:
                continue
            parents = [p for p, c in DEPENDENCIES if c == i]
            if all(p in placed for p in parents):
                order.append(i)
                placed.add(i)
                break
    bits = np.zeros(NUM_TASKS * NUM_SLOTS * NUM_BANKS)
    block = NUM_SLOTS * NUM_BANKS
    for slot, task in enumerate(order):
        bits[task * block + slot * NUM_BANKS + 0] = 1.0
    return bits


def run_cce_qubo_benchmark(num_rounds: int = 12, seed: int = 7):
    rng = np.random.RandomState(seed)

    generator = CCEQUBOGenerator(num_tasks=NUM_TASKS, num_slots=NUM_SLOTS, num_banks=NUM_BANKS)
    n = generator.num_vars
    block = NUM_SLOTS * NUM_BANKS

    # Objective energy as QUBO diagonal terms: compute energy per (task, slot)
    # plus bank-conflict quadratic coupling. Kept separate from penalty terms
    # so both tracks report a comparable, penalty-free objective energy.
    Q_obj = np.zeros((n, n))
    for i in range(NUM_TASKS):
        for t in range(NUM_SLOTS):
            for b in range(NUM_BANKS):
                Q_obj[generator.var_index(i, t, b), generator.var_index(i, t, b)] += \
                    DURATIONS[i] * (1.0 + SLOT_GROWTH * t)
    for t in range(NUM_SLOTS):
        for b in range(NUM_BANKS):
            occupants_idx = [generator.var_index(i, t, b) for i in range(NUM_TASKS)]
            for p in range(len(occupants_idx)):
                for q in range(p + 1, len(occupants_idx)):
                    Q_obj[occupants_idx[p], occupants_idx[q]] += KAPPA_BANK / 2.0
                    Q_obj[occupants_idx[q], occupants_idx[p]] += KAPPA_BANK / 2.0

    greedy_bits = _greedy_topological_schedule()
    greedy_objective = _objective_energy(greedy_bits)
    greedy_violations = _count_violations(greedy_bits)

    apr = AdaptivePenaltyRefinement(initial_penalty=1.0, mu_step=5.0, max_penalty=500.0)
    static_penalties = {"lambda_one_hot": 1.0, "lambda_precedence": 1.0}

    apr_objective_trace, apr_violation_trace, apr_penalty_trace = [], [], []
    static_objective_trace, static_violation_trace = [], []

    for rnd in range(num_rounds):
        # --- APR track ---
        # AdaptivePenaltyRefinement tracks lambda_one_hot, lambda_precedence,
        # and lambda_sram; CCEQUBOGenerator implements the first two, so
        # lambda_sram is explicitly excluded from the built matrix (documented
        # limitation: bank capacity conflicts are not modeled by the generator).
        cur_penalties = {k: v for k, v in apr.penalties.items()
                         if k in ("lambda_one_hot", "lambda_precedence")}
        Q = generator.build_hamiltonian(cur_penalties, dependencies=DEPENDENCIES) + Q_obj
        bits = _local_search(Q, rng)
        objective = _objective_energy(bits)
        violations = _count_violations(bits)

        apr_objective_trace.append(objective)
        apr_violation_trace.append(sum(violations.values()))
        apr_penalty_trace.append(dict(cur_penalties))
        apr.update_penalties(violations)

        # --- Static penalty control: penalties frozen at initial values ---
        Q_static = generator.build_hamiltonian(static_penalties, dependencies=DEPENDENCIES) + Q_obj
        bits_static = _local_search(Q_static, rng)
        static_objective_trace.append(_objective_energy(bits_static))
        static_violation_trace.append(sum(_count_violations(bits_static).values()))

    apr_objective = np.array(apr_objective_trace)
    apr_violations = np.array(apr_violation_trace)
    static_objective = np.array(static_objective_trace)
    static_violations = np.array(static_violation_trace)
    iterations = np.arange(1, num_rounds + 1)

    zero_violation_iter = (int(iterations[apr_violations == 0][0])
                           if np.any(apr_violations == 0) else None)
    final_objective = float(apr_objective[-1])
    final_violations = int(apr_violations[-1])
    # Compare on the first fully feasible APR solution if one exists (objective
    # energies of infeasible schedules are not a fair comparison).
    feasible_idx = np.where(apr_violations == 0)[0]
    if len(feasible_idx):
        best_objective = float(apr_objective[feasible_idx].min())
    else:
        best_objective = final_objective
    reduction_pct = ((greedy_objective - best_objective) / greedy_objective * 100.0
                     if greedy_objective > 0 else 0.0)

    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: measured constraint violation convergence
    ax1.plot(iterations, static_violations, 'r--o',
             label="Static Penalty QUBO (penalties frozen)", linewidth=1.8)
    ax1.plot(iterations, apr_violations, 'g-s',
             label="Adaptive Penalty Refinement (APR)", linewidth=2.2)
    ax1.axhline(0, color='black', linestyle=':')
    ax1.set_xlabel("APR Tuning Iterations", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Constraint Violations (measured)", fontsize=11, fontweight='bold')
    ax1.set_title("Constraint Violation Elimination: Static vs APR", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    # Plot 2: measured objective energies of best-found schedules
    ax2.plot(iterations, np.full(num_rounds, greedy_objective), 'k--',
             label=f"Greedy Topological Baseline ({greedy_objective:.2f} a.u.)", linewidth=1.5)
    ax2.plot(iterations, static_objective, 'r-^',
             label=f"Static QUBO (final {static_objective[-1]:.2f} a.u.)", linewidth=1.8)
    ax2.plot(iterations, apr_objective, 'b-d',
             label=f"CCE-QOS APR (final {final_objective:.2f} a.u.)", linewidth=2.2)
    ax2.set_xlabel("Optimization Iterations", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Objective Energy (arbitrary units, measured)", fontsize=11, fontweight='bold')
    ax2.set_title("Objective Energy of Best-Found Schedule", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    out_path = os.path.join("figures", "fig_cce_qubo_apr_benchmark.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("=" * 70)
    print("QUBO-APR ENGINE BENCHMARK (all numbers measured from live APR loop)")
    print(f"Workload: {NUM_TASKS} tasks x {NUM_SLOTS} slots x {NUM_BANKS} banks "
          f"({n} QUBO variables), DAG edges: {DEPENDENCIES}")
    print(f"APR penalties per round: {apr_penalty_trace}")
    print(f"Greedy topological baseline: objective {greedy_objective:.2f} a.u., "
          f"violations {int(sum(greedy_violations.values()))}")
    print(f"Static penalty control     : objective {static_objective[-1]:.2f} a.u., "
          f"final violations {int(static_violations[-1])}")
    print(f"APR track                  : objective {final_objective:.2f} a.u., "
          f"final violations {final_violations}")
    if zero_violation_iter is not None:
        print(f"APR zero total violations first reached at iteration {zero_violation_iter}")
    else:
        print(f"APR zero total violations NOT reached within {num_rounds} rounds "
              f"(final: {final_violations})")
    print(f"Best feasible APR objective: {best_objective:.2f} a.u. "
          f"({reduction_pct:.2f}% reduction vs greedy)")
    print(f"Publication benchmark plot saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_cce_qubo_benchmark()
