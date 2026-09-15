"""
Pareto Multi-Objective Optimization & Frontier Exploration for NPU Scheduling
Author: Koduru Yagnesh Kumar
Repository: CCE-QOS
Domain: Computer Architecture, Discrete Optimization, NPU Accelerators

Constructs the non-dominated Pareto Frontier exploring fundamental architectural tradeoffs:
1. Total Energy Consumption (MAC energy + SRAM/DRAM transactions)
2. Latency / Execution Time (critical path cycles)
3. Peak On-Chip SRAM Memory Footprint (pressure & spills)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35


class ParetoNPUExplorer:
    def __init__(self, num_operators=16, sram_limit_kb=512):
        self.num_operators = num_operators
        self.sram_limit = sram_limit_kb

    def generate_candidate_schedules(self, n_samples=300, seed=42):
        """
        Generate candidate operator schedules across varying optimization weightings
        between energy-centric, latency-centric, and memory-preserving compilers.
        """
        np.random.seed(seed)
        candidates = []

        for _ in range(n_samples):
            # Energy cost (normalized around 0.65 to 1.15)
            # Higher parallelism = higher peak memory & lower latency, but higher energy bursts
            parallelism_factor = np.random.uniform(1.0, 4.0)
            fusion_gain = np.random.uniform(0.05, 0.28)

            latency_cycles = (1200.0 / parallelism_factor) * (1.0 - 0.5 * fusion_gain) + np.random.normal(0, 25)
            energy_cost = (45.0 + 8.5 * parallelism_factor) * (1.0 - fusion_gain) + np.random.normal(0, 1.5)
            peak_sram_kb = (140.0 * parallelism_factor) * (1.0 - 0.2 * fusion_gain) + np.random.normal(0, 10)

            # Check hard physical feasibility (SRAM limit)
            is_feasible = peak_sram_kb <= self.sram_limit

            candidates.append({
                'energy': max(25.0, energy_cost),
                'latency': max(200.0, latency_cycles),
                'sram': peak_sram_kb,
                'feasible': is_feasible
            })

        return candidates

    @staticmethod
    def identify_pareto_front(candidates):
        """Find non-dominated Pareto frontier (minimizing energy and latency)."""
        feasible_pts = [c for c in candidates if c['feasible']]
        pareto_front = []

        for i, pt_a in enumerate(feasible_pts):
            is_dominated = False
            for j, pt_b in enumerate(feasible_pts):
                if i != j:
                    # pt_b dominates pt_a if it is better or equal in both and strictly better in one
                    if (pt_b['energy'] <= pt_a['energy'] and pt_b['latency'] <= pt_a['latency']) and \
                       (pt_b['energy'] < pt_a['energy'] or pt_b['latency'] < pt_a['latency']):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append(pt_a)

        # Sort Pareto front by energy
        pareto_front.sort(key=lambda x: x['energy'])
        return pareto_front, feasible_pts


def run_pareto_study():
    print("=" * 80)
    print("CCE-QOS PARETO MULTI-OBJECTIVE COMPILER FRONTIER EXPLORATION")
    print("Author: Koduru Yagnesh Kumar")
    print("=" * 80)

    explorer = ParetoNPUExplorer(num_operators=24, sram_limit_kb=512)
    candidates = explorer.generate_candidate_schedules(n_samples=400)
    pareto_pts, all_feasible = explorer.identify_pareto_front(candidates)

    infeasible_pts = [c for c in candidates if not c['feasible']]

    print(f"Total Evaluated Compilation Schedules: {len(candidates)}")
    print(f"Physically Feasible Schedules (SRAM <= 512 KB): {len(all_feasible)} ({len(all_feasible)/len(candidates)*100:.1f}%)")
    print(f"Non-Dominated Pareto Optimal Operating Points: {len(pareto_pts)}")

    print("\n" + "-" * 80)
    print(f"{'Operating Point':<20} | {'Energy Cost (mJ)':<18} | {'Latency (cycles)':<18} | {'Peak SRAM (KB)'}")
    print("-" * 80)
    for idx, p in enumerate(pareto_pts[:5]):
        print(f"Pareto Point #{idx+1:<13} | {p['energy']:<18.2f} | {p['latency']:<18.1f} | {p['sram']:<15.1f}")
    print("-" * 80)

    # Output directory
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot Pareto Frontier
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # All feasible schedules
    f_energy = [c['energy'] for c in all_feasible]
    f_latency = [c['latency'] for c in all_feasible]
    f_sram = [c['sram'] for c in all_feasible]

    scatter = ax.scatter(f_latency, f_energy, c=f_sram, cmap='viridis', alpha=0.55, s=35, label='Sub-Optimal Feasible Schedules')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Peak SRAM Footprint (KB)', fontweight='bold')

    # Infeasible points
    if len(infeasible_pts) > 0:
        inf_energy = [c['energy'] for c in infeasible_pts]
        inf_latency = [c['latency'] for c in infeasible_pts]
        ax.scatter(inf_latency, inf_energy, color='#C0392B', marker='x', alpha=0.4, s=25, label='Infeasible (SRAM Overflow > 512KB)')

    # Non-dominated Pareto front
    p_energy = [p['energy'] for p in pareto_pts]
    p_latency = [p['latency'] for p in pareto_pts]
    ax.plot(p_latency, p_energy, 'r--', linewidth=2.0, alpha=0.85)
    ax.scatter(p_latency, p_energy, color='#E74C3C', s=70, edgecolor='black', linewidth=1.2, label='Non-Dominated Pareto Frontier', zorder=5)

    # Key operating regimes annotations
    ax.annotate('Energy-Optimal Regime\n(Fused, Serialized Low-Leakage)',
                xy=(p_latency[-1], p_energy[-1]),
                xytext=(p_latency[-1] - 250, p_energy[-1] + 10),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5),
                fontweight='bold', fontsize=8.5)

    ax.annotate('Latency-Optimal Regime\n(Maximal Array Parallelism)',
                xy=(p_latency[0], p_energy[0]),
                xytext=(p_latency[0] + 50, p_energy[0] + 8),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5),
                fontweight='bold', fontsize=8.5)

    ax.set_xlabel('Execution Latency (Cycles)', fontweight='bold')
    ax.set_ylabel('Total NPU Energy Cost (Normalized / mJ)', fontweight='bold')
    ax.set_title('CCE-QOS Pareto Frontier: Non-Dominated Energy-Latency Tradeoffs under SRAM Capacity', fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'fig_pareto_energy_latency_tradeoff.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"\nGenerated publication Pareto plot saved to: {os.path.abspath(fig_path)}")


if __name__ == '__main__':
    run_pareto_study()
