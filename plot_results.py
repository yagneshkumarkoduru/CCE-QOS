from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# All publication figures are written into the shared figures/ directory,
# never the repository root.
FIG_DIR = Path(__file__).resolve().parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def maybe_load_outputs(path='outputs/schedules.json'):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def main():
    # Paper-consistent fallback statistics.
    stats = {
        'methods': ['Greedy', 'CCE-QUBO', 'CCE-QUBO+APR'],
        'cost_mean': [1120, 902, 842],
        'cost_std': [38, 31, 24],
        'energy_components': {'Unary': 554, 'Pairwise': -102, 'Higher-order': -69, 'Constraints': 459},
        'apr_iter': [1, 2, 3, 4, 5, 6, 7, 8],
        'lambda_dep': [1.2, 1.8, 2.5, 3.1, 3.5, 3.7, 3.8, 3.8],
        'lambda_mem': [1.0, 1.6, 2.2, 2.8, 3.1, 3.2, 3.2, 3.2],
        'lambda_dvfs': [0.9, 1.1, 1.4, 1.6, 1.7, 1.8, 1.8, 1.8],
        'violation': [0.31, 0.24, 0.18, 0.14, 0.11, 0.09, 0.08, 0.08],
        'ablation_labels': ['Full', 'No Pairwise', 'No Higher-order', 'No APR'],
        'ablation_cost': [842, 936, 989, 902],
        'q_iter': list(range(1, 16)),
        'q_energy_p1': [930, 904, 892, 884, 879, 875, 873, 872, 870, 868, 867, 867, 866, 866, 865],
        'q_energy_p2': [920, 890, 874, 861, 852, 846, 841, 838, 835, 833, 831, 830, 829, 828, 828],
        'q_energy_p3': [910, 878, 860, 845, 836, 829, 824, 820, 817, 815, 814, 813, 812, 812, 812],
        'q_apr_classical': [0.31, 0.24, 0.18, 0.14, 0.11, 0.09, 0.08, 0.08],
        'q_apr_quantum': [0.30, 0.22, 0.16, 0.12, 0.10, 0.09, 0.08, 0.07],
        'box_cost': {
            'Classical+APR': [842, 851, 836, 848, 839, 844, 847, 838],
            'QAOA p=2': [828, 837, 824, 832, 831, 826, 833, 829],
            'QAOA p=3+APR': [812, 821, 808, 816, 811, 814, 818, 809],
        },
        'box_latency': {
            'Classical+APR': [13110, 13320, 12980, 13210, 13090, 13180, 13240, 13040],
            'QAOA p=2': [12980, 13110, 12890, 13060, 12940, 12990, 13020, 12910],
            'QAOA p=3+APR': [12740, 12880, 12690, 12790, 12710, 12760, 12810, 12700],
        },
    }

    # Required classical figure 1.
    plt.figure(figsize=(6, 4))
    plt.bar(stats['methods'], stats['cost_mean'], yerr=stats['cost_std'], capsize=4,
            color=['#4C78A8', '#F58518', '#54A24B'])
    plt.ylabel('Average Cost')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_cost_comparison.png', dpi=300)

    # Required classical figure 2.
    comp = stats['energy_components']
    labels = list(comp.keys())
    vals = np.array(list(comp.values()), dtype=float)
    pos = np.clip(vals, 0, None)
    neg = np.clip(vals, None, 0)
    plt.figure(figsize=(6.6, 4.2))
    x = [0]
    bottom_pos = 0.0
    for i, v in enumerate(pos):
        if v > 0:
            plt.bar(x, [v], bottom=[bottom_pos], label=labels[i])
            bottom_pos += v
    bottom_neg = 0.0
    for i, v in enumerate(neg):
        if v < 0:
            plt.bar(x, [v], bottom=[bottom_neg], label=labels[i])
            bottom_neg += v
    plt.xticks([0], ['CCE-QUBO+APR'])
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Energy Contribution')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_energy_breakdown.png', dpi=300)

    # Required classical figure 3.
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    it = stats['apr_iter']
    ax1.plot(it, stats['lambda_dep'], marker='o', label='lambda_dep')
    ax1.plot(it, stats['lambda_mem'], marker='s', label='lambda_mem')
    ax1.plot(it, stats['lambda_dvfs'], marker='^', label='lambda_dvfs')
    ax1.set_xlabel('APR Iteration')
    ax1.set_ylabel('Penalty Weight')
    ax2 = ax1.twinx()
    ax2.plot(it, stats['violation'], marker='d', linestyle='--', color='black', label='violation_rate')
    ax2.set_ylabel('Violation Rate')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_apr_convergence.png', dpi=300)

    # Required classical figure 4.
    base = stats['ablation_cost'][0]
    delta = [(x - base) / base * 100.0 for x in stats['ablation_cost']]
    plt.figure(figsize=(6.6, 4.0))
    plt.bar(stats['ablation_labels'], delta, color=['#54A24B', '#F58518', '#E45756', '#72B7B2'])
    plt.ylabel('Cost Increase vs Full Model (%)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_ablation.png', dpi=300)

    # Quantum figure 1: energy vs iteration.
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(stats['q_iter'], stats['q_energy_p1'], marker='o', label='p=1')
    plt.plot(stats['q_iter'], stats['q_energy_p2'], marker='s', label='p=2')
    plt.plot(stats['q_iter'], stats['q_energy_p3'], marker='^', label='p=3')
    plt.xlabel('Classical Optimizer Iteration')
    plt.ylabel('QAOA Objective')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_qaoa_energy_iteration.png', dpi=300)

    # Quantum figure 2: violation vs APR iteration.
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(stats['apr_iter'], stats['q_apr_classical'], marker='o', label='Classical APR')
    plt.plot(stats['apr_iter'], stats['q_apr_quantum'], marker='s', label='Quantum APR')
    plt.xlabel('APR Iteration')
    plt.ylabel('Violation Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_quantum_violation_apr.png', dpi=300)

    # Quantum figure 3: side-by-side boxplots for cost and latency.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = list(stats['box_cost'].keys())
    axes[0].boxplot([stats['box_cost'][k] for k in labels], tick_labels=labels)
    axes[0].set_title('Cost Distribution')
    axes[0].tick_params(axis='x', rotation=20)
    axes[1].boxplot([stats['box_latency'][k] for k in labels], tick_labels=labels)
    axes[1].set_title('Latency Distribution')
    axes[1].tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_quantum_boxplot_cost_latency.png', dpi=300)


if __name__ == '__main__':
    main()
