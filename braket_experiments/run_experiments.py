"""Braket experiment orchestrator for CCE-QOS.

Commands:

- ``local``        optimize angles locally and run the exact local
                   simulator for both instances and depths 1-3
- ``submit``       submit QAOA tasks to a managed simulator or QPU
- ``aquila``       submit the analog Rydberg chain program
- ``collect``      poll and evaluate all submitted ledger tasks
- ``report``       build results/quantum_braket/qaoa_braket_results.json
                   and REPORT.md from the ledger and local runs
"""

from . import attribution  # noqa: F401  (attribution must precede SDK use)

import argparse
import json

from QAOA_solver import DEFAULT_QAOA_SEED, QAOASolver

from . import aquila_experiment, ledger, runner
from .instances import get_instance

RESULTS_DIR = ledger.RESULTS_DIR
LOCAL_JSON = RESULTS_DIR / "qaoa_local_simulator.json"
RESULTS_JSON = RESULTS_DIR / "qaoa_braket_results.json"
REPORT_MD = RESULTS_DIR / "REPORT.md"
INSTANCES = ("chain3", "random11")
DEFAULT_DEPTHS = (1, 2, 3)
OPTIMIZER_BUDGET = 60


def optimized_angles(instance: str, depth: int):
    solver = QAOASolver(get_instance(instance), p=depth)
    gamma, beta, energy = solver.optimize_parameters_cobyla(
        max_iter=OPTIMIZER_BUDGET, seed=DEFAULT_QAOA_SEED
    )
    return solver, gamma, beta, energy


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def cmd_local(args: argparse.Namespace) -> int:
    runs = []
    for instance in INSTANCES:
        for depth in DEFAULT_DEPTHS:
            solver, gamma, beta, energy = optimized_angles(instance, depth)
            stats = runner.run_local_simulator(solver, gamma, beta, shots=args.shots)
            runs.append(
                {
                    "instance": instance,
                    "depth": depth,
                    "gamma": [float(value) for value in gamma],
                    "beta": [float(value) for value in beta],
                    "optimizer_energy": float(energy),
                    "optimizer_approx_ratio": float(solver.approximation_ratio(energy)),
                    "e_min": float(solver.e_min),
                    "e_max": float(solver.e_max),
                    "num_qubits": solver.num_qubits,
                    "local_simulator": stats,
                }
            )
            print(
                f"local {instance} p={depth}: mean={stats['mean_energy']:.4f} "
                f"best={stats['best_energy']:.4f} "
                f"ratio_mean={stats['approx_ratio_mean']:.4f}"
            )
    _write_json(
        LOCAL_JSON,
        {
            "schema": "cce-qos-braket-local-v1",
            "seed": DEFAULT_QAOA_SEED,
            "optimizer_budget": OPTIMIZER_BUDGET,
            "runs": runs,
        },
    )
    print(f"wrote {LOCAL_JSON}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    depths = tuple(args.depths) if args.depths else DEFAULT_DEPTHS
    for instance in INSTANCES:
        for depth in depths:
            solver, gamma, beta, _ = optimized_angles(instance, depth)
            task_arn = runner.submit(
                args.device, instance, depth, solver, gamma, beta, args.shots
            )
            print(f"submitted {instance} p={depth} -> {task_arn}")
    return 0


def cmd_aquila(args: argparse.Namespace) -> int:
    task_arn = aquila_experiment.submit_aquila(shots=args.shots)
    print(f"submitted aquila chain3 -> {task_arn}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    summaries = runner.collect_all(args.max_wait, args.poll)
    for summary in summaries:
        print(
            f"{summary['task_arn'][-12:]} {summary['status']} "
            f"{summary.get('state', '')}"
        )
    if not summaries:
        print("no pending tasks")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    local = json.loads(LOCAL_JSON.read_text(encoding="utf-8")) if LOCAL_JSON.exists() else {"runs": []}
    cloud = [
        entry
        for entry in ledger.load_ledger()["tasks"]
        if entry.get("status") == "COLLECTED"
    ]
    payload = {
        "schema": "cce-qos-braket-results-v1",
        "devices": runner.DEVICE_ARNS,
        "costs_read_from_pricing_api_us_east_1": {
            "ionq_forte_enterprise_1": {"per_task_usd": 0.30, "per_shot_usd": 0.08},
            "quera_aquila": {"per_task_usd": 0.30, "per_shot_usd": 0.01},
            "managed_simulators": "billed per minute of execution",
        },
        "local_runs": local.get("runs", []),
        "cloud_runs": cloud,
    }
    _write_json(RESULTS_JSON, payload)

    lines = [
        "# CCE-QOS on Amazon Braket",
        "",
        "QAOA and analog Ising experiments executed on Amazon Braket devices.",
        "Costs: IonQ Forte Enterprise 1 $0.30/task + $0.08/shot; QuEra Aquila",
        "$0.30/task + $0.01/shot; managed simulators per minute (AWS Pricing API,",
        "us-east-1, read during the experiment session).",
        "",
        "| instance | p | device | shots | mean energy | best energy | ratio(mean) | ratio(best) | ground prob | task |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in local.get("runs", []):
        stats = run["local_simulator"]
        lines.append(
            f"| {run['instance']} | {run['depth']} | local-sim | {stats['shots']} | "
            f"{stats['mean_energy']:.4f} | {stats['best_energy']:.4f} | "
            f"{stats['approx_ratio_mean']:.4f} | {stats['approx_ratio_best']:.4f} | "
            f"{stats['ground_state_probability']:.4f} | - |"
        )
    for entry in cloud:
        stats = entry.get("stats", {})
        lines.append(
            f"| {entry['instance']} | {entry.get('depth', 0)} | {entry['device_key']} | "
            f"{stats.get('shots', entry.get('shots'))} | "
            f"{stats.get('mean_energy', 0):.4f} | {stats.get('best_energy', 0):.4f} | "
            f"{stats.get('approx_ratio_mean', 0):.4f} | "
            f"{stats.get('approx_ratio_best', 0):.4f} | "
            f"{stats.get('ground_state_probability', 0):.4f} | "
            f"...{entry['task_arn'][-12:]} |"
        )
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {RESULTS_JSON}")
    print(f"wrote {REPORT_MD}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braket_experiments", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    local = sub.add_parser("local", help="exact local-simulator runs")
    local.add_argument("--shots", type=int, default=4000)
    local.set_defaults(func=cmd_local)

    submit = sub.add_parser("submit", help="submit QAOA tasks")
    submit.add_argument("--device", required=True, choices=["sv1", "dm1", "ionq"])
    submit.add_argument("--shots", type=int, default=1000)
    submit.add_argument("--depths", type=int, nargs="*", default=None)
    submit.set_defaults(func=cmd_submit)

    aquila = sub.add_parser("aquila", help="submit the analog Rydberg program")
    aquila.add_argument("--shots", type=int, default=200)
    aquila.set_defaults(func=cmd_aquila)

    collect = sub.add_parser("collect", help="poll and evaluate submitted tasks")
    collect.add_argument("--max-wait", type=float, default=600.0)
    collect.add_argument("--poll", type=float, default=10.0)
    collect.set_defaults(func=cmd_collect)

    report = sub.add_parser("report", help="build results JSON and REPORT.md")
    report.set_defaults(func=cmd_report)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
