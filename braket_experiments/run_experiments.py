"""Braket experiment orchestrator for CCE-QOS.

Commands:

- ``local``            optimize angles locally and run the exact local
                       simulator for every instance and depth 1-3
- ``submit``           submit QAOA tasks to a managed simulator or QPU
                       (idempotent: already-submitted configs are skipped)
- ``aquila``           submit one analog Rydberg program
- ``aquila-campaign``  submit the full Aquila matrix (chain3/chain5 x
                       three detuning schedules, 1000 shots each)
- ``collect``          poll and evaluate all submitted ledger tasks
- ``report``           build results JSON, cost estimate, and REPORT.md
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
INSTANCES = ("chain3", "chain5", "random11")
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


BUDGET_GATE_USD = 100.0
SHOT_OPTIONS = (100, 250, 500, 1000)


def _batch_cost(device_key: str, shots: int, tasks: int) -> float:
    if device_key == "ionq":
        return tasks * (0.30 + 0.08 * shots)
    if device_key == "aquila":
        return tasks * (0.30 + 0.01 * shots)
    return 0.0


def cmd_status(args: argparse.Namespace) -> int:
    rows = runner.task_states()
    state_counts: dict[str, int] = {}
    for row in rows:
        state = row["aws_state"]
        state_counts[state] = state_counts.get(state, 0) + 1
        print(
            f"{row['task_arn'][-12:]} {row['device_key']:7s} "
            f"{row['instance']:9s} p={row.get('depth', 0)} "
            f"shots={row['shots']:5d} schedule={row.get('schedule', '-')[:11]:11s} "
            f"ledger={row['status']:10s} aws={state}"
        )
    print()
    print("aws states:", state_counts)
    print(f"estimated QPU spend (submitted configs): ${ledger.estimate_spend(rows):.2f}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    import math

    for shots in SHOT_OPTIONS:
        cost = _batch_cost(args.device, shots, args.tasks)
        se = 1.5 / math.sqrt(shots) if shots else float("inf")
        print(f"  {args.tasks} tasks x {shots:5d} shots: ${cost:8.2f}   SE(mean energy) ~ {se:.3f}")
    needed = math.ceil((1.5 / args.target_se) ** 2)
    print()
    print(f"shots needed for SE {args.target_se}: ~{needed} per config")
    print(f"cost at that shot count: ${_batch_cost(args.device, needed, args.tasks):.2f}")
    print()
    print("guidance: choose the smallest shot count whose SE is well below the")
    print("effect size being measured; 200-300 shots is usually sufficient for")
    print("hardware-vs-ideal comparisons at these instance sizes.")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    depths = tuple(args.depths) if args.depths else DEFAULT_DEPTHS
    new_configs = [
        (instance, depth)
        for instance in INSTANCES
        for depth in depths
        if not ledger.has_submission(args.device, instance, depth, args.shots)
    ]
    estimated = _batch_cost(args.device, args.shots, len(new_configs))
    print(f"batch: {len(new_configs)} new tasks x {args.shots} shots -> ${estimated:.2f}")
    if estimated > BUDGET_GATE_USD and not args.yes:
        print(
            f"REFUSED: estimated batch cost exceeds the ${BUDGET_GATE_USD:.0f} gate; "
            "re-run with --yes after reviewing the batch (see `estimate`)."
        )
        return 1
    for instance, depth in new_configs:
        solver, gamma, beta, _ = optimized_angles(instance, depth)
        try:
            task_arn = runner.submit(
                args.device, instance, depth, solver, gamma, beta, args.shots
            )
            print(f"submitted {instance} p={depth} @{args.shots} -> {task_arn}")
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"FAILED {instance} p={depth}: {exc}")
    return 0


def cmd_aquila(args: argparse.Namespace) -> int:
    instance = f"chain{args.num_atoms}"
    if ledger.has_ahs_submission(instance, args.schedule, args.shots):
        print(f"skip aquila {instance} {args.schedule} already submitted")
        return 0
    try:
        task_arn = aquila_experiment.submit_aquila(
            shots=args.shots, num_atoms=args.num_atoms, schedule=args.schedule
        )
        print(f"submitted aquila {instance} {args.schedule} -> {task_arn}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED aquila {instance} {args.schedule}: {exc}")
    return 0


def cmd_aquila_campaign(args: argparse.Namespace) -> int:
    for num_atoms in (3, 5):
        instance = f"chain{num_atoms}"
        for schedule in aquila_experiment.SCHEDULES:
            if ledger.has_ahs_submission(instance, schedule, args.shots):
                print(f"skip aquila {instance} {schedule} already submitted")
                continue
            try:
                task_arn = aquila_experiment.submit_aquila(
                    shots=args.shots, num_atoms=num_atoms, schedule=schedule
                )
                print(f"submitted aquila {instance} {schedule} -> {task_arn}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED aquila {instance} {schedule}: {exc}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    summaries = runner.sweep_collect(args.max_wait, args.poll)
    for summary in summaries:
        print(f"{summary['task_arn'][-12:]} {summary['status']} {summary.get('state', '')}")
    remaining = len(ledger.pending_tasks())
    print(f"collected {len(summaries)} task(s); {remaining} still pending")
    if not summaries and remaining == 0:
        print("nothing to collect")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    local = json.loads(LOCAL_JSON.read_text(encoding="utf-8")) if LOCAL_JSON.exists() else {"runs": []}
    tasks = ledger.load_ledger()["tasks"]
    cloud = [entry for entry in tasks if entry.get("status") == "COLLECTED"]
    pending = [entry for entry in tasks if entry.get("status") == "SUBMITTED"]
    estimated_spend = ledger.estimate_spend(tasks)
    payload = {
        "schema": "cce-qos-braket-results-v1",
        "devices": runner.DEVICE_ARNS,
        "costs_read_from_pricing_api_us_east_1": {
            "ionq_forte_enterprise_1": {"per_task_usd": 0.30, "per_shot_usd": 0.08},
            "quera_aquila": {"per_task_usd": 0.30, "per_shot_usd": 0.01},
            "managed_simulators": "billed per minute of execution",
        },
        "estimated_qpu_spend_usd": round(estimated_spend, 2),
        "pending_tasks": len(pending),
        "local_runs": local.get("runs", []),
        "cloud_runs": cloud,
    }
    _write_json(RESULTS_JSON, payload)

    lines = [
        "# CCE-QOS on Amazon Braket",
        "",
        "QAOA and analog Ising experiments executed on Amazon Braket devices.",
        f"Estimated QPU spend from published rates: ${estimated_spend:.2f}",
        f"(pending tasks: {len(pending)}).",
        "IonQ Forte Enterprise 1 $0.30/task + $0.08/shot; QuEra Aquila",
        "$0.30/task + $0.01/shot; managed simulators per minute (AWS Pricing API,",
        "us-east-1, read during the experiment session).",
        "",
        "| instance | p | device | shots | schedule | mean energy | best energy | ratio(mean) | ratio(best) | ground prob | task |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in local.get("runs", []):
        stats = run["local_simulator"]
        lines.append(
            f"| {run['instance']} | {run['depth']} | local-sim | {stats['shots']} | - | "
            f"{stats['mean_energy']:.4f} | {stats['best_energy']:.4f} | "
            f"{stats['approx_ratio_mean']:.4f} | {stats['approx_ratio_best']:.4f} | "
            f"{stats['ground_state_probability']:.4f} | - |"
        )
    for entry in cloud:
        stats = entry.get("stats", {})
        lines.append(
            f"| {entry['instance']} | {entry.get('depth', 0)} | {entry['device_key']} | "
            f"{stats.get('shots', entry.get('shots'))} | "
            f"{entry.get('schedule', '-')} | "
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
    print(f"estimated QPU spend: ${estimated_spend:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braket_experiments", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    local = sub.add_parser("local", help="exact local-simulator runs")
    local.add_argument("--shots", type=int, default=4000)
    local.set_defaults(func=cmd_local)

    submit = sub.add_parser("submit", help="submit QAOA tasks")
    submit.add_argument("--device", required=True, choices=["sv1", "dm1", "ionq"])
    submit.add_argument("--shots", type=int, default=250)
    submit.add_argument("--depths", type=int, nargs="*", default=None)
    submit.add_argument(
        "--yes",
        action="store_true",
        help=f"confirm batches estimated above ${BUDGET_GATE_USD:.0f}",
    )
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="non-blocking state table for all tasks")
    status.set_defaults(func=cmd_status)

    estimate = sub.add_parser("estimate", help="cost/statistics planner")
    estimate.add_argument("--device", default="ionq", choices=["sv1", "dm1", "ionq", "aquila"])
    estimate.add_argument("--tasks", type=int, default=9)
    estimate.add_argument("--target-se", type=float, default=0.10)
    estimate.set_defaults(func=cmd_estimate)

    aquila = sub.add_parser("aquila", help="submit one analog Rydberg program")
    aquila.add_argument("--shots", type=int, default=1000)
    aquila.add_argument("--num-atoms", type=int, default=3)
    aquila.add_argument("--schedule", default="slow_sweep", choices=list(aquila_experiment.SCHEDULES))
    aquila.set_defaults(func=cmd_aquila)

    campaign = sub.add_parser("aquila-campaign", help="submit the full Aquila matrix")
    campaign.add_argument("--shots", type=int, default=1000)
    campaign.set_defaults(func=cmd_aquila_campaign)

    collect = sub.add_parser("collect", help="poll and evaluate submitted tasks")
    collect.add_argument("--max-wait", type=float, default=900.0)
    collect.add_argument("--poll", type=float, default=20.0)
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
