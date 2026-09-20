"""Multi-workload statistical sweep of the CCE-QOS pipeline.

Protocol strengthening for the single-synthetic-DAG limitation: the pipeline
is run over a seeded population of layered DAGs (benchmarks/workload_generator.py)
and every headline strategy is reported as mean +/- std across workloads.
Repository-root canonical outputs are left untouched (write_root_copies=False).

Usage:
    py -3 benchmarks/sweep_workloads.py --family perturb --seeds 6
    py -3 benchmarks/sweep_workloads.py --family layered --sizes 24,31,48 --seeds 2
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_experiment import load_config, run_pipeline  # noqa: E402
from schedule_analysis import ScheduleAnalysis  # noqa: E402
from workload_generator import perturb_workload, write_workload  # noqa: E402

STRATEGY_KEYS = ("total_cost", "total_energy", "latency", "feasibility_percent")
HEADLINE = (
    "baseline::Greedy",
    "baseline::Lookahead",
    "baseline::Beam Search",
    "baseline::Simulated Annealing",
    "cce::CCE + APR",
    "quantum::candidates",
)


def _summaries(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    analysis = ScheduleAnalysis()
    out: Dict[str, Dict[str, float]] = {}
    for name, payload in report["results"]["baseline_cost"].items():
        out[f"baseline::{name}"] = analysis.summarize(name, payload["evaluation"])
    for name, payload in report["results"]["cce_qubo"].items():
        out[f"cce::{name}"] = analysis.summarize(name, payload["evaluation"])
    out["cce::CCE + APR"] = analysis.summarize("CCE + APR", report["results"]["cce_qubo_apr"]["evaluation"])
    out["quantum::candidates"] = analysis.summarize(
        "Quantum", report["results"]["quantum_candidates"]["evaluation"]
    )
    return out


def _mean_std(values: List[float]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean, 4), round(std, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="CCE-QOS multi-workload sweep")
    parser.add_argument("--family", choices=["perturb", "layered"], default="perturb",
                        help="perturb: seeded parameter perturbations of the reference DAG; layered: structural synthetic family")
    parser.add_argument("--sizes", type=str, default="24,31,48")
    parser.add_argument("--seeds", type=int, default=3, help="seed count per size (layered) or family (perturb)")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / "sweeps" / stamp
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    reference = json.loads((ROOT / "example_workload.json").read_text(encoding="utf-8"))
    if args.family == "perturb":
        cases = [("reference", 9000 + i, 31) for i in range(args.seeds)]
    else:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        cases = [(f"{size}n", 1000 * size + i, size) for size in sizes for i in range(args.seeds)]

    rows: List[Dict[str, Any]] = []
    for label, seed, size in cases:
        wl_path = out_dir / "workloads" / f"wl_{label}_seed{seed}.json"
        if args.family == "perturb":
            workload = perturb_workload(reference, seed)
            wl_path.parent.mkdir(parents=True, exist_ok=True)
            wl_path.write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8", newline="\n")
        else:
            write_workload(wl_path, size, seed)
        config = copy.deepcopy(load_config(ROOT / "config.yaml"))
        config["input"]["workload"] = str(wl_path.relative_to(ROOT)).replace("\\", "/")
        run_dir = out_dir / "runs" / f"wl_{label}_seed{seed}"
        config["output"]["directory"] = str(run_dir.relative_to(ROOT)).replace("\\", "/")
        started = time.perf_counter()
        report = run_pipeline(config, ROOT, write_root_copies=False)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "family": args.family,
                "label": label,
                "seed": seed,
                "workload": str(wl_path.relative_to(ROOT)).replace("\\", "/"),
                "num_nodes": report["workload"]["num_nodes"],
                "num_edges": report["workload"]["num_edges"],
                "runtime_seconds": round(elapsed, 3),
                "summaries": _summaries(report),
            }
        )
        print(f"completed {label} seed {seed}: {elapsed:.1f}s ({len(rows)} workloads)", flush=True)

    aggregate: Dict[str, Any] = {
        "family": args.family,
        "workloads": len(rows),
        "seeds": args.seeds,
        "by_strategy": {},
    }
    for name in HEADLINE:
        stats_per_key: Dict[str, Optional[Tuple[float, float]]] = {}
        for key in STRATEGY_KEYS:
            values = [r["summaries"][name][key] for r in rows if name in r["summaries"]]
            stats_per_key[key] = _mean_std(values)
        aggregate["by_strategy"][name] = stats_per_key

    summary = {"aggregate": aggregate, "rows": rows}
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    lines = [f"CCE-QOS multi-workload sweep ({len(rows)} workloads, family {args.family})", ""]
    for name, stats in aggregate["by_strategy"].items():
        cost = stats["total_cost"]
        feas = stats["feasibility_percent"]
        if cost is None or feas is None:
            lines.append(f"{name:32s} (no data)")
            continue
        lines.append(
            f"{name:32s} cost {cost[0]:10.2f} +/- {cost[1]:7.2f}   feasible {feas[0]:6.2f} +/- {feas[1]:5.2f} %"
        )
    lines.append("")
    lines.append(f"written: {out_dir / 'sweep_summary.json'}")
    (out_dir / "sweep_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
