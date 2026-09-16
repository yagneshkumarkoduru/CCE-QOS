"""
CCE-QOS Scaling Benchmark: Real Scheduler + Cost Model
Tests Greedy, Simulated Annealing, and Lookahead scheduling on synthetic DAGs
of increasing size using the real SchedulingEngine and ScheduleCostModel from
the pipeline (config.yaml hardware, fusion, and cost weights).
"""
import argparse, sys, json, random, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_types import OperatorNode, OperatorGraph
from scheduling_engine import SchedulingEngine
from cost_model import ScheduleCostModel
from memory_hierarchy import MemoryHierarchy
from bandwidth_estimator import BandwidthEstimator
from fusion_logic import FusionLogic


def load_config(path):
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import yaml  # type: ignore
        return yaml.safe_load(raw)


def make_synthetic_graph(n_nodes, seed=42):
    """Build a synthetic DAG with realistic tensor sizes and dependencies."""
    rng = random.Random(seed)
    node_types = ['conv', 'norm', 'act', 'pool', 'fc', 'attn', 'ffn']
    nodes = []

    for i in range(n_nodes):
        # 0-3 predecessors (topological constraint)
        n_deps = rng.randint(0, min(3, i))
        deps = sorted(rng.sample(range(i), n_deps)) if n_deps > 0 and i > 0 else []

        input_bytes = rng.uniform(100, 800)
        output_bytes = rng.uniform(100, 800)
        compute_cycles = rng.randint(10, 300)
        flops = compute_cycles * 1e6

        op = OperatorNode(
            id=i,
            name=f"op_{i}",
            op_type=rng.choice(node_types),
            flops=flops,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compute_cycles=compute_cycles,
            dependencies=deps,
            reuse_groups=[],
            fusible_with=[],
            attrs={
                "criticality": round(rng.uniform(0.5, 2.0), 2),
                "sram_hint": round(rng.uniform(100, 500), 1),
                "volatility": round(rng.uniform(0.01, 0.2), 2),
            }
        )
        nodes.append(op)

    # Build edges from dependencies
    edges = []
    for i, nd in enumerate(nodes):
        for d in nd.dependencies:
            edges.append((d, i))

    return OperatorGraph(nodes, edges)


def main():
    parser = argparse.ArgumentParser(description="CCE-QOS scaling benchmark (real scheduler)")
    parser.add_argument("--sizes", default="16,32,64", help="Comma-separated node counts")
    parser.add_argument("--seeds", default="42", help="Comma-separated DAG seeds")
    parser.add_argument("--sa-iterations", type=int, default=320, help="SA iterations")
    parser.add_argument("--sa-start-temp", type=float, default=3.8, help="SA start temperature")
    parser.add_argument("--sa-end-temp", type=float, default=0.04, help="SA end temperature")
    parser.add_argument("--lookahead-depth", type=int, default=2, help="Lookahead depth")
    parser.add_argument("--tag", default="", help="Suffix for output filename")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config file")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    hw = dict(config.get("hardware", {}))
    fusion = dict(config.get("fusion", {}))
    weights = dict(config.get("cost_weights", {}))
    base_penalties = dict(config.get("apr", {}).get("initial_penalties", {}))

    SIZES = [int(s) for s in args.sizes.split(",") if s.strip()]
    SEEDS = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = []
    print("\nCCE-QOS Scaling Benchmark (Real Scheduler + Cost Model)")
    print(f"{'Seed':>5} | {'Nodes':>6} | {'Greedy':>10} | {'F%':>5} | {'SA':>10} | {'F%':>5} | {'SA %imp':>8} | "
          f"{'Lookahead':>10} | {'F%':>5} | {'LA %imp':>8} | {'LA time':>8}")
    print("-" * 122)

    for seed in SEEDS:
        for n in SIZES:
            graph = make_synthetic_graph(n, seed=seed)
            scheduler = SchedulingEngine(graph, random_seed=seed)
            cost_model = ScheduleCostModel(
                memory_hierarchy=MemoryHierarchy(dict(hw)),
                bandwidth_estimator=BandwidthEstimator(dict(hw)),
                fusion_logic=FusionLogic(dict(fusion)),
                weights=dict(weights),
            )

            def evaluate_full(order):
                return cost_model.evaluate(graph, list(order), penalties=dict(base_penalties))

            def evaluator(order):
                return float(evaluate_full(order)["breakdown"]["total_cost"])

            # Greedy
            t0 = time.perf_counter()
            g_result = scheduler.greedy(penalties=dict(base_penalties))
            g_ev = evaluate_full(g_result.order)
            g_time = time.perf_counter() - t0

            # SA
            t0 = time.perf_counter()
            sa_result = scheduler.simulated_annealing(
                penalties=dict(base_penalties),
                evaluator=evaluator,
                iterations=args.sa_iterations,
                start_temp=args.sa_start_temp,
                end_temp=args.sa_end_temp,
            )
            sa_ev = evaluate_full(sa_result.order)
            sa_time = time.perf_counter() - t0

            # Lookahead
            t0 = time.perf_counter()
            la_result = scheduler.lookahead(
                penalties=dict(base_penalties),
                lookahead_depth=args.lookahead_depth,
                evaluator=evaluator,
            )
            la_ev = evaluate_full(la_result.order)
            la_time = time.perf_counter() - t0

            g_cost = float(g_ev["breakdown"]["total_cost"])
            sa_cost = float(sa_ev["breakdown"]["total_cost"])
            la_cost = float(la_ev["breakdown"]["total_cost"])
            g_feas = float(g_ev["feasibility"])
            sa_feas = float(sa_ev["feasibility"])
            la_feas = float(la_ev["feasibility"])

            sa_imp = (g_cost - sa_cost) / g_cost * 100 if g_cost > 0 else 0.0
            la_imp = (g_cost - la_cost) / g_cost * 100 if g_cost > 0 else 0.0

            print(f"{seed:>5} | {n:>6} | {g_cost:>10.1f} | {g_feas:>4.0%} | {sa_cost:>10.1f} | {sa_feas:>4.0%} | "
                  f"{sa_imp:>7.1f}% | {la_cost:>10.1f} | {la_feas:>4.0%} | {la_imp:>7.1f}% | {la_time*1000:>6.0f}ms")

            rows.append({
                "seed": seed,
                "n_nodes": n,
                "greedy_cost": round(g_cost, 2),
                "greedy_feasibility": round(g_feas, 4),
                "greedy_time_ms": round(g_time * 1000, 1),
                "sa_cost": round(sa_cost, 2),
                "sa_feasibility": round(sa_feas, 4),
                "sa_improvement_pct": round(sa_imp, 2),
                "sa_time_ms": round(sa_time * 1000, 1),
                "lookahead_cost": round(la_cost, 2),
                "lookahead_feasibility": round(la_feas, 4),
                "lookahead_improvement_pct": round(la_imp, 2),
                "lookahead_time_ms": round(la_time * 1000, 1),
            })

    print()
    print("Cost model: ScheduleCostModel (DRAM, SRAM, bandwidth, bank conflicts) from config.yaml")
    print(f"SA: {args.sa_iterations} iterations, T0={args.sa_start_temp} -> {args.sa_end_temp}")
    print(f"Lookahead: depth={args.lookahead_depth}")

    out = ROOT / "results" / "dag_scaling"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"real_scheduler_scaling{suffix}.json").write_text(json.dumps({
        "description": "CCE-QOS solver scaling using real SchedulingEngine and ScheduleCostModel",
        "class": "BENCHMARK (Python scheduler model, synthetic NPU DAGs, real cost function)",
        "config": {
            "sizes": SIZES,
            "seeds": SEEDS,
            "sa_iterations": args.sa_iterations,
            "sa_start_temp": args.sa_start_temp,
            "sa_end_temp": args.sa_end_temp,
            "lookahead_depth": args.lookahead_depth,
        },
        "results": rows,
    }, indent=2))
    print(f"\nSaved: {out}/real_scheduler_scaling{suffix}.json")


if __name__ == "__main__":
    main()
