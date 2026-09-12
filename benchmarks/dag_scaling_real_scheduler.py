"""
CCE-QOS Scaling Benchmark: Real Scheduler + Cost Model
Tests solver quality (greedy, SA, lookahead) on synthetic DAGs of 16, 32, 64, 128 nodes.
Uses actual SchedulingEngine with _priority_score cost model.
"""
import sys, json, random, time, pathlib
sys.path.insert(0, r'C:\Research\CCE-QOS')

from core_types import OperatorNode, OperatorGraph
from scheduling_engine import SchedulingEngine
from cost_model import ScheduleCostModel
from memory_hierarchy import MemoryHierarchy

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


def compute_schedule_cost(order, graph, cost_model):
    """Compute the schedule cost using the real cost model."""
    try:
        cost = cost_model.evaluate_schedule(order)
        return cost
    except Exception as e:
        return float('nan')


SIZES = [16, 32, 64]
penalty_set = {
    "dram_pressure": 1.0,
    "sram_capacity": 1.0,
    "bandwidth_capacity": 1.0,
}

rows = []
print("\nCCE-QOS Scaling Benchmark (Real Scheduler + Cost Model)")
print(f"{'Nodes':>6} | {'Greedy':>10} | {'SA':>10} | {'SA %imp':>8} | {'Lookahead':>10} | {'LA %imp':>8} | {'LA time':>8}")
print("-" * 75)

for n in SIZES:
    graph = make_synthetic_graph(n, seed=42)
    scheduler = SchedulingEngine(graph, random_seed=42)
    
    # Load hardware config and cost model
    hw_config = {
        "sram_capacity": 720.0,
        "sram_banks": 6,
        "max_bytes_per_cycle": 6.6,
        "read_bytes_per_cycle": 6.4,
        "write_bytes_per_cycle": 6.9,
        "bandwidth_window": 4,
        "backlog_decay": 0.66,
        "prefetch_slots": 3,
        "bank_conflict_factor": 0.11,
        "eviction_idle_factor": 0.038,
        "write_back_factor": 1.0,
        "prefetch_ratio": 0.22,
        "stall_factor": 1.25,
    }
    mem_hier = MemoryHierarchy(hw_config)
    cost_model = ScheduleCostModel(graph, mem_hier)
    
    # Greedy
    t0 = time.perf_counter()
    greedy_result = scheduler.greedy(penalty_set)
    g_cost = compute_schedule_cost(greedy_result.order, graph, cost_model)
    g_time = time.perf_counter() - t0
    
    # SA
    t0 = time.perf_counter()
    sa_result = scheduler.simulated_annealing(penalty_set, iterations=500, temperature=100.0)
    sa_cost = compute_schedule_cost(sa_result.order, graph, cost_model)
    sa_time = time.perf_counter() - t0
    
    # Lookahead (depth 2)
    t0 = time.perf_counter()
    la_result = scheduler.lookahead(
        penalty_set,
        lookahead_depth=2,
        evaluator=lambda order: compute_schedule_cost(order, graph, cost_model)
    )
    la_cost = compute_schedule_cost(la_result.order, graph, cost_model)
    la_time = time.perf_counter() - t0
    
    # Compute improvements
    sa_imp = (g_cost - sa_cost) / g_cost * 100 if g_cost > 0 else 0.0
    la_imp = (g_cost - la_cost) / g_cost * 100 if g_cost > 0 else 0.0
    
    print(f"{n:>6} | {g_cost:>10.1f} | {sa_cost:>10.1f} | {sa_imp:>7.1f}% | "
          f"{la_cost:>10.1f} | {la_imp:>7.1f}% | {la_time*1000:>6.0f}ms")
    
    rows.append({
        "n_nodes": n,
        "greedy_cost": round(g_cost, 2),
        "greedy_time_ms": round(g_time * 1000, 1),
        "sa_cost": round(sa_cost, 2),
        "sa_time_ms": round(sa_time * 1000, 1),
        "sa_improvement_pct": round(sa_imp, 2),
        "lookahead_cost": round(la_cost, 2),
        "lookahead_time_ms": round(la_time * 1000, 1),
        "lookahead_improvement_pct": round(la_imp, 2),
    })

print()
print("Cost model: ScheduleCostModel (DRAM, SRAM, bandwidth, bank conflicts)")
print("SA: 500 iterations, T0=100")
print("Lookahead: depth=2, 4 rollout trials per prefix")

out = pathlib.Path(r"C:\Research\CCE-QOS\results\dag_scaling")
out.mkdir(parents=True, exist_ok=True)
(out / "real_scheduler_scaling.json").write_text(json.dumps({
    "description": "CCE-QOS solver scaling using real SchedulingEngine and ScheduleCostModel",
    "class": "BENCHMARK (Python scheduler model, synthetic NPU DAGs, real cost function)",
    "results": rows,
}, indent=2))
print(f"\nSaved: {out}/real_scheduler_scaling.json")
