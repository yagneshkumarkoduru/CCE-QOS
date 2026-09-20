"""Seeded synthetic workload generator for CCE-QOS multi-workload sweeps.

Generates layered neural DAGs in the exact schema of example_workload.json
(id, type, compute_cycles, input_size, output_size, dependencies, attrs) so
the standard pipeline consumes them unmodified. Topology, operator mix,
sizes, and attributes are drawn from a seeded numpy generator, which makes
every workload reproducible from (num_nodes, seed) alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Operator mix approximating the 31-node reference workload (dense/conv/
# matmul dominate compute; norm/relu/add are light; pool/softmax rare).
OPERATOR_MIX = [
    ("dense", 9),
    ("conv", 4),
    ("add", 4),
    ("norm", 3),
    ("relu", 3),
    ("matmul", 3),
    ("pool", 2),
    ("batchnorm", 1),
    ("softmax", 1),
    ("gelu", 1),
]
COMPUTE_RANGES = {
    "dense": (90, 200),
    "conv": (140, 260),
    "matmul": (100, 220),
    "pool": (40, 90),
    "norm": (28, 60),
    "batchnorm": (28, 60),
    "relu": (16, 32),
    "gelu": (20, 42),
    "add": (14, 34),
    "softmax": (22, 52),
}
MIN_SIZE = 64.0
START_SIZE = 480.0


def generate_workload(num_nodes: int, seed: int) -> Dict[str, Any]:
    """Return one layered-DAG workload dict; deterministic in (num_nodes, seed)."""

    if num_nodes < 8:
        raise ValueError("num_nodes must be >= 8")
    rng = np.random.default_rng(seed)
    # The reference workload is a mostly sequential backbone: 14 levels for
    # 31 nodes (25 single-parent nodes, one input, four joins).
    depth = max(6, int(round(num_nodes * 0.45)))
    inner = depth - 2
    sizes = np.ones(depth, dtype=int)
    remaining = num_nodes - 2
    base = remaining // inner
    sizes[1:-1] = base
    leftovers = remaining - base * inner
    if leftovers:
        for extra in rng.choice(inner, size=leftovers, replace=False):
            sizes[1 + int(extra)] += 1

    types, counts = zip(*OPERATOR_MIX)
    weights = np.array(counts, dtype=float) / float(sum(counts))

    nodes: List[Dict[str, Any]] = []
    layer_nodes: List[List[str]] = []
    size_in = START_SIZE
    for layer in range(depth):
        layer_factor = float(rng.uniform(0.93, 1.0))
        current: List[str] = []
        for idx in range(int(sizes[layer])):
            op = str(rng.choice(types, p=weights))
            out_size = max(MIN_SIZE, round(size_in * layer_factor * float(rng.uniform(0.95, 1.02)), 1))
            node_id = f"{op}_{layer}_{idx}"
            deps: List[str] = []
            if layer > 0:
                parents = layer_nodes[layer - 1]
                if len(parents) == 1 or float(rng.uniform()) < 0.8:
                    k = 1
                else:
                    k = 2
                k = min(k, len(parents))
                chosen = np.atleast_1d(rng.choice(len(parents), size=k, replace=False))
                deps = [parents[int(c)] for c in chosen]
            lo, hi = COMPUTE_RANGES[op]
            nodes.append(
                {
                    "id": node_id,
                    "type": op,
                    "compute_cycles": int(rng.integers(lo, hi + 1)),
                    "input_size": float(size_in),
                    "output_size": out_size,
                    "dependencies": deps,
                    "attrs": {
                        "criticality": round(float(rng.uniform(1.0, 1.35)), 2),
                        "sram_hint": round(size_in * 0.66, 1),
                        "volatility": round(float(rng.uniform(0.04, 0.12)), 2),
                    },
                }
            )
            current.append(node_id)
        layer_nodes.append(current)
        size_in = max(MIN_SIZE, round(size_in * layer_factor, 1))

    return {"name": f"synthetic_{num_nodes}n_seed{seed}", "nodes": nodes}


def perturb_workload(reference: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Seeded parameter perturbation of the reference DAG (topology unchanged).

    Sizes and compute cycles vary by +/-15-20 percent, attribute hints by
    +/-5-10 percent, so the family stays near the feasible design point while
    still exercising the scheduler's sensitivity to instance parameters.
    """

    rng = np.random.default_rng(seed)
    nodes: List[Dict[str, Any]] = []
    for node in reference["nodes"]:
        compute = max(8, int(round(float(node["compute_cycles"]) * float(rng.uniform(0.80, 1.20)))))
        input_size = max(32.0, round(float(node["input_size"]) * float(rng.uniform(0.85, 1.15)), 1))
        output_size = max(32.0, round(float(node["output_size"]) * float(rng.uniform(0.85, 1.15)), 1))
        attrs = node.get("attrs", {})
        nodes.append(
            {
                "id": node["id"],
                "type": node["type"],
                "compute_cycles": compute,
                "input_size": input_size,
                "output_size": output_size,
                "dependencies": list(node.get("dependencies", [])),
                "attrs": {
                    "criticality": round(float(attrs.get("criticality", 1.0)) * float(rng.uniform(0.95, 1.05)), 2),
                    "sram_hint": round(float(attrs.get("sram_hint", input_size * 0.66)) * float(rng.uniform(0.90, 1.10)), 1),
                    "volatility": round(min(0.25, max(0.01, float(attrs.get("volatility", 0.05)) + float(rng.uniform(-0.02, 0.02)))), 2),
                },
            }
        )
    return {"name": f"perturbed_{reference.get('name', 'reference')}_seed{seed}", "nodes": nodes}


def write_workload(path: Path, num_nodes: int, seed: int) -> Dict[str, Any]:
    workload = generate_workload(num_nodes, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return workload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one synthetic CCE-QOS workload")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli_args = parser.parse_args()
    written = write_workload(cli_args.out, cli_args.nodes, cli_args.seed)
    print(f"wrote {cli_args.out} ({len(written['nodes'])} nodes)")
