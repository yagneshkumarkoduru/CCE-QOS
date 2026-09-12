import json, pathlib, random

def make_workload(n_nodes, seed=42):
    rng = random.Random(seed)
    node_types = ['conv', 'norm', 'act', 'pool', 'fc', 'attn', 'ffn']
    nodes = []
    ids = [f'op_{i}' for i in range(n_nodes)]
    for i, nid in enumerate(ids):
        max_deps = min(3, i)
        n_deps = rng.randint(0, max_deps)
        deps = rng.sample(ids[:i], n_deps) if n_deps > 0 else []
        nodes.append({
            'id': nid,
            'type': rng.choice(node_types),
            'compute_cycles': rng.randint(10, 300),
            'input_size': round(rng.uniform(100, 800), 1),
            'output_size': round(rng.uniform(100, 800), 1),
            'dependencies': deps,
            'attrs': {
                'criticality': round(rng.uniform(0.5, 2.0), 2),
                'sram_hint': round(rng.uniform(100, 500), 1),
                'volatility': round(rng.uniform(0.01, 0.2), 2),
            }
        })
    return {'name': f'synthetic_{n_nodes}node', 'nodes': nodes}

wl = make_workload(48, seed=42)
out = pathlib.Path('benchmarks/workload_48nodes.json')
out.write_text(json.dumps(wl, indent=2))
print(f'Created: {out} (48 nodes)')

wl64 = make_workload(64, seed=42)
out64 = pathlib.Path('benchmarks/workload_64nodes_new.json')
out64.write_text(json.dumps(wl64, indent=2))
print(f'Created: {out64} (64 nodes)')
