"""Realistic NPU operator-graph generator for CCE-QOS benchmarks.

Generates workload JSON files matching the pipeline schema (id, type,
compute_cycles, input_size, output_size, dependencies, attrs) from actual
neural network architecture patterns: ResNet-50 bottleneck blocks,
MobileNet-V2 inverted residuals, and BERT transformer layers.

These workloads expose realistic bank-contention, fusion, and dependency
patterns that random DAGs cannot reproduce, making scheduler comparisons
meaningful for the CCE-QOS novelty claim (first QUBO formulation for NPU
scheduling on real architecture patterns).

Usage:
    python benchmarks/realistic_workload_extractor.py
    # writes JSON files to benchmarks/realistic_workloads/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# -- cycle cost estimators (NPU-scaled, proportional to real FLOP ratios) ---

def _conv3x3_cycles(in_ch: int, out_ch: int, spatial: int) -> int:
    """3x3 convolution: O(in*out*9*H*W), scaled to [150, 500]."""
    raw = in_ch * out_ch * 9 * spatial * spatial
    return max(150, min(500, int(raw / 2e6)))

def _conv1x1_cycles(in_ch: int, out_ch: int, spatial: int) -> int:
    """1x1 convolution: O(in*out*H*W), scaled to [60, 250]."""
    raw = in_ch * out_ch * spatial * spatial
    return max(60, min(250, int(raw / 2e6)))

def _dwconv_cycles(channels: int, spatial: int) -> int:
    """Depthwise 3x3: O(ch*9*H*W), scaled to [30, 120]."""
    raw = channels * 9 * spatial * spatial
    return max(30, min(120, int(raw / 5e5)))

def _bn_cycles(channels: int, spatial: int) -> int:
    """BatchNorm/LayerNorm: O(ch*H*W), scaled to [20, 80]."""
    raw = channels * spatial * spatial
    return max(20, min(80, int(raw / 5e5)))

def _relu_cycles(channels: int, spatial: int) -> int:
    """ReLU/GeLU: O(ch*H*W), scaled to [10, 40]."""
    raw = channels * spatial * spatial
    return max(10, min(40, int(raw / 1e6)))

def _matmul_cycles(m: int, k: int, n: int) -> int:
    """Matrix multiply: O(m*k*n), scaled to [200, 600]."""
    raw = m * k * n
    return max(200, min(600, int(raw / 5e6)))

def _pool_cycles(channels: int, spatial: int) -> int:
    return max(15, min(50, int(channels * spatial * spatial / 1e6)))

def _add_cycles(channels: int, spatial: int) -> int:
    return max(10, min(30, int(channels * spatial * spatial / 2e6)))

def _tensor_kb(channels: int, spatial: int) -> float:
    """Activation tensor size in KB (float16)."""
    return round(channels * spatial * spatial * 2 / 1024, 1)

def _node(nid: str, ntype: str, cycles: int, in_kb: float, out_kb: float,
          deps: list[str], mem_kb: float = 0.0, bank: int = 0) -> dict[str, Any]:
    return {
        "id": nid,
        "type": ntype,
        "compute_cycles": cycles,
        "input_size": in_kb,
        "output_size": out_kb,
        "dependencies": deps,
        "attrs": {"memory_kb": mem_kb or round(in_kb * 0.1, 1), "bank_preference": bank},
    }


# -- ResNet-50 bottleneck block --------------------------------------------

def generate_resnet50_block(
    block_idx: int = 0,
    in_ch: int = 256,
    inner_ch: int = 64,
    out_ch: int = 256,
    stride: int = 1,
    spatial: int = 56,
    prefix: str = "res",
) -> dict[str, Any]:
    """One ResNet-50 bottleneck: conv1x1->BN->ReLU->conv3x3->BN->ReLU->conv1x1->BN->add->ReLU."""
    p = f"{prefix}_b{block_idx}"
    sp_out = spatial // stride
    nodes: list[dict] = []
    prev = []  # empty = block input

    # Main path
    nid = f"{p}_conv1"
    nodes.append(_node(nid, "conv", _conv1x1_cycles(in_ch, inner_ch, spatial),
                        _tensor_kb(in_ch, spatial), _tensor_kb(inner_ch, spatial), prev, bank=0))
    prev = [nid]
    nid = f"{p}_bn1"
    nodes.append(_node(nid, "norm", _bn_cycles(inner_ch, spatial),
                        _tensor_kb(inner_ch, spatial), _tensor_kb(inner_ch, spatial), prev, bank=0))
    prev = [nid]
    nid = f"{p}_relu1"
    nodes.append(_node(nid, "relu", _relu_cycles(inner_ch, spatial),
                        _tensor_kb(inner_ch, spatial), _tensor_kb(inner_ch, spatial), prev, bank=0))
    prev = [nid]

    nid = f"{p}_conv2"
    nodes.append(_node(nid, "conv", _conv3x3_cycles(inner_ch, inner_ch, spatial),
                        _tensor_kb(inner_ch, spatial), _tensor_kb(inner_ch, sp_out), prev, bank=1))
    prev = [nid]
    nid = f"{p}_bn2"
    nodes.append(_node(nid, "norm", _bn_cycles(inner_ch, sp_out),
                        _tensor_kb(inner_ch, sp_out), _tensor_kb(inner_ch, sp_out), prev, bank=1))
    prev = [nid]
    nid = f"{p}_relu2"
    nodes.append(_node(nid, "relu", _relu_cycles(inner_ch, sp_out),
                        _tensor_kb(inner_ch, sp_out), _tensor_kb(inner_ch, sp_out), prev, bank=1))
    prev = [nid]

    nid = f"{p}_conv3"
    nodes.append(_node(nid, "conv", _conv1x1_cycles(inner_ch, out_ch, sp_out),
                        _tensor_kb(inner_ch, sp_out), _tensor_kb(out_ch, sp_out), prev, bank=0))
    main_out = nid

    nid = f"{p}_bn3"
    nodes.append(_node(nid, "norm", _bn_cycles(out_ch, sp_out),
                        _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out), [main_out], bank=0))
    main_out = nid

    # Shortcut (projection if dims change)
    if stride != 1 or in_ch != out_ch:
        sc = f"{p}_sc_conv"
        nodes.append(_node(sc, "conv", _conv1x1_cycles(in_ch, out_ch, spatial),
                            _tensor_kb(in_ch, spatial), _tensor_kb(out_ch, sp_out), [], bank=2))
        sc_bn = f"{p}_sc_bn"
        nodes.append(_node(sc_bn, "norm", _bn_cycles(out_ch, sp_out),
                            _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out), [sc], bank=2))
        shortcut = sc_bn
    else:
        shortcut = None

    # Add + ReLU
    add_deps = [main_out] + ([shortcut] if shortcut else [])
    nid = f"{p}_add"
    nodes.append(_node(nid, "add", _add_cycles(out_ch, sp_out),
                        _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out), add_deps, bank=0))
    nid_relu = f"{p}_relu3"
    nodes.append(_node(nid_relu, "relu", _relu_cycles(out_ch, sp_out),
                        _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out), [nid], bank=0))

    return {"nodes": nodes}


def generate_resnet50_full() -> dict[str, Any]:
    """Full ResNet-50 inference graph: stem + 4 layers (3+4+6+3 blocks) = ~50 ops."""
    nodes: list[dict] = []

    # Stem: conv7x7 + BN + ReLU + pool
    nodes.append(_node("stem_conv", "conv", 350, 301.1, _tensor_kb(64, 112), [], bank=0))
    nodes.append(_node("stem_bn", "norm", 40, _tensor_kb(64, 112), _tensor_kb(64, 112), ["stem_conv"], bank=0))
    nodes.append(_node("stem_relu", "relu", 20, _tensor_kb(64, 112), _tensor_kb(64, 112), ["stem_bn"], bank=0))
    nodes.append(_node("stem_pool", "pool", 30, _tensor_kb(64, 112), _tensor_kb(64, 56), ["stem_relu"], bank=0))

    # Layer configs: (n_blocks, in_ch, inner, out_ch, stride_first, spatial_in)
    layers = [
        (3, 64, 64, 256, 1, 56),     # layer1: 56x56
        (4, 256, 128, 512, 2, 56),    # layer2: 56->28
        (6, 512, 256, 1024, 2, 28),   # layer3: 28->14
        (3, 1024, 512, 2048, 2, 14),  # layer4: 14->7
    ]

    prev_layer_out = "stem_pool"
    for li, (n_blocks, in_ch, inner, out_ch, stride0, sp) in enumerate(layers):
        sp_cur = sp
        ch_cur = in_ch
        for bi in range(n_blocks):
            stride = stride0 if bi == 0 else 1
            block = generate_resnet50_block(
                block_idx=bi, in_ch=ch_cur, inner_ch=inner, out_ch=out_ch,
                stride=stride, spatial=sp_cur, prefix=f"l{li+1}",
            )
            # Wire first node's deps to previous layer output
            if block["nodes"][0]["dependencies"] == []:
                block["nodes"][0]["dependencies"] = [prev_layer_out]
            # Wire shortcut input (if projection block, shortcut conv has no deps)
            for n in block["nodes"]:
                if n["id"].endswith("_sc_conv") and n["dependencies"] == []:
                    n["dependencies"] = [prev_layer_out]
            nodes.extend(block["nodes"])
            prev_layer_out = block["nodes"][-1]["id"]
            sp_cur = sp_cur // stride
            ch_cur = out_ch

    # Global average pool + FC
    nodes.append(_node("gap", "pool", _pool_cycles(2048, 7),
                        _tensor_kb(2048, 7), _tensor_kb(2048, 1), [prev_layer_out], bank=0))
    nodes.append(_node("fc", "dense", 200,
                        _tensor_kb(2048, 1), 2.0, ["gap"], bank=0))

    return {"nodes": nodes}


# -- MobileNet-V2 inverted residual ----------------------------------------

def generate_mobilenetv2_block(
    block_idx: int = 0,
    in_ch: int = 32,
    out_ch: int = 32,
    expand_ratio: int = 6,
    stride: int = 1,
    spatial: int = 56,
    prefix: str = "mb",
) -> dict[str, Any]:
    """One MobileNet-V2 inverted residual block."""
    p = f"{prefix}_b{block_idx}"
    exp_ch = in_ch * expand_ratio
    sp_out = spatial // stride
    nodes: list[dict] = []

    # Expand (pointwise conv, skip if expand_ratio == 1)
    if expand_ratio != 1:
        nodes.append(_node(f"{p}_exp_conv", "conv", _conv1x1_cycles(in_ch, exp_ch, spatial),
                            _tensor_kb(in_ch, spatial), _tensor_kb(exp_ch, spatial), [], bank=0))
        nodes.append(_node(f"{p}_exp_bn", "norm", _bn_cycles(exp_ch, spatial),
                            _tensor_kb(exp_ch, spatial), _tensor_kb(exp_ch, spatial), [f"{p}_exp_conv"], bank=0))
        nodes.append(_node(f"{p}_exp_relu", "relu", _relu_cycles(exp_ch, spatial),
                            _tensor_kb(exp_ch, spatial), _tensor_kb(exp_ch, spatial), [f"{p}_exp_bn"], bank=0))
        prev = [f"{p}_exp_relu"]
    else:
        prev = []

    # Depthwise conv
    nodes.append(_node(f"{p}_dw_conv", "conv", _dwconv_cycles(exp_ch, spatial),
                        _tensor_kb(exp_ch, spatial), _tensor_kb(exp_ch, sp_out), prev, bank=1))
    nodes.append(_node(f"{p}_dw_bn", "norm", _bn_cycles(exp_ch, sp_out),
                        _tensor_kb(exp_ch, sp_out), _tensor_kb(exp_ch, sp_out), [f"{p}_dw_conv"], bank=1))
    nodes.append(_node(f"{p}_dw_relu", "relu", _relu_cycles(exp_ch, sp_out),
                        _tensor_kb(exp_ch, sp_out), _tensor_kb(exp_ch, sp_out), [f"{p}_dw_bn"], bank=1))

    # Project (pointwise conv, no activation)
    nodes.append(_node(f"{p}_proj_conv", "conv", _conv1x1_cycles(exp_ch, out_ch, sp_out),
                        _tensor_kb(exp_ch, sp_out), _tensor_kb(out_ch, sp_out), [f"{p}_dw_relu"], bank=0))
    nodes.append(_node(f"{p}_proj_bn", "norm", _bn_cycles(out_ch, sp_out),
                        _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out), [f"{p}_proj_conv"], bank=0))

    # Residual add (only when stride==1 and in_ch==out_ch)
    if stride == 1 and in_ch == out_ch:
        nodes.append(_node(f"{p}_add", "add", _add_cycles(out_ch, sp_out),
                            _tensor_kb(out_ch, sp_out), _tensor_kb(out_ch, sp_out),
                            [f"{p}_proj_bn"], bank=0))

    return {"nodes": nodes}


def generate_mobilenetv2_full() -> dict[str, Any]:
    """Full MobileNet-V2 inference graph: ~52 operators."""
    nodes: list[dict] = []

    # Stem: conv3x3 stride 2
    nodes.append(_node("stem_conv", "conv", 120, 301.1, _tensor_kb(32, 112), [], bank=0))
    nodes.append(_node("stem_bn", "norm", 30, _tensor_kb(32, 112), _tensor_kb(32, 112), ["stem_conv"], bank=0))
    nodes.append(_node("stem_relu", "relu", 15, _tensor_kb(32, 112), _tensor_kb(32, 112), ["stem_bn"], bank=0))

    # Layer configs: (n_blocks, in_ch, out_ch, expand, stride_first, spatial_in)
    layers = [
        (1, 32, 16, 1, 1, 112),      # layer1
        (2, 16, 24, 6, 2, 112),       # layer2: 112->56
        (3, 24, 32, 6, 2, 56),        # layer3: 56->28
        (4, 32, 64, 6, 2, 28),        # layer4: 28->14
        (3, 64, 96, 6, 1, 14),        # layer5
        (3, 96, 160, 6, 2, 14),       # layer6: 14->7
        (1, 160, 320, 6, 1, 7),       # layer7
    ]

    prev_out = "stem_relu"
    for li, (n_blocks, in_ch, out_ch, expand, stride0, sp) in enumerate(layers):
        sp_cur = sp
        ch_cur = in_ch
        for bi in range(n_blocks):
            stride = stride0 if bi == 0 else 1
            ch_out = out_ch if bi == 0 else out_ch  # all blocks in layer output same ch
            block = generate_mobilenetv2_block(
                block_idx=bi, in_ch=ch_cur if bi == 0 else out_ch, out_ch=out_ch,
                expand_ratio=expand, stride=stride, spatial=sp_cur,
                prefix=f"l{li+1}",
            )
            # Wire first node deps
            if block["nodes"][0]["dependencies"] == []:
                block["nodes"][0]["dependencies"] = [prev_out]
            # Wire residual add input dep (if present)
            for n in block["nodes"]:
                if n["id"].endswith("_add") and len(n["dependencies"]) == 1:
                    n["dependencies"].append(prev_out)  # add skip connection
            nodes.extend(block["nodes"])
            prev_out = block["nodes"][-1]["id"]
            sp_cur = sp_cur // stride
            ch_cur = out_ch

    # Tail: conv1x1 + pool + fc
    nodes.append(_node("tail_conv", "conv", _conv1x1_cycles(320, 1280, 7),
                        _tensor_kb(320, 7), _tensor_kb(1280, 7), [prev_out], bank=0))
    nodes.append(_node("tail_bn", "norm", _bn_cycles(1280, 7),
                        _tensor_kb(1280, 7), _tensor_kb(1280, 7), ["tail_conv"], bank=0))
    nodes.append(_node("tail_relu", "relu", _relu_cycles(1280, 7),
                        _tensor_kb(1280, 7), _tensor_kb(1280, 7), ["tail_bn"], bank=0))
    nodes.append(_node("gap", "pool", 20, _tensor_kb(1280, 7), 2.5, ["tail_relu"], bank=0))
    nodes.append(_node("fc", "dense", 160, 2.5, 2.0, ["gap"], bank=0))

    return {"nodes": nodes}


# -- BERT transformer layer -------------------------------------------------

def generate_bert_layer(
    layer_idx: int = 0,
    hidden: int = 768,
    seq_len: int = 512,
    num_heads: int = 12,
    prefix: str = "bert",
) -> dict[str, Any]:
    """One BERT transformer layer: ~15 operators."""
    p = f"{prefix}_L{layer_idx}"
    tok_kb = round(seq_len * hidden * 2 / 1024, 1)  # float16
    head_dim = hidden // num_heads
    attn_kb = round(seq_len * seq_len * num_heads * 2 / 1024, 1)
    ffn_kb = round(seq_len * hidden * 4 * 2 / 1024, 1)
    nodes: list[dict] = []

    # QKV projections (parallel)
    for name in ("q_proj", "k_proj", "v_proj"):
        nodes.append(_node(f"{p}_{name}", "matmul",
                            _matmul_cycles(seq_len, hidden, hidden),
                            tok_kb, tok_kb, [], bank=0))

    # Scaled dot-product attention: Q @ K^T
    nodes.append(_node(f"{p}_attn_score", "matmul",
                        _matmul_cycles(seq_len, head_dim, seq_len),
                        tok_kb, attn_kb,
                        [f"{p}_q_proj", f"{p}_k_proj"], bank=1))

    # Softmax
    nodes.append(_node(f"{p}_softmax", "softmax", max(50, min(120, seq_len * num_heads // 50)),
                        attn_kb, attn_kb, [f"{p}_attn_score"], bank=1))

    # Weighted sum: attn @ V
    nodes.append(_node(f"{p}_attn_out", "matmul",
                        _matmul_cycles(seq_len, seq_len, head_dim),
                        attn_kb, tok_kb,
                        [f"{p}_softmax", f"{p}_v_proj"], bank=1))

    # Output projection
    nodes.append(_node(f"{p}_out_proj", "matmul",
                        _matmul_cycles(seq_len, hidden, hidden),
                        tok_kb, tok_kb, [f"{p}_attn_out"], bank=0))

    # Residual + LayerNorm (post-LN)
    nodes.append(_node(f"{p}_add1", "add", max(15, seq_len * hidden // 200000),
                        tok_kb, tok_kb, [f"{p}_out_proj"], bank=0))
    nodes.append(_node(f"{p}_ln1", "norm", max(30, seq_len * hidden // 100000),
                        tok_kb, tok_kb, [f"{p}_add1"], bank=0))

    # FFN: linear1 -> GeLU -> linear2
    nodes.append(_node(f"{p}_ffn1", "dense",
                        _matmul_cycles(seq_len, hidden, hidden * 4),
                        tok_kb, ffn_kb, [f"{p}_ln1"], bank=0))
    nodes.append(_node(f"{p}_gelu", "gelu", max(15, seq_len * hidden * 4 // 500000),
                        ffn_kb, ffn_kb, [f"{p}_ffn1"], bank=0))
    nodes.append(_node(f"{p}_ffn2", "dense",
                        _matmul_cycles(seq_len, hidden * 4, hidden),
                        ffn_kb, tok_kb, [f"{p}_gelu"], bank=0))

    # Residual + LayerNorm
    nodes.append(_node(f"{p}_add2", "add", max(15, seq_len * hidden // 200000),
                        tok_kb, tok_kb, [f"{p}_ffn2", f"{p}_ln1"], bank=0))
    nodes.append(_node(f"{p}_ln2", "norm", max(30, seq_len * hidden // 100000),
                        tok_kb, tok_kb, [f"{p}_add2"], bank=0))

    return {"nodes": nodes}


def generate_bert_full(n_layers: int = 12) -> dict[str, Any]:
    """Full BERT inference graph: 12 layers x 15 ops = ~180 operators."""
    nodes: list[dict] = []

    # Embedding (simplified as one dense layer)
    tok_kb = round(512 * 768 * 2 / 1024, 1)
    nodes.append(_node("embed", "dense", 300, tok_kb, tok_kb, [], bank=0))

    prev_out = "embed"
    for i in range(n_layers):
        layer = generate_bert_layer(layer_idx=i)
        # Wire input deps
        for n in layer["nodes"]:
            if n["dependencies"] == []:
                n["dependencies"] = [prev_out]
            # Wire add1 residual skip
            if n["id"].endswith("_add1") and len(n["dependencies"]) == 1:
                n["dependencies"].append(prev_out)
        nodes.extend(layer["nodes"])
        prev_out = layer["nodes"][-1]["id"]

    # Pooler
    nodes.append(_node("pooler", "dense", 200, tok_kb, round(768 * 2 / 1024, 1), [prev_out], bank=0))

    return {"nodes": nodes}


# -- mid-scale workloads (30-50 nodes, sweet spot for QUBO) ----------------

def generate_resnet50_stage(n_blocks: int = 3) -> dict[str, Any]:
    """One ResNet-50 stage: stem + n_blocks bottlenecks. ~34-46 nodes."""
    nodes: list[dict] = []
    nodes.append(_node("stem_conv", "conv", 350, 301.1, _tensor_kb(64, 56), [], bank=0))
    nodes.append(_node("stem_bn", "norm", 40, _tensor_kb(64, 56), _tensor_kb(64, 56), ["stem_conv"], bank=0))
    nodes.append(_node("stem_relu", "relu", 20, _tensor_kb(64, 56), _tensor_kb(64, 56), ["stem_bn"], bank=0))
    prev = "stem_relu"
    ch = 64
    for bi in range(n_blocks):
        stride = 1 if bi > 0 else 1
        out_ch = 256
        block = generate_resnet50_block(
            block_idx=bi, in_ch=ch, inner_ch=64, out_ch=out_ch,
            stride=stride, spatial=56, prefix="s1",
        )
        if block["nodes"][0]["dependencies"] == []:
            block["nodes"][0]["dependencies"] = [prev]
        for n in block["nodes"]:
            if n["id"].endswith("_sc_conv") and n["dependencies"] == []:
                n["dependencies"] = [prev]
        nodes.extend(block["nodes"])
        prev = block["nodes"][-1]["id"]
        ch = out_ch
    return {"nodes": nodes}


def generate_bert_2layer() -> dict[str, Any]:
    """Two BERT layers: ~30 operators - good QUBO size with parallelism."""
    nodes: list[dict] = []
    tok_kb = round(512 * 768 * 2 / 1024, 1)
    nodes.append(_node("embed", "dense", 300, tok_kb, tok_kb, [], bank=0))
    prev = "embed"
    for i in range(2):
        layer = generate_bert_layer(layer_idx=i)
        for n in layer["nodes"]:
            if n["dependencies"] == []:
                n["dependencies"] = [prev]
            if n["id"].endswith("_add1") and len(n["dependencies"]) == 1:
                n["dependencies"].append(prev)
        nodes.extend(layer["nodes"])
        prev = layer["nodes"][-1]["id"]
    return {"nodes": nodes}


# -- save all workloads ----------------------------------------------------

def save_all_workloads(output_dir: Path | str) -> list[Path]:
    """Generate and save all workload JSON files. Returns list of paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = {
        "resnet50_block": generate_resnet50_block(),
        "resnet50_full": generate_resnet50_full(),
        "resnet50_stage": generate_resnet50_stage(),
        "mobilenetv2_block": generate_mobilenetv2_block(),
        "mobilenetv2_full": generate_mobilenetv2_full(),
        "bert_layer": generate_bert_layer(),
        "bert_2layer": generate_bert_2layer(),
        "bert_full": generate_bert_full(),
    }

    paths: list[Path] = []
    for name, wl in workloads.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(wl, indent=2) + "\n", encoding="utf-8")
        n = len(wl["nodes"])
        print(f"  {name}: {n} operators -> {path}")
        paths.append(path)

    return paths


# -- validation ------------------------------------------------------------

def validate_workload(wl: dict[str, Any]) -> None:
    """Check schema, unique IDs, valid deps, positive cycles, no cycles."""
    ids_seen: set[str] = set()
    for node in wl["nodes"]:
        assert node["id"] not in ids_seen, f"duplicate id: {node['id']}"
        ids_seen.add(node["id"])
        assert node["compute_cycles"] > 0, f"non-positive cycles: {node['id']}"
        for dep in node["dependencies"]:
            assert dep in ids_seen, f"forward dep: {node['id']} depends on unseen {dep}"


if __name__ == "__main__":
    out = Path(__file__).parent / "realistic_workloads"
    print("Generating realistic NPU workloads:")
    paths = save_all_workloads(out)
    print(f"\nValidating {len(paths)} workloads...")
    for p in paths:
        wl = json.loads(p.read_text())
        validate_workload(wl)
        print(f"  {p.name}: OK ({len(wl['nodes'])} nodes)")
    print("All workloads valid.")
