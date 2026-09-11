from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from bandwidth_estimator import BandwidthEstimator
from core_types import HardwareConfig, OperatorGraph
from cost_model import ScheduleCostModel
from fusion_logic import FusionLogic
from graph_builder import load_hardware_config, load_operator_graph
from memory_hierarchy import MemoryHierarchy
from penalty_tuner import PenaltyTuner
from quantum_interface import (
    ProblemSpec,
    build_qubo,
    qubo_energy,
    run_qaoa_statevector,
    run_qaoa_stub,
)
from qubo_types import QUBOData
from schedule_analysis import ScheduleAnalysis
from schedule_explainer import ScheduleExplainer
from scheduling_engine import ScheduleResult, SchedulingEngine


def load_config(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration root must be an object.")
        return parsed
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install PyYAML or keep config JSON-compatible.") from exc
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration root must be an object.")
        return parsed


def _derive_weights(config: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    energy = dict(config.get("energy_weights", {}))
    alpha = dict(config.get("alpha", {}))
    beta = dict(config.get("beta", {}))
    gamma = dict(config.get("gamma", {}))
    return (
        {
            "comp": float(alpha.get("comp", energy.get("unary_cost", 1.0))),
            "energy": float(alpha.get("energy", 1.0)),
            "lat": float(alpha.get("lat", 0.25)),
            "dvfs": float(alpha.get("dvfs", 0.2)),
        },
        {
            "reuse": float(beta.get("reuse", energy.get("data_reuse_reward", 1.0))),
            "fuse": float(beta.get("fuse", energy.get("fusion_reward", 1.0))),
            "bw": float(beta.get("bw", energy.get("bandwidth_spike_penalty", 1.0))),
        },
        {
            "bank": float(gamma.get("bank", energy.get("memory_conflict_penalty", 0.8))),
            "burst": float(gamma.get("burst", 0.6)),
            "stall": float(gamma.get("stall", 0.4)),
            "parallelism": float(gamma.get("parallelism", 0.3)),
        },
    )


def _map_penalties_to_qubo(p: Mapping[str, float]) -> Dict[str, float]:
    return {
        "unique_exec": float(p.get("unique_exec", p.get("dependency_conflict", 1.0) * 1.6)),
        "dep": float(p.get("dep", p.get("dependency_conflict", 1.0))),
        "dvfs_one_hot": float(p.get("dvfs_one_hot", p.get("bandwidth_capacity", 1.0) * 0.6)),
        "mem_cap": float(p.get("mem_cap", p.get("sram_capacity", 1.0))),
        "mem_bind": float(p.get("mem_bind", p.get("sram_capacity", 1.0))),
    }


def _complete_order(order: Sequence[int], graph: OperatorGraph) -> list[int]:
    seen = set()
    candidate = []
    for x in order:
        node_id = int(x)
        if node_id in seen or node_id not in graph.node_by_id:
            continue
        seen.add(node_id)
        candidate.append(node_id)

    final: list[int] = []
    scheduled = set()
    pending = list(candidate)
    while pending:
        changed = False
        rest = []
        for node_id in pending:
            deps = graph.node_by_id[node_id].dependencies
            if all(dep in scheduled for dep in deps):
                final.append(node_id)
                scheduled.add(node_id)
                changed = True
            else:
                rest.append(node_id)
        pending = rest
        if not changed:
            break

    while len(final) < len(graph.nodes):
        ready = [n for n in graph.ready_nodes(scheduled) if n not in scheduled]
        if not ready:
            break
        chosen = ready[0]
        final.append(chosen)
        scheduled.add(chosen)
    return final


class CCEEvaluator:
    def __init__(
        self,
        graph: OperatorGraph,
        hardware: HardwareConfig,
        cost_model: ScheduleCostModel,
        alpha: Dict[str, float],
        beta: Dict[str, float],
        gamma: Dict[str, float],
        base_penalties: Dict[str, float],
    ) -> None:
        self.graph = graph
        self.hardware = hardware
        self.cost_model = cost_model
        self.alpha = dict(alpha)
        self.beta = dict(beta)
        self.gamma = dict(gamma)
        self.base_penalties = dict(base_penalties)
        self._cache: Dict[Tuple[Any, ...], QUBOData] = {}

    def _qubo(self, penalties: Dict[str, float], ablation: str | None) -> QUBOData:
        beta = dict(self.beta)
        gamma = dict(self.gamma)
        if ablation == "remove_pairwise":
            beta = {k: 0.0 for k in beta}
        if ablation == "remove_higher_order":
            gamma = {k: 0.0 for k in gamma}
        key = (
            tuple(sorted(_map_penalties_to_qubo(penalties).items())),
            tuple(sorted(beta.items())),
            tuple(sorted(gamma.items())),
        )
        if key in self._cache:
            return self._cache[key]
        spec = ProblemSpec(
            graph=self.graph,
            hardware=self.hardware,
            alpha=self.alpha,
            beta=beta,
            gamma=gamma,
            penalties=_map_penalties_to_qubo(penalties),
        )
        qubo = build_qubo(spec)
        self._cache[key] = qubo
        return qubo

    def _encode(self, order: Sequence[int], qubo: QUBOData) -> list[int]:
        bits = [0] * qubo.num_variables
        x_lookup: Dict[Tuple[int, int, str], int] = {}
        x_fallback: Dict[int, int] = {}
        m_lookup: Dict[Tuple[int, str], int] = {}
        m_fallback: Dict[int, int] = {}
        f_lookup: Dict[Tuple[int, str], int] = {}
        f_fallback: Dict[int, int] = {}

        for idx, meta in qubo.var_metadata.items():
            kind = meta.get("kind")
            if kind == "x":
                op = int(meta["op"])
                x_lookup[(op, int(meta["t"]), str(meta["r"]))] = idx
                x_fallback.setdefault(op, idx)
            elif kind == "m":
                op = int(meta["op"])
                m_lookup[(op, str(meta["level"]))] = idx
                m_fallback.setdefault(op, idx)
            elif kind == "f":
                t = int(meta["t"])
                f_lookup[(t, str(meta["state"]))] = idx
                f_fallback.setdefault(t, idx)

        primary_r = self.hardware.resources[0] if self.hardware.resources else ""
        levels = [lv.name for lv in self.hardware.memory_levels]
        level = "L2" if "L2" in levels else (levels[0] if levels else "")
        states = [s.name for s in self.hardware.dvfs_states]
        state = "nominal" if "nominal" in states else (states[0] if states else "")

        for pos, op in enumerate(order):
            t = min(pos, self.hardware.max_time_slots - 1)
            x = x_lookup.get((int(op), t, primary_r), x_fallback.get(int(op)))
            m = m_lookup.get((int(op), level), m_fallback.get(int(op)))
            if x is not None:
                bits[x] = 1
            if m is not None:
                bits[m] = 1
        for t in range(self.hardware.max_time_slots):
            f = f_lookup.get((t, state), f_fallback.get(t))
            if f is not None:
                bits[f] = 1
        return bits

    def evaluate(self, order: Sequence[int], penalties: Dict[str, float], ablation: str | None = None) -> Dict[str, Any]:
        valid_order = _complete_order(order, self.graph)
        qubo = self._qubo(penalties, ablation)
        bits = self._encode(valid_order, qubo)
        linear = sum(float(c) * bits[i] for i, c in qubo.linear.items())
        quadratic = sum(float(c) * bits[i] * bits[j] for (i, j), c in qubo.quadratic.items())
        total = float(qubo.constant) + linear + quadratic
        base = self.cost_model.evaluate(self.graph, valid_order, penalties=penalties)
        out = dict(base)
        out["energy_breakdown"] = {
            "unary_cost": linear,
            "pairwise_total": quadratic,
            "constant": float(qubo.constant),
            "total_energy": total,
        }
        out["qubo_snapshot"] = {
            "num_variables": qubo.num_variables,
            "linear_terms": len(qubo.linear),
            "quadratic_terms": len(qubo.quadratic),
            "active_bits": int(sum(bits)),
            "ablation": ablation or "none",
        }
        return out

    def build_qubo_data(self, penalties: Dict[str, float], ablation: str | None = None) -> QUBOData:
        return self._qubo(penalties, ablation)


def _rank(evaluation: Dict[str, Any], objective: float) -> float:
    return objective - 120.0 * float(evaluation.get("feasibility", 0.0))


def _run_trials(
    graph: OperatorGraph,
    trials: int,
    seed: int,
    builder,
    evaluate,
    objective_fn,
) -> Tuple[ScheduleResult, Dict[str, Any]]:
    best_result = None
    best_eval = None
    best_rank = float("inf")
    for idx in range(trials):
        engine = SchedulingEngine(graph, random_seed=seed + idx * 37)
        result = builder(engine)
        order = _complete_order(result.order, graph)
        evaluation = evaluate(order)
        objective = float(objective_fn(evaluation))
        rank = _rank(evaluation, objective)
        if rank < best_rank:
            best_rank = rank
            result.order = order
            result.score = objective
            best_result = result
            best_eval = evaluation
    if best_result is None or best_eval is None:
        raise RuntimeError("No valid schedule found.")
    return best_result, best_eval


def _run_suite(graph, config, penalties, evaluate, objective_fn, seed_offset):
    exp = dict(config.get("experiment", {}))
    search = dict(config.get("search", {}))
    trials = int(exp.get("search_trials", 5))
    seed = int(exp.get("seed", 17)) + seed_offset
    methods = {
        "Greedy": lambda e: e.greedy(penalties=penalties),
        "Lookahead": lambda e: e.lookahead(
            penalties=penalties,
            lookahead_depth=int(search.get("lookahead_depth", 3)),
            evaluator=lambda o: float(objective_fn(evaluate(o))),
        ),
        "Beam Search": lambda e: e.beam_search(
            penalties=penalties,
            beam_width=int(search.get("beam_width", 5)),
            evaluator=lambda o: float(objective_fn(evaluate(o))),
        ),
        "Simulated Annealing": lambda e: e.simulated_annealing(
            penalties=penalties,
            evaluator=lambda o: float(objective_fn(evaluate(o))),
            iterations=int(search.get("annealing_iterations", 240)),
            start_temp=float(search.get("annealing_start_temp", 3.2)),
            end_temp=float(search.get("annealing_end_temp", 0.05)),
        ),
    }
    out = {}
    for i, (name, builder) in enumerate(methods.items()):
        best, ev = _run_trials(graph, trials, seed + i * 97, builder, evaluate, objective_fn)
        out[name] = {"order": best.order, "metadata": best.metadata, "evaluation": ev}
    return out


def _violation_mass(evaluation: Mapping[str, Any]) -> float:
    """Aggregate hard-constraint mass (mirrors the feasibility accounting)."""
    memory = evaluation.get("memory", {}) or {}
    bandwidth = evaluation.get("bandwidth", {}) or {}
    rates = evaluation.get("violation_rate", {}) or {}
    return (
        float((memory.get("violations", {}) or {}).get("sram_capacity", 0.0))
        + float((bandwidth.get("violations", {}) or {}).get("bandwidth_capacity", 0.0))
        + 0.5 * float(rates.get("memory_bank_conflict", 0.0))
    )


def _run_apr_arm(
    graph: OperatorGraph,
    config: Mapping[str, Any],
    penalties: Mapping[str, float],
    cce: CCEEvaluator,
    objective_fn,
) -> Dict[str, Any]:
    """
    Adaptive Penalty Refinement (APR) arm.

    Diagnosis of the earlier regression (see EVIDENCE.md): rounds were compared
    on the QUBO energy evaluated at each round's own penalty multipliers, so
    growing multipliers inflated the recorded objective monotonically
    (160.27 -> 167.81 across five rounds at identical schedules) and the
    round-1 (lowest-penalty) schedule always won the best-candidate selection,
    leaving APR indistinguishable from greedy. The historical cost of 4943.55
    was the same artifact: a true cost near 4169-4300 plus ~650-775 of
    inflated penalty terms recorded with multipliers escalated to 14.

    Fixes implemented here:
    - Canonical candidate ranking: lexicographic (violation mass, energy at
      the base multipliers). Penalty-scale-free, so inflated rounds can no
      longer mask better schedules.
    - Penalty annealing: when a round does not improve the best-known
      violation mass for `anneal_patience` consecutive rounds, multipliers
      decay by `anneal_decay` back toward the configured base values (they
      never drop below base). Escalation itself stays driven by the tuner's
      measured violation rates.
    - Feasibility-preserving polish: a final local descent (swap/insert/
      block-reverse moves) that only accepts moves which strictly reduce the
      violation mass, or keep it unchanged while strictly reducing the
      canonical energy.
    """
    apr_cfg = dict(config.get("apr", {}))
    search_cfg = dict(config.get("search", {}))
    tuner = PenaltyTuner(
        eta1=float(apr_cfg.get("eta1", 0.9)),
        eta2=float(apr_cfg.get("eta2", 0.6)),
        lam_min=float(apr_cfg.get("lam_min", 0.1)),
        lam_max=float(apr_cfg.get("lam_max", 20.0)),
    )
    rounds = max(1, int(apr_cfg.get("rounds", 5)))
    iterations = max(1, int(apr_cfg.get("iterations_per_round", 70)))
    anneal_patience = max(1, int(apr_cfg.get("anneal_patience", 2)))
    anneal_decay = float(apr_cfg.get("anneal_decay", 0.75))
    polish_iterations = max(0, int(apr_cfg.get("polish_iterations", 300)))

    base_penalties = dict(penalties)
    cur_pen = dict(penalties)
    best_key: Tuple[float, float] | None = None
    best_payload: Dict[str, Any] | None = None
    candidates: Dict[Tuple[float, float], List[int]] = {}
    zero_violation_rounds = 0
    trace: list = []

    for rnd in range(rounds):
        engine = SchedulingEngine(graph, random_seed=900 + rnd * 17)
        result = engine.simulated_annealing(
            penalties=cur_pen,
            evaluator=lambda o: float(objective_fn(cce.evaluate(o, penalties=cur_pen))),
            iterations=iterations,
            start_temp=float(search_cfg.get("annealing_start_temp", 3.0)),
            end_temp=float(search_cfg.get("annealing_end_temp", 0.05)),
        )
        order = _complete_order(result.order, graph)
        ev_penalized = cce.evaluate(order, penalties=cur_pen)
        ev_canonical = cce.evaluate(order, penalties=penalties)
        round_violation = _violation_mass(ev_canonical)
        round_objective = float(ev_canonical["energy_breakdown"]["total_energy"])
        key = (round(round_violation, 6), round(round_objective, 6))
        candidates[key] = order
        if best_key is None or key < best_key:
            best_key, best_payload = key, {"order": order, "evaluation": ev_canonical}

        # Anneal once zero violations hold for anneal_patience consecutive
        # rounds; escalation itself stays driven by the tuner's measured
        # violation rates.
        if round_violation <= 1e-9:
            zero_violation_rounds += 1
        else:
            zero_violation_rounds = 0
        cur_pen = tuner.update(cur_pen, ev_penalized.get("violation_rate", {}), ev_penalized.get("cost_impact", {}))
        if zero_violation_rounds >= anneal_patience:
            cur_pen = {
                pen_key: max(float(base_penalties.get(pen_key, tuner.lam_min)), pen_value * anneal_decay)
                for pen_key, pen_value in cur_pen.items()
            }
        trace.append(
            {
                "round": rnd + 1,
                "objective": round_objective,
                "penalized_objective": float(ev_penalized["energy_breakdown"]["total_energy"]),
                "violation_mass": round_violation,
                "penalties": dict(cur_pen),
            }
        )

    assert best_key is not None and best_payload is not None

    # Feasibility-preserving polish: lexicographic descent from each round
    # candidate; the final answer keeps the lexicographic best.
    polish_selected = False
    polished_objective = float(best_payload["evaluation"]["energy_breakdown"]["total_energy"])
    polished_violation = best_key[0]
    starts = sorted(candidates.items())
    for start_idx, (_start_key, start_order) in enumerate(starts):
        polish_engine = SchedulingEngine(graph, random_seed=900 + rounds * 191 + start_idx * 131)
        cand_order, cand_objective, cand_violation = polish_engine.feasibility_preserving_descent(
            start_order,
            lambda o: (lambda e: (float(e["energy_breakdown"]["total_energy"]), _violation_mass(e)))(
                cce.evaluate(o, penalties=penalties)
            ),
            iterations=polish_iterations,
        )
        cand_key = (round(cand_violation, 6), round(cand_objective, 6))
        if cand_key < best_key:
            best_key = cand_key
            best_payload = {"order": cand_order, "evaluation": cce.evaluate(cand_order, penalties=penalties)}
            polished_objective = cand_objective
            polished_violation = cand_violation
            polish_selected = True

    return {
        "order": best_payload["order"],
        "evaluation": best_payload["evaluation"],
        "metadata": {
            "round_trace": trace,
            "final_penalties": cur_pen,
            "polish": {
                "starts": len(starts),
                "iterations": polish_iterations,
                "objective": polished_objective,
                "violation_mass": polished_violation,
                "selected": polish_selected,
            },
        },
    }


def _x_metric(base_eval, cand_eval, weights):
    def improve(old, new):
        return 0.0 if abs(old) < 1e-9 else (old - new) / abs(old)

    base_idle = float(base_eval["memory"].get("idle_cycles", 0.0)) + float(base_eval["bandwidth"].get("pipeline_stalls", 0.0))
    cand_idle = float(cand_eval["memory"].get("idle_cycles", 0.0)) + float(cand_eval["bandwidth"].get("pipeline_stalls", 0.0))
    return (
        float(weights.get("cost", 0.3)) * improve(base_eval["breakdown"]["total_cost"], cand_eval["breakdown"]["total_cost"])
        + float(weights.get("latency", 0.25)) * improve(base_eval["latency_cycles"], cand_eval["latency_cycles"])
        + float(weights.get("dram", 0.25)) * improve(base_eval["memory"].get("dram_access", 0.0), cand_eval["memory"].get("dram_access", 0.0))
        + float(weights.get("stalls", 0.2)) * improve(base_idle, cand_idle)
    )


def _format_results_table(
    baseline_summary: Mapping[str, Mapping[str, Any]],
    cce_summary: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Mapping[str, Any]],
    x_cce: float,
    x_quantum: float,
    quantum_label: str,
    quantum_backend: str,
    generated: str,
) -> str:
    """Compact digest of one pipeline run (same numbers as metrics.txt)."""

    def row(summary: Mapping[str, Any]) -> str:
        return (
            f"{summary['strategy']}: cost {summary['total_cost']:.2f}, "
            f"energy {summary['total_energy']:.2f}, "
            f"latency {summary['latency']:.2f} cycles, "
            f"feasibility {summary['feasibility_percent']:.2f}%"
        )

    greedy = baseline_summary.get("Greedy")
    best_base = min(baseline_summary.values(), key=lambda item: item["total_cost"])
    best_cce = min(cce_summary.values(), key=lambda item: item["total_energy"])
    lines = [
        "# CCE-QOS benchmark results table",
        f"# Current pipeline output, regenerated {generated}.",
        "# Full tables: metrics.txt and outputs/metrics.txt (identical content).",
        "",
        "[Baseline Cost Objective]",
    ]
    if greedy is not None:
        lines.append(f"- Greedy baseline: {row(greedy)}")
    lines.append(f"- Best classical search: {row(best_base)}")
    lines.append("")
    lines.append("[CCE-QUBO Objective (energy formulation)]")
    lines.append(f"- Best classical search on the CCE-QUBO energy: {row(best_cce)}")
    for name in ("CCE + APR", quantum_label):
        if name in extra:
            lines.append(f"- {name}: {row(extra[name])}")
    lines.append("")
    lines.append("[X Metric]")
    lines.append(f"- CCE vs baseline: {x_cce:.4f}")
    lines.append(f"- {quantum_label} vs baseline: {x_quantum:.4f}")
    lines.append(f"- Quantum backend used: {quantum_backend}")
    return "\n".join(lines) + "\n"


def main() -> None:
    start = time.perf_counter()
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    graph = load_operator_graph(root / str(config["input"]["workload"]))
    hardware = load_hardware_config(config.get("hardware", {}), num_nodes=len(graph.nodes))
    penalties = dict(config.get("apr", {}).get("initial_penalties", {}))

    cost_model = ScheduleCostModel(
        memory_hierarchy=MemoryHierarchy(dict(config.get("hardware", {}))),
        bandwidth_estimator=BandwidthEstimator(dict(config.get("hardware", {}))),
        fusion_logic=FusionLogic(dict(config.get("fusion", {}))),
        weights=dict(config.get("cost_weights", {})),
    )
    alpha, beta, gamma = _derive_weights(config)
    cce = CCEEvaluator(graph, hardware, cost_model, alpha, beta, gamma, penalties)

    eval_cost = lambda order: cost_model.evaluate(graph, order, penalties=penalties)
    eval_cce = lambda order: cce.evaluate(order, penalties=penalties)
    obj_cost = lambda ev: float(ev["breakdown"]["total_cost"])
    obj_energy = lambda ev: float(ev["energy_breakdown"]["total_energy"])

    baseline = _run_suite(graph, config, penalties, eval_cost, obj_cost, seed_offset=0)
    cce_suite = _run_suite(graph, config, penalties, eval_cce, obj_energy, seed_offset=400)

    cce_apr = _run_apr_arm(graph, config, penalties, cce, obj_energy)

    qubo_data = cce.build_qubo_data(penalties=penalties)
    q_cfg = dict(config.get("quantum", {}))
    seed = int(config.get("experiment", {}).get("seed", 17)) + 1234
    quantum_backend = "local_search_fallback"
    quantum_candidates: list = []
    if qubo_data.num_variables <= 12:
        # Real statevector QAOA via QAOA_solver.py: only feasible for small
        # problem sizes (2^num_qubits statevector). The arm sweeps circuit
        # depths p = 1..p_max with a COBYLA budget per depth and keeps the
        # depth with the best ground-state approximation ratio. Falls back to
        # the deterministic classical local-search fallback on any failure.
        try:
            quantum_candidates = run_qaoa_statevector(
                qubo_data,
                num_samples=int(q_cfg.get("samples", 48)),
                p_layers=int(q_cfg.get("layers", 2)),
                seed=seed,
                p_sweep=bool(q_cfg.get("p_sweep", True)),
                p_max=int(q_cfg.get("p_max", 3)),
                cobyla_budget=int(q_cfg.get("cobyla_budget", 60)),
            )
            if quantum_candidates:
                quantum_backend = "qaoa_statevector"
        except Exception:
            quantum_candidates = []
            quantum_backend = "local_search_fallback"
    if not quantum_candidates:
        # Large QUBOs (the 31-node example workload has 4247 variables, far
        # beyond a 2^n statevector) use a deterministic classical multi-start
        # bit-flip local search on the QUBO energy, exposed through the same
        # candidate-generation interface. Labeled honestly as a fallback.
        quantum_candidates = run_qaoa_stub(
            qubo_data,
            num_samples=int(q_cfg.get("samples", 48)),
            num_steps=int(q_cfg.get("iterations", 220)),
            seed=seed,
        )
        quantum_backend = "local_search_fallback"

    quantum_label = "Quantum (QAOA)" if quantum_backend == "qaoa_statevector" else "Quantum (local-search fallback)"
    best_q = None
    best_q_obj = float("inf")
    for c in quantum_candidates:
        order = _complete_order(c.get("schedule_projection", []), graph)
        ev = cce.evaluate(order, penalties=penalties)
        obj = obj_energy(ev)
        if obj < best_q_obj:
            best_q_obj = obj
            best_q = {"order": order, "evaluation": ev, "metadata": {"candidate": c, "backend": quantum_backend}}
    quantum_stub = best_q

    ablations = {
        "full_cce": cce_suite["Simulated Annealing"],
        "no_pairwise": {"order": quantum_stub["order"], "evaluation": cce.evaluate(quantum_stub["order"], penalties=penalties, ablation="remove_pairwise"), "metadata": {}},
        "no_higher_order": {"order": quantum_stub["order"], "evaluation": cce.evaluate(quantum_stub["order"], penalties=penalties, ablation="remove_higher_order"), "metadata": {}},
    }

    analysis = ScheduleAnalysis()
    explainer = ScheduleExplainer()
    baseline_summary = {k: analysis.summarize(k, v["evaluation"]) for k, v in baseline.items()}
    cce_summary = {k: analysis.summarize(k, v["evaluation"]) for k, v in cce_suite.items()}
    extra = {
        "CCE + APR": analysis.summarize("CCE + APR", cce_apr["evaluation"]),
        quantum_label: analysis.summarize(quantum_label, quantum_stub["evaluation"]),
    }

    best_base = min(baseline.items(), key=lambda kv: obj_cost(kv[1]["evaluation"]))
    best_cce = min(cce_suite.items(), key=lambda kv: obj_energy(kv[1]["evaluation"]))
    x_weights = dict(config.get("x_metric_weights", {"cost": 0.3, "latency": 0.25, "dram": 0.25, "stalls": 0.2}))
    x_cce = _x_metric(best_base[1]["evaluation"], best_cce[1]["evaluation"], x_weights)
    x_quantum = _x_metric(best_base[1]["evaluation"], quantum_stub["evaluation"], x_weights)

    output_dir = root / str(config.get("output", {}).get("directory", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "workload": {"num_nodes": len(graph.nodes), "num_edges": len(graph.edges)},
        "results": {
            "baseline_cost": baseline,
            "cce_qubo": cce_suite,
            "cce_qubo_apr": cce_apr,
            # Backend is "qaoa_statevector" (real statevector QAOA, small QUBOs
            # only) or "local_search_fallback" (deterministic classical
            # multi-start bit-flip search on the QUBO energy, large QUBOs).
            "quantum_candidates": quantum_stub,
        },
        "ablations": ablations,
        "x_metric": {"weights": x_weights, "cce_vs_baseline": x_cce, "quantum_vs_baseline": x_quantum},
        "runtime_seconds": time.perf_counter() - start,
    }
    (output_dir / "schedules.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics_text = (
        "[Baseline Cost Objective]\n"
        + analysis.comparison_table(baseline_summary)
        + "\n\n[CCE-QUBO Objective]\n"
        + analysis.comparison_table(cce_summary)
        + f"\n\n[APR / {quantum_label}]\n"
        + analysis.comparison_table(extra)
        + "\n\n[X Metric]\n"
        + f"CCE vs baseline: {x_cce:.4f}\n"
        + f"{quantum_label} vs baseline: {x_quantum:.4f}\n"
        + f"Quantum backend used: {quantum_backend}\n"
    )
    (output_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    results_table_text = _format_results_table(
        baseline_summary,
        cce_summary,
        extra,
        x_cce,
        x_quantum,
        quantum_label,
        quantum_backend,
        time.strftime("%Y-%m-%d"),
    )
    explanations = [explainer.explain(name, payload, baseline_summary[name]) for name, payload in baseline.items()]
    explanations.extend(explainer.explain(name, payload, cce_summary[name]) for name, payload in cce_suite.items())
    explanations.append(explainer.explain("CCE + APR", cce_apr, extra["CCE + APR"]))
    explanations.append(explainer.explain(quantum_label, quantum_stub, extra[quantum_label]))
    explanations_text = "\n\n".join(explanations) + "\n"
    (output_dir / "explanations.txt").write_text(explanations_text, encoding="utf-8")
    # Root-level benchmark files are regenerated from this same run so the
    # repository root and outputs/ always agree in format and numbers.
    (root / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    (root / "results_table.txt").write_text(results_table_text, encoding="utf-8")
    (root / "schedules.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "explanations.txt").write_text(explanations_text, encoding="utf-8")

    print("Experiment completed.")
    print(f"Best baseline: {best_base[0]}")
    print(f"Best CCE-QUBO: {best_cce[0]}")
    print(f"X (CCE vs baseline): {x_cce:.4f}")
    print(f"X ({quantum_label} vs baseline): {x_quantum:.4f}")
    print(f"Quantum backend: {quantum_backend}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
