"""Adaptive Penalty Refinement (APR) arm tests on a small synthetic workload.

The APR round loop must be deterministic (fixed seeds), and the final
schedule's violation mass must never exceed the violation mass of any
candidate the loop produced (the feasibility-preserving polish only accepts
improvements on the lexicographic (violation mass, canonical energy) key).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_experiment as run_experiment
from bandwidth_estimator import BandwidthEstimator
from core_types import DVFSState, HardwareConfig, MemoryLevel, OperatorGraph, OperatorNode
from cost_model import ScheduleCostModel
from fusion_logic import FusionLogic
from memory_hierarchy import MemoryHierarchy


def _small_graph() -> OperatorGraph:
    nodes = [
        OperatorNode(id=0, name="a_conv", op_type="conv", flops=8e7, input_bytes=300.0,
                     output_bytes=260.0, compute_cycles=80, dependencies=[]),
        OperatorNode(id=1, name="a_norm", op_type="batchnorm", flops=2e7, input_bytes=260.0,
                     output_bytes=260.0, compute_cycles=30, dependencies=[0]),
        OperatorNode(id=2, name="a_pool", op_type="pool", flops=1.5e7, input_bytes=260.0,
                     output_bytes=120.0, compute_cycles=24, dependencies=[1]),
        OperatorNode(id=3, name="a_dense", op_type="dense", flops=4e7, input_bytes=120.0,
                     output_bytes=90.0, compute_cycles=42, dependencies=[2]),
    ]
    return OperatorGraph(nodes=nodes, edges=[(0, 1), (1, 2), (2, 3)])


def _small_hardware() -> HardwareConfig:
    return HardwareConfig(
        resources=["npu0"],
        memory_levels=[MemoryLevel(name="L2", capacity_bytes=1440.0, bandwidth_gbps=120.0)],
        dvfs_states=[DVFSState(name="nominal", freq_ghz=1.0, voltage_v=0.9, energy_per_cycle=1.0)],
        max_time_slots=6,
        bank_count=2,
    )


def _build_cce(graph: OperatorGraph, hardware: HardwareConfig):
    hw_cfg = {"sram_capacity": hardware.memory_levels[0].capacity_bytes, "sram_banks": hardware.bank_count}
    cost_model = ScheduleCostModel(
        memory_hierarchy=MemoryHierarchy(hw_cfg),
        bandwidth_estimator=BandwidthEstimator(hw_cfg),
        fusion_logic=FusionLogic({}),
        weights={},
    )
    penalties = {"unique_exec": 1.6, "dep": 1.0, "dvfs_one_hot": 0.6, "mem_cap": 0.0, "mem_bind": 0.0}
    cce = run_experiment.CCEEvaluator(
        graph,
        hardware,
        cost_model,
        {"comp": 1.0, "energy": 1.0, "lat": 0.25, "dvfs": 0.2},
        {"reuse": 1.1, "fuse": 1.8, "bw": 0.7},
        {"bank": 0.0, "burst": 0.0, "stall": 0.0, "parallelism": 0.0},
        penalties,
    )
    return cce, penalties


def _apr_config() -> dict:
    return {
        "apr": {
            "rounds": 3,
            "iterations_per_round": 24,
            "anneal_patience": 2,
            "anneal_decay": 0.75,
            "polish_iterations": 60,
        },
        "search": {"annealing_start_temp": 3.0, "annealing_end_temp": 0.05},
    }


def test_apr_arm_is_deterministic():
    graph = _small_graph()
    cce, penalties = _build_cce(graph, _small_hardware())
    result_a = run_experiment._run_apr_arm(
        graph, _apr_config(), penalties, cce, lambda ev: float(ev["energy_breakdown"]["total_energy"])
    )
    result_b = run_experiment._run_apr_arm(
        graph, _apr_config(), penalties, cce, lambda ev: float(ev["energy_breakdown"]["total_energy"])
    )
    assert result_a["order"] == result_b["order"]
    assert result_a["evaluation"]["energy_breakdown"] == result_b["evaluation"]["energy_breakdown"]
    assert result_a["metadata"]["round_trace"] == result_b["metadata"]["round_trace"]


def test_apr_arm_final_violation_mass_does_not_exceed_candidates():
    graph = _small_graph()
    cce, penalties = _build_cce(graph, _small_hardware())
    result = run_experiment._run_apr_arm(
        graph, _apr_config(), penalties, cce, lambda ev: float(ev["energy_breakdown"]["total_energy"])
    )
    evaluation = result["evaluation"]
    final_violation = run_experiment._violation_mass(evaluation)
    # The polish only accepts moves that strictly reduce the violation mass or
    # keep it while strictly reducing the canonical energy, so the final
    # schedule's violation mass is <= the starting candidate's mass.
    assert final_violation <= result["metadata"]["polish"]["violation_mass"] + 1e-9
    # Every recorded round ran on the same workload size.
    assert len(result["metadata"]["round_trace"]) == _apr_config()["apr"]["rounds"]


def test_apr_arm_round_trace_records_penalized_objective():
    graph = _small_graph()
    cce, penalties = _build_cce(graph, _small_hardware())
    result = run_experiment._run_apr_arm(
        graph, _apr_config(), penalties, cce, lambda ev: float(ev["energy_breakdown"]["total_energy"])
    )
    trace = result["metadata"]["round_trace"]
    for entry in trace:
        assert {"round", "objective", "penalized_objective", "violation_mass", "penalties"} <= set(entry)
        # The penalized objective at the round's own multipliers can differ
        # from the canonical objective; both must be recorded explicitly.
        assert isinstance(entry["penalized_objective"], float)
        assert isinstance(entry["objective"], float)
