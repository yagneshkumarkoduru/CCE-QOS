"""Tests for the calibrated vLLM dispatch simulation.

The simulation replays the recorded 48-request prefix trace through a
wave-based engine model calibrated against the 2026-09-11 local vLLM run.
It must reproduce both measured arms within +/-5 percent and the measured
CCE-over-FIFO duration delta, and it must stay deterministic.
"""

from benchmarks.vllm.calibrated_dispatch_sim import (
    MEASURED,
    _simulate_arm,
    calibrate,
    run_validation,
)
from benchmarks.vllm.benchmark_protocol import load_trace
from pathlib import Path


TRACE = Path(__file__).resolve().parent.parent / "benchmarks" / "vllm" / "traces" / "prefix-reuse-full-20260911.jsonl"


def test_calibrated_sim_reproduces_both_arms_within_five_percent():
    report = run_validation(TRACE)
    assert abs(report["fifo_prefix_cache"]["duration_pct_error"]) <= 5.0
    assert abs(report["cce_prefix_bounded"]["duration_pct_error"]) <= 5.0


def test_calibrated_sim_reproduces_measured_cce_regression_sign_and_size():
    report = run_validation(TRACE)
    sim_delta = report["sim_cce_over_fifo_duration_pct"]
    meas_delta = report["measured_cce_over_fifo_duration_pct"]
    # Same sign (a regression, not an improvement) and within 0.5pp.
    assert sim_delta > 0.0
    assert abs(sim_delta - meas_delta) <= 0.5


def test_cache_pressure_sweep_prices_evictions():
    report = run_validation(TRACE)
    sweep = {row["cache_capacity_groups"]: row for row in report["cache_pressure_sweep"]}
    # Tighter cache -> fewer hits for FIFO, never more.
    assert sweep[3]["fifo_cache_hits"] < sweep["unlimited"]["fifo_cache_hits"]
    assert sweep[1]["fifo_cache_hits"] < sweep[3]["fifo_cache_hits"]
    # On this bursty trace the bounded dispatcher cannot beat FIFO even
    # under pressure: the delta stays positive (a regression).
    for row in report["cache_pressure_sweep"]:
        assert row["cce_over_fifo_pct"] > 0.0


def test_simulation_is_deterministic():
    trace = load_trace(TRACE)
    rate = calibrate(trace)
    first = _simulate_arm(trace, "prefix_bounded", 100, rate)
    second = _simulate_arm(trace, "prefix_bounded", 100, rate)
    assert first == second
    assert MEASURED["fifo_prefix_cache"]["duration_ms"] == 1204.116
