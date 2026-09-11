"""Tests for matched-trial validation and descriptive vLLM aggregation."""

import json

import pytest

from benchmarks.vllm.summarize import summarize_trials


def _record(configuration, duration_ms):
    return {
        "workload_sha256": "trace-sha",
        "workload_requests": 4,
        "requests_succeeded": 4,
        "requests_failed": 0,
        "usage_available_for_all_successes": True,
        "duration_ms": duration_ms,
        "mean_queue_delay_ms": 10.0,
        "p95_queue_delay_ms": 12.0,
        "mean_end_to_end_latency_ms": 20.0,
        "p95_end_to_end_latency_ms": 24.0,
        "mean_ttft_ms": 15.0,
        "p95_ttft_ms": 18.0,
        "output_tokens_per_second": 100.0,
        "run_metadata": {
            "configuration": configuration,
            "trace_sha256": "trace-sha",
            "model": "model",
            "gpu_memory_utilization": 0.72,
            "max_model_len": 2048,
            "max_num_seqs": 16,
            "max_inflight": 16,
            "max_reorder_wait_ms": 100,
            "warmup_requests": 4,
            "enforce_eager": False,
            "seed": 7,
            "environment": {
                "vllm_version": "0.29.0",
                "torch_version": "2.13.0",
                "gpu": "GPU",
            },
            "server_environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
        },
    }


def _write_trial(directory, durations):
    directory.mkdir()
    for configuration, duration_ms in durations.items():
        (directory / (configuration + ".json")).write_text(
            json.dumps(_record(configuration, duration_ms)), encoding="utf-8"
        )


def test_summary_uses_matched_trial_medians(tmp_path):
    first = tmp_path / "trial-1"
    second = tmp_path / "trial-2"
    _write_trial(
        first,
        {
            "fifo_no_prefix_cache": 100.0,
            "fifo_prefix_cache": 80.0,
            "cce_prefix_bounded": 72.0,
        },
    )
    _write_trial(
        second,
        {
            "fifo_no_prefix_cache": 110.0,
            "fifo_prefix_cache": 90.0,
            "cce_prefix_bounded": 81.0,
        },
    )

    summary = summarize_trials([first, second])

    assert summary["trial_count"] == 2
    assert summary["configuration_summaries"]["fifo_prefix_cache"]["duration_ms"]["median"] == 85.0
    assert summary["paired_cce_over_fifo_prefix_cache"]["duration_ms"]["median_ratio"] == 0.9


def test_summary_rejects_a_failed_trial(tmp_path):
    trial = tmp_path / "trial"
    _write_trial(
        trial,
        {
            "fifo_no_prefix_cache": 100.0,
            "fifo_prefix_cache": 80.0,
            "cce_prefix_bounded": 72.0,
        },
    )
    broken = json.loads((trial / "cce_prefix_bounded.json").read_text(encoding="utf-8"))
    broken["requests_failed"] = 1
    (trial / "cce_prefix_bounded.json").write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ValueError, match="failed requests"):
        summarize_trials([trial])
