"""Validate and summarize matched vLLM benchmark trials.

The utility intentionally reports descriptive statistics only. It rejects
incomplete, failed, or mismatched trial records rather than averaging them.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CONFIGURATIONS = (
    "fifo_no_prefix_cache",
    "fifo_prefix_cache",
    "cce_prefix_bounded",
)

METRICS = (
    "duration_ms",
    "mean_queue_delay_ms",
    "p95_queue_delay_ms",
    "mean_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "output_tokens_per_second",
)

MATCHED_FIELDS: Sequence[Tuple[str, ...]] = (
    ("workload_sha256",),
    ("workload_requests",),
    ("run_metadata", "trace_sha256"),
    ("run_metadata", "model"),
    ("run_metadata", "gpu_memory_utilization"),
    ("run_metadata", "max_model_len"),
    ("run_metadata", "max_num_seqs"),
    ("run_metadata", "max_inflight"),
    ("run_metadata", "max_reorder_wait_ms"),
    ("run_metadata", "warmup_requests"),
    ("run_metadata", "enforce_eager"),
    ("run_metadata", "seed"),
    ("run_metadata", "environment", "vllm_version"),
    ("run_metadata", "environment", "torch_version"),
    ("run_metadata", "environment", "gpu"),
    ("run_metadata", "server_environment", "VLLM_USE_V2_MODEL_RUNNER"),
)


def _value_at(record: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("missing required field: %s" % ".".join(path))
        value = value[key]
    return value


def _load_record(path: Path) -> Dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON result: %s" % path) from exc
    if not isinstance(record, dict):
        raise ValueError("result must be a JSON object: %s" % path)
    return record


def _load_trial(directory: Path) -> Dict[str, Dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError("trial directory does not exist: %s" % directory)

    records: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = _load_record(path)
        metadata = record.get("run_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("missing run metadata: %s" % path)
        configuration = metadata.get("configuration")
        if configuration not in CONFIGURATIONS:
            raise ValueError("unexpected configuration in %s: %r" % (path, configuration))
        if configuration in records:
            raise ValueError("duplicate configuration in %s: %s" % (directory, configuration))
        records[configuration] = record

    if set(records) != set(CONFIGURATIONS):
        missing = sorted(set(CONFIGURATIONS) - set(records))
        extra = sorted(set(records) - set(CONFIGURATIONS))
        raise ValueError("trial must contain exactly the three configurations; missing=%s extra=%s" % (missing, extra))

    for configuration, record in records.items():
        if record.get("requests_failed") != 0:
            raise ValueError("failed requests invalidate %s in %s" % (configuration, directory))
        if record.get("requests_succeeded") != record.get("workload_requests"):
            raise ValueError("incomplete workload invalidates %s in %s" % (configuration, directory))
        if record.get("usage_available_for_all_successes") is not True:
            raise ValueError("missing streaming usage invalidates %s in %s" % (configuration, directory))
        for metric in METRICS:
            if not isinstance(record.get(metric), (int, float)):
                raise ValueError("missing numeric %s in %s" % (metric, directory))
    return records


def _validate_matched(trials: Iterable[Dict[str, Dict[str, Any]]]) -> None:
    reference: Optional[Dict[str, Any]] = None
    for trial in trials:
        for record in trial.values():
            if reference is None:
                reference = record
                continue
            for path in MATCHED_FIELDS:
                if _value_at(record, path) != _value_at(reference, path):
                    raise ValueError("mismatched benchmark field: %s" % ".".join(path))


def _aggregate(values: List[float]) -> Dict[str, float]:
    return {
        "mean": round(sum(values) / len(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def summarize_trials(trial_directories: Sequence[Path]) -> Dict[str, Any]:
    """Return descriptive summaries after enforcing a matched, complete design."""
    if not trial_directories:
        raise ValueError("at least one trial directory is required")

    trials = [_load_trial(directory) for directory in trial_directories]
    _validate_matched(trials)
    reference = trials[0]["fifo_prefix_cache"]

    configuration_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for configuration in CONFIGURATIONS:
        configuration_summaries[configuration] = {
            metric: _aggregate([float(trial[configuration][metric]) for trial in trials])
            for metric in METRICS
        }

    paired_cce_over_prefix_fifo: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        ratios = [
            float(trial["cce_prefix_bounded"][metric]) / float(trial["fifo_prefix_cache"][metric])
            for trial in trials
        ]
        median_ratio = statistics.median(ratios)
        paired_cce_over_prefix_fifo[metric] = {
            "median_ratio": round(median_ratio, 6),
            "median_percent_change": round((median_ratio - 1.0) * 100.0, 3),
        }

    return {
        "schema_version": 1,
        "trial_count": len(trials),
        "trial_directories": [str(directory.resolve()) for directory in trial_directories],
        "comparison_metadata": {
            "workload_sha256": reference["workload_sha256"],
            "workload_requests": reference["workload_requests"],
            "model": _value_at(reference, ("run_metadata", "model")),
            "vllm_version": _value_at(reference, ("run_metadata", "environment", "vllm_version")),
            "torch_version": _value_at(reference, ("run_metadata", "environment", "torch_version")),
            "gpu": _value_at(reference, ("run_metadata", "environment", "gpu")),
            "max_model_len": _value_at(reference, ("run_metadata", "max_model_len")),
            "max_num_seqs": _value_at(reference, ("run_metadata", "max_num_seqs")),
            "max_inflight": _value_at(reference, ("run_metadata", "max_inflight")),
            "max_reorder_wait_ms": _value_at(reference, ("run_metadata", "max_reorder_wait_ms")),
            "warmup_requests": _value_at(reference, ("run_metadata", "warmup_requests")),
            "enforce_eager": _value_at(reference, ("run_metadata", "enforce_eager")),
            "seed": _value_at(reference, ("run_metadata", "seed")),
            "vllm_use_v2_model_runner": _value_at(
                reference, ("run_metadata", "server_environment", "VLLM_USE_V2_MODEL_RUNNER")
            ),
        },
        "configuration_summaries": configuration_summaries,
        "paired_cce_over_fifo_prefix_cache": paired_cce_over_prefix_fifo,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize matched vLLM benchmark trials.")
    parser.add_argument("--trial", type=Path, action="append", required=True, help="Directory containing one three-way trial.")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON destination outside a trial directory.")
    args = parser.parse_args(argv)

    summary = summarize_trials(args.trial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output), "trial_count": summary["trial_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
