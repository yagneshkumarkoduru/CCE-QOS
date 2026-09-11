"""Generate a deterministic, explicitly synthetic prefix-reuse trace.

The trace is an intended-workload benchmark for prefix-aware admission. It is
not labelled ShareGPT, Alpaca, or a reproduction of the vLLM SOSP workload.
Each prefix group is exact-string identical so vLLM automatic prefix caching
and the CCE-QOS dispatcher can be evaluated under a documented condition.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

try:
    from .benchmark_protocol import BenchmarkRequest, write_trace
except ImportError:  # Direct script execution from this directory.
    from benchmark_protocol import BenchmarkRequest, write_trace


PREFIX_TOPICS = (
    "embedded robotics safety review",
    "medical sensing reliability assessment",
    "edge-model memory planning",
    "event-driven vision validation",
    "wireless latency budget analysis",
    "compiler scheduling constraint review",
    "sensor calibration experiment",
    "low-power inference deployment",
)


def _prefix_text(topic: str, group_index: int) -> str:
    clauses = [
        "You are assisting with a bounded technical analysis.",
        "Use the supplied context only and state assumptions explicitly.",
        "Preserve units, distinguish measurements from models, and avoid unsupported conclusions.",
        "The shared context concerns %s for study group %d." % (topic, group_index),
    ]
    # Repetition makes each group large enough to exercise an actual KV prefix
    # without hiding the fact that the workload is constructed.
    return " ".join(clauses * 20)


def build_trace(
    request_count: int,
    prefix_groups: int,
    arrival_interval_ms: int,
    max_tokens: int,
    seed: int,
) -> List[BenchmarkRequest]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if prefix_groups <= 0 or prefix_groups > len(PREFIX_TOPICS):
        raise ValueError("prefix_groups must be between 1 and %d" % len(PREFIX_TOPICS))
    if arrival_interval_ms < 0:
        raise ValueError("arrival_interval_ms must be non-negative")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    rng = random.Random(seed)
    group_assignment = [index % prefix_groups for index in range(request_count)]
    # Shuffle group assignment deterministically so FIFO lacks artificial
    # prefix locality while every group receives nearly equal traffic.
    rng.shuffle(group_assignment)
    requests: List[BenchmarkRequest] = []
    for index, prefix_index in enumerate(group_assignment):
        prefix_id = "group-%02d" % prefix_index
        prefix = _prefix_text(PREFIX_TOPICS[prefix_index], prefix_index)
        question = (
            "\n\nRequest %d: identify two risks and one bounded next experiment "
            "for scenario code %04d. Answer in three concise sentences."
        ) % (index, rng.randrange(10_000))
        requests.append(
            BenchmarkRequest(
                request_id="prefix-request-%04d" % index,
                arrival_offset_ms=index * arrival_interval_ms,
                prompt=prefix + question,
                max_tokens=max_tokens,
                prefix_id=prefix_id,
            )
        )
    return requests


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic prefix-reuse vLLM trace.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--prefix-groups", type=int, default=4)
    parser.add_argument("--arrival-interval-ms", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args(argv)

    trace = build_trace(
        request_count=args.requests,
        prefix_groups=args.prefix_groups,
        arrival_interval_ms=args.arrival_interval_ms,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    digest = write_trace(args.output, trace)
    print("wrote %d requests to %s" % (len(trace), args.output))
    print("trace_sha256=%s" % digest)
    print("workload_class=synthetic-prefix-reuse-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
