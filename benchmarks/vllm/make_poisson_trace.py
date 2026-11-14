"""Generate a deterministic synthetic Poisson interleaved prefix-reuse trace.

This trace satisfies the two positive-result conditions diagnosed in
EVIDENCE.md (root cause of the -2.06% RTX 4060 result, 2026-11-13):

1. Requests arrive INTERLEAVED with waves: Poisson arrivals at a mean
   inter-arrival of about 50 ms, so later arrivals queue while earlier waves
   still run and dispatch order can change wave composition.
2. The prefix working set (8 groups) is larger than the KV prefix cache the
   server is configured to hold, so ordering changes which prefixes are cold.

It is an intended-workload benchmark, not ShareGPT, Alpaca, or a reproduction
of the vLLM SOSP workload. Each prefix group is exact-string identical so
vLLM automatic prefix caching and the CCE-QOS dispatcher are evaluated under a
documented condition.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

try:
    from .benchmark_protocol import BenchmarkRequest, write_trace
    from .make_prefix_trace import PREFIX_TOPICS, _prefix_text
except ImportError:  # Direct script execution from this directory.
    from benchmark_protocol import BenchmarkRequest, write_trace
    from make_prefix_trace import PREFIX_TOPICS, _prefix_text


def build_poisson_trace(
    request_count: int,
    prefix_groups: int,
    mean_inter_arrival_ms: float,
    max_tokens: int,
    seed: int,
) -> List[BenchmarkRequest]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if prefix_groups <= 0 or prefix_groups > len(PREFIX_TOPICS):
        raise ValueError("prefix_groups must be between 1 and %d" % len(PREFIX_TOPICS))
    if mean_inter_arrival_ms <= 0:
        raise ValueError("mean_inter_arrival_ms must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    rng = random.Random(seed)
    rate_per_ms = 1.0 / mean_inter_arrival_ms
    requests: List[BenchmarkRequest] = []
    arrival_ms = 0.0
    for index in range(request_count):
        # Poisson process: exponential inter-arrival gaps.
        arrival_ms += rng.expovariate(rate_per_ms)
        prefix_index = rng.randrange(prefix_groups)
        prefix_id = "group-%02d" % prefix_index
        prefix = _prefix_text(PREFIX_TOPICS[prefix_index], prefix_index)
        question = (
            "\n\nRequest %d: identify two risks and one bounded next experiment "
            "for scenario code %04d. Answer in three concise sentences."
        ) % (index, rng.randrange(10_000))
        requests.append(
            BenchmarkRequest(
                request_id="poisson-request-%04d" % index,
                arrival_offset_ms=int(round(arrival_ms)),
                prompt=prefix + question,
                max_tokens=max_tokens,
                prefix_id=prefix_id,
            )
        )
    return requests


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic Poisson interleaved prefix-reuse vLLM trace.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--prefix-groups", type=int, default=8)
    parser.add_argument("--mean-inter-arrival-ms", type=float, default=50.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args(argv)

    trace = build_poisson_trace(
        request_count=args.requests,
        prefix_groups=args.prefix_groups,
        mean_inter_arrival_ms=args.mean_inter_arrival_ms,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    digest = write_trace(args.output, trace)
    span_ms = trace[-1].arrival_offset_ms - trace[0].arrival_offset_ms if trace else 0
    print("wrote %d requests to %s" % (len(trace), args.output))
    print("arrival_span_ms=%d" % span_ms)
    print("trace_sha256=%s" % digest)
    print("workload_class=synthetic-poisson-interleaved-prefix-reuse-v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
