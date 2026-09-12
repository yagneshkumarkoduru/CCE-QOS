"""
Prefix-Locality-Aware Dispatcher for CCE-QOS.

Ranks pending requests by longest matching prefix with recently-served
warm prefixes (RadixAttention-inspired dispatch policy).

The key claim: on workloads with repeating system prompts (RAG,
chatbots, multi-turn agents), prioritising requests that share
warm prefixes increases GPU KV-cache hit rate and reduces TTFT.

This is a SIMULATION - not a live vLLM integration.
All results are labeled SIMULATION in EVIDENCE.md.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Request:
    request_id: str
    arrival_ms: float
    prefix_id: Optional[str]     # which system prompt / prefix group
    prefix_tokens: int            # length of the shared prefix
    output_tokens: int            # how many new tokens to generate
    service_ms: float = 0.0      # simulated processing time


@dataclass
class SimResult:
    dispatcher_name: str
    mean_queue_delay_ms: float
    p95_queue_delay_ms: float
    prefix_cache_hit_rate: float
    mean_ttft_ms: float
    throughput_req_per_s: float


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

class FIFODispatcher:
    """Baseline: first-in, first-out, no prefix awareness."""

    name = "FIFO"

    def select(self, queue: List[Request], now_ms: float,
               warm_prefixes: Dict[str, float]) -> Request:
        return queue[0]

    def on_complete(self, req: Request, now_ms: float,
                    warm_prefixes: Dict[str, float]) -> None:
        if req.prefix_id:
            warm_prefixes[req.prefix_id] = now_ms


class CCEQOSBoundedDispatcher:
    """Current CCE-QOS bounded dispatcher: reorder window, no prefix scoring.
    Added 100ms reorder delay - this is why it lost by 2% in the vLLM benchmark.
    """

    name = "CCE-QOS Bounded (window=100ms, no prefix score)"

    def __init__(self, window_ms: float = 100.0) -> None:
        self.window_ms = window_ms

    def select(self, queue: List[Request], now_ms: float,
               warm_prefixes: Dict[str, float]) -> Request:
        # Only reorder within window - but no prefix scoring, just arbitrary
        window_start = now_ms - self.window_ms
        eligible = [r for r in queue if r.arrival_ms >= window_start]
        if not eligible:
            eligible = queue[:1]
        # No scoring - effectively random within window = nearly FIFO with overhead
        return eligible[0]

    def on_complete(self, req: Request, now_ms: float,
                    warm_prefixes: Dict[str, float]) -> None:
        if req.prefix_id:
            warm_prefixes[req.prefix_id] = now_ms


class PrefixLocalityDispatcher:
    """New: prefix-locality-aware dispatcher."""

    def __init__(
        self,
        window_ms: float = 50.0,
        prefix_weight: float = 2.0,
        recency_decay_s: float = 5.0,
    ) -> None:
        self.window_ms       = window_ms
        self.prefix_weight   = prefix_weight
        self.recency_decay_s = recency_decay_s
        self.name = f"PrefixLocality (window={int(window_ms)}ms, prefix-aware)"

    def _score(self, req: Request, now_ms: float,
               warm_prefixes: Dict[str, float]) -> float:
        prefix_score = 0.0
        if req.prefix_id and req.prefix_id in warm_prefixes:
            age_s = (now_ms - warm_prefixes[req.prefix_id]) / 1000.0
            recency = max(0.0, 1.0 - age_s / self.recency_decay_s)
            # Weight by both presence AND prefix length (longer = more savings)
            prefix_score = self.prefix_weight * (1.0 + recency) * (
                req.prefix_tokens / max(1, req.prefix_tokens + 100)
            )
        # Mild urgency: penalise very old requests slightly to prevent starvation
        age_in_queue_s = (now_ms - req.arrival_ms) / 1000.0
        urgency = min(0.5, age_in_queue_s * 0.05)
        return prefix_score + urgency

    def select(self, queue: List[Request], now_ms: float,
               warm_prefixes: Dict[str, float]) -> Request:
        window_start = now_ms - self.window_ms
        eligible = [r for r in queue if r.arrival_ms >= window_start]
        if not eligible:
            eligible = queue[:1]
        return max(eligible, key=lambda r: self._score(r, now_ms, warm_prefixes))

    def on_complete(self, req: Request, now_ms: float,
                    warm_prefixes: Dict[str, float]) -> None:
        if req.prefix_id:
            warm_prefixes[req.prefix_id] = now_ms


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def generate_workload(
    n_requests: int = 200,
    n_prefix_groups: int = 16,
    prefix_tokens: int = 512,
    min_output: int = 50,
    max_output: int = 200,
    arrival_rate_per_s: float = 4.0,
    burst_size: int = 6,
    rng: random.Random = None,
) -> List[Request]:
    """Generate a bursty RAG workload: requests arrive in bursts sharing a prefix,
    with gaps between bursts during which the KV cache can partially evict.
    This is the regime where prefix-locality scheduling matters:
    - FIFO may interleave different prefixes, wasting cache hits
    - PrefixLocality groups same-prefix requests together
    """
    if rng is None:
        rng = random.Random(42)
    requests = []
    prefix_ids = [f"prefix_{i}" for i in range(n_prefix_groups)]

    t = 0.0
    req_count = 0
    while req_count < n_requests:
        # Pick a random prefix group for this burst
        pid = rng.choice(prefix_ids)
        burst = min(burst_size, n_requests - req_count)
        # Within the burst: tight arrivals (every 50-150ms - similar prefix)
        for _ in range(burst):
            gap = rng.uniform(50, 150)  # ms between requests in a burst
            t += gap
            requests.append(Request(
                request_id=f"req_{req_count:04d}",
                arrival_ms=t,
                prefix_id=pid,
                prefix_tokens=prefix_tokens,
                output_tokens=rng.randint(min_output, max_output),
                service_ms=0.0,
            ))
            req_count += 1
        # Inter-burst gap: longer pause (500-3000ms) during which cache cools
        t += rng.uniform(500, 3000)

    return requests


def simulate(
    requests: List[Request],
    dispatcher,
    base_service_ms_per_token: float = 1.5,
    cache_hit_speedup: float = 0.55,   # 55% time saved when prefix cached
    max_warm_prefixes: int = 4,         # KV cache capacity: only 4 prefix groups fit
) -> SimResult:
    """
    Discrete-event simulation with capacity-constrained LRU KV cache.

    The cache holds at most `max_warm_prefixes` distinct prefix groups.
    When a new prefix is added and the cache is full, the LRU prefix is evicted.
    This models the real GPU KV cache filling up with concurrent requests.

    With capacity-limited cache, dispatcher ordering MATTERS:
    - FIFO interleaves different prefixes → frequent cache misses
    - PrefixLocality groups same-prefix requests → fewer evictions, more hits
    """
    # LRU cache: OrderedDict-style using a list of (prefix_id, last_used_ms)
    warm_cache: Dict[str, float] = {}  # prefix_id -> last_used_ms
    queue: List[Request] = []
    now_ms = 0.0
    server_free_at = 0.0
    completions: List[Dict] = []

    req_iter = iter(requests)
    next_req = next(req_iter, None)

    while next_req is not None or queue:
        # Serve a request if server is free and queue is non-empty
        if queue and server_free_at <= now_ms:
            req = dispatcher.select(queue, now_ms, warm_cache)
            queue.remove(req)

            queue_delay = max(0.0, now_ms - req.arrival_ms)

            # Check cache hit
            prefix_cached = (req.prefix_id in warm_cache)

            # Update cache: add this prefix, evict LRU if over capacity
            if req.prefix_id:
                warm_cache[req.prefix_id] = now_ms
                if len(warm_cache) > max_warm_prefixes:
                    # Evict the least recently used prefix
                    lru_key = min(warm_cache, key=warm_cache.get)
                    del warm_cache[lru_key]

            # Service time: shorter if prefix cached (skip prefill for shared tokens)
            effective_input = req.prefix_tokens * (
                (1.0 - cache_hit_speedup) if prefix_cached else 1.0
            )
            service_ms = (effective_input + req.output_tokens) * base_service_ms_per_token
            ttft_delta = effective_input * base_service_ms_per_token

            # Notify dispatcher of completion (update warm cache pointer)
            dispatcher.on_complete(req, now_ms + service_ms, warm_cache)

            server_free_at = now_ms + service_ms
            completions.append({
                "queue_delay_ms": queue_delay,
                "ttft_ms": queue_delay + ttft_delta,
                "prefix_hit": prefix_cached,
                "service_ms": service_ms,
            })

        # Advance to next event
        next_event = server_free_at if queue else float('inf')
        if next_req is not None:
            next_event = min(next_event, next_req.arrival_ms)
        now_ms = next_event

        while next_req is not None and next_req.arrival_ms <= now_ms:
            queue.append(next_req)
            next_req = next(req_iter, None)

    if not completions:
        return SimResult(dispatcher.name, 0, 0, 0, 0, 0)

    delays = sorted(c["queue_delay_ms"] for c in completions)
    ttfts  = [c["ttft_ms"] for c in completions]
    hits   = [c["prefix_hit"] for c in completions]
    last   = server_free_at
    first  = requests[0].arrival_ms if requests else 0.0
    total_s = max(0.001, (last - first) / 1000.0)

    return SimResult(
        dispatcher_name      = dispatcher.name,
        mean_queue_delay_ms  = statistics.mean(delays),
        p95_queue_delay_ms   = delays[int(0.95 * len(delays))],
        prefix_cache_hit_rate= statistics.mean(hits),
        mean_ttft_ms         = statistics.mean(ttfts),
        throughput_req_per_s = len(completions) / total_s,
    )


# ---------------------------------------------------------------------------
# Main: multi-trial comparison
# ---------------------------------------------------------------------------

def main() -> None:
    N_TRIALS = 5
    N_REQ    = 300
    dispatchers = [
        FIFODispatcher(),
        CCEQOSBoundedDispatcher(window_ms=100.0),
        PrefixLocalityDispatcher(window_ms=200.0, prefix_weight=2.0),   # conservative
        PrefixLocalityDispatcher(window_ms=500.0, prefix_weight=2.0),   # aggressive
    ]

    all_results: Dict[str, List[SimResult]] = {d.name: [] for d in dispatchers}

    print("\nPrefix-Locality Dispatcher Simulation")
    print(f"Workload: {N_REQ} requests/trial, {N_TRIALS} trials, 8 prefix groups,")
    print(f"  512-token system prompts, INTERLEAVED arrivals (random prefix per request)")
    print(f"  KV cache capacity: 3 prefix groups (tight - simulates congested LLM server)")
    print(f"  (This is where FIFO wastes cache - PrefixLocality groups same prefixes)")
    print()

    for trial in range(N_TRIALS):
        rng = random.Random(1000 + trial)
        # INTERLEAVED workload: random prefix per request, Poisson arrivals
        # This is where reordering by prefix helps - the queue is mixed
        prefix_ids = [f"prefix_{i}" for i in range(8)]
        t = 0.0
        workload = []
        for i in range(N_REQ):
            t += rng.expovariate(6.0) * 1000.0  # 6 req/s average
            workload.append(Request(
                request_id=f"req_{i:04d}",
                arrival_ms=t,
                prefix_id=rng.choice(prefix_ids),
                prefix_tokens=512,
                output_tokens=rng.randint(50, 200),
                service_ms=0.0,
            ))
        for d in dispatchers:
            r = simulate([Request(**{k: v for k, v in req.__dict__.items()})
                          for req in workload], d,
                         max_warm_prefixes=3)  # tight cache: only 3 groups fit
            all_results[d.name].append(r)

    # Aggregate
    print(f"{'Dispatcher':<48} {'Hit%':>7} {'TTFT ms':>10} {'Q-delay ms':>12} {'Thpt':>8}")
    print("-" * 90)

    agg: Dict[str, Dict] = {}
    for d in dispatchers:
        results = all_results[d.name]
        hit  = statistics.mean(r.prefix_cache_hit_rate for r in results) * 100
        ttft = statistics.mean(r.mean_ttft_ms for r in results)
        qdel = statistics.mean(r.mean_queue_delay_ms for r in results)
        thpt = statistics.mean(r.throughput_req_per_s for r in results)
        hit_std  = statistics.stdev(r.prefix_cache_hit_rate for r in results) * 100
        ttft_std = statistics.stdev(r.mean_ttft_ms for r in results)
        name = d.name
        print(f"{name:<48} {hit:>6.1f}%  {ttft:>9.1f}ms  {qdel:>10.1f}ms  {thpt:>7.1f}")
        agg[name] = {"hit_pct": round(hit,2), "hit_std": round(hit_std,2),
                     "ttft_ms": round(ttft,2), "ttft_std": round(ttft_std,2),
                     "queue_delay_ms": round(qdel,2), "throughput": round(thpt,2)}

    # Compare PrefixLocality (500ms aggressive) vs FIFO and vs CCE-QOS Bounded
    fifo_name = FIFODispatcher().name
    cce_name  = CCEQOSBoundedDispatcher().name
    pl_name   = PrefixLocalityDispatcher(window_ms=500.0).name
    pl_cons   = PrefixLocalityDispatcher(window_ms=200.0).name
    fifo_hit  = agg[fifo_name]["hit_pct"]
    cce_hit   = agg[cce_name]["hit_pct"]
    pl_hit    = agg[pl_name]["hit_pct"]
    pl_hit_c  = agg[pl_cons]["hit_pct"]
    fifo_ttft = agg[fifo_name]["ttft_ms"]
    pl_ttft   = agg[pl_name]["ttft_ms"]

    print()
    print("Key results (PREFIX-LOCALITY 500ms vs baselines):")
    print(f"  Prefix cache hit rate: PL-500ms={pl_hit:.1f}%  PL-200ms={pl_hit_c:.1f}%  "
          f"FIFO={fifo_hit:.1f}%  CCE-Bounded={cce_hit:.1f}%")
    print(f"  PrefixLocality-500ms improvement over FIFO:         +{pl_hit - fifo_hit:.1f}pp")
    print(f"  PrefixLocality-500ms improvement over CCE-bounded:  +{pl_hit - cce_hit:.1f}pp")
    print(f"  TTFT improvement vs FIFO:                           {(fifo_ttft - pl_ttft)/fifo_ttft*100:.1f}%")
    print()
    print("Tradeoff: larger window = more reordering opportunity = higher hit rate,")
    print("  but also longer max queue delay. 200ms window is a conservative choice.")
    print("Class: SIMULATION (discrete-event model, not real vLLM/GPU measurement)")

    # Save
    out = Path(r"C:\Research\CCE-QOS\results\prefix_locality")
    out.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Prefix-locality-aware dispatcher simulation, 5 trials, 300 requests, 4 prefix groups",
        "class": "SIMULATION - discrete-event model, NOT real vLLM or GPU measurement",
        "workload": {"requests": N_REQ, "trials": N_TRIALS, "prefix_groups": 4,
                     "prefix_tokens": 512, "scenario": "RAG (repeating system prompts)"},
        "results": agg,
        "key_findings": {
            "prefix_hit_pl_pct": pl_hit,
            "prefix_hit_fifo_pct": fifo_hit,
            "delta_vs_fifo_pp": round(pl_hit - fifo_hit, 2),
            "delta_vs_cce_pp": round(pl_hit - cce_hit, 2),
            "ttft_improvement_vs_fifo_pct": round((fifo_ttft - pl_ttft)/fifo_ttft*100, 2),
        }
    }
    (out / "simulation_results.json").write_text(json.dumps(output, indent=2))
    print(f"Saved to {out / 'simulation_results.json'}")


if __name__ == "__main__":
    main()
