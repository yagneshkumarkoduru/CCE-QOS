"""Deterministic KV-cache admission and prefix-cache simulation.

This module is deliberately a *simulation*, not a vLLM implementation and not
a GPU measurement. It compares two capacity-management policies using the
same request trace:

* ``contiguous`` reserves ``max_model_tokens`` for every admitted request.
  Requests wait in a FIFO queue until one complete reservation is available.
* ``paged`` commits only the request's known prompt-plus-output block demand,
  physically materialises blocks as prefill/decode progress, and optionally
  shares complete, block-aligned prompt prefixes.

Neither policy pretends that a normal vLLM GPU cache pages active blocks to
host DRAM. When GPU capacity is exhausted, requests queue. This makes the
reported throughput and latency metrics useful capacity-model results instead
of invented DRAM-fault measurements.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _percentile(values: List[int], percentile: float) -> float:
    """Return a deterministic linear-interpolated percentile without NumPy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


@dataclass(frozen=True)
class KVRequest:
    """One request in a discrete, token-stepped serving trace.

    ``prefix_tokens`` must be a complete prefix of ``prompt_tokens``. Prefix
    reuse is only modeled for block-aligned prefixes, matching the safe reuse
    boundary used by block-table based KV caches.
    """

    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int
    prefix_id: Optional[str] = None
    prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.prefix_tokens < 0 or self.prefix_tokens > self.prompt_tokens:
            raise ValueError("prefix_tokens must be within prompt_tokens")
        if self.prefix_tokens and not self.prefix_id:
            raise ValueError("prefix_id is required when prefix_tokens is non-zero")
        if self.prefix_id and not self.prefix_tokens:
            raise ValueError("prefix_tokens is required when prefix_id is set")


@dataclass(frozen=True)
class PagingConfig:
    """Capacity and service assumptions for a trace replay.

    ``prefill_tokens_per_step`` is a deliberately explicit model parameter,
    not a claim about a particular GPU. Keep it identical between policies.
    """

    block_size_tokens: int = 16
    pool_blocks: int = 1600
    max_model_tokens: int = 1024
    prefill_tokens_per_step: int = 128
    max_active_requests: Optional[int] = None

    def __post_init__(self) -> None:
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if self.pool_blocks <= 0:
            raise ValueError("pool_blocks must be positive")
        if self.max_model_tokens <= 0:
            raise ValueError("max_model_tokens must be positive")
        if self.prefill_tokens_per_step <= 0:
            raise ValueError("prefill_tokens_per_step must be positive")
        if self.max_active_requests is not None and self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive when set")


@dataclass
class _PrefixEntry:
    block_ids: List[int]
    active_references: int = 0
    last_used_step: int = 0


@dataclass
class _RequestState:
    request: KVRequest
    admitted_step: int
    reserved_private_blocks: int
    prefix_entry: Optional[_PrefixEntry]
    private_prefill_tokens: int = 0
    generated_tokens: int = 0
    private_block_ids: List[int] = field(default_factory=list)
    first_token_step: Optional[int] = None


@dataclass(frozen=True)
class SimulationResult:
    """Metrics from one policy on one immutable request trace."""

    mode: str
    prefix_caching: bool
    requests_completed: int
    total_output_tokens: int
    elapsed_steps: int
    throughput_output_tokens_per_step: float
    mean_queue_delay_steps: float
    p95_queue_delay_steps: float
    mean_ttft_steps: float
    p95_ttft_steps: float
    mean_end_to_end_latency_steps: float
    p95_end_to_end_latency_steps: float
    mean_pool_utilization_pct: float
    peak_pool_utilization_pct: float
    mean_committed_capacity_pct: float
    peak_active_requests: int
    peak_committed_blocks: int
    cached_prefix_evictions: int
    reused_prefix_blocks: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMKVCacheScheduler:
    """Trace-replay simulator for contiguous and paged KV-cache admission.

    The simulator keeps a capacity commitment for every active paged request.
    This prevents the invalid behavior of admitting work that cannot complete
    later because active KV blocks cannot be evicted safely. Prefix-cache
    blocks with no active references are the only blocks eligible for LRU
    eviction.
    """

    def __init__(self, block_size_tokens: int = 16, num_gpu_blocks: int = 1600):
        self.block_size = block_size_tokens
        self.total_blocks = num_gpu_blocks

    def _validate_requests(self, requests: Iterable[KVRequest], config: PagingConfig) -> List[KVRequest]:
        trace = sorted(list(requests), key=lambda request: (request.arrival_step, request.request_id))
        ids = set()
        for request in trace:
            if request.request_id in ids:
                raise ValueError("request IDs must be unique")
            ids.add(request.request_id)
            if request.prompt_tokens + request.output_tokens > config.max_model_tokens:
                raise ValueError(
                    "request %s exceeds max_model_tokens (%d > %d)"
                    % (request.request_id, request.prompt_tokens + request.output_tokens, config.max_model_tokens)
                )
            if request.prefix_tokens % config.block_size_tokens:
                raise ValueError(
                    "prefix_tokens must be block-aligned for safe cache reuse: %s" % request.request_id
                )
        return trace

    @staticmethod
    def _private_prompt_tokens(request: KVRequest, prefix_caching: bool) -> int:
        return request.prompt_tokens - (request.prefix_tokens if prefix_caching else 0)

    def _max_private_blocks(self, request: KVRequest, config: PagingConfig, prefix_caching: bool) -> int:
        tokens = self._private_prompt_tokens(request, prefix_caching) + request.output_tokens
        return _ceil_div(tokens, config.block_size_tokens)

    @staticmethod
    def _release_blocks(free_blocks: List[int], block_ids: Iterable[int]) -> None:
        free_blocks.extend(block_ids)
        free_blocks.sort()

    def simulate(
        self,
        requests: Iterable[KVRequest],
        config: PagingConfig,
        mode: str = "paged",
        enable_prefix_caching: bool = False,
    ) -> SimulationResult:
        """Replay ``requests`` under one allocation policy.

        ``mode`` is either ``"paged"`` or ``"contiguous"``. Prefix caching
        is meaningful only in paged mode and is rejected for contiguous mode to
        keep the baseline definition unambiguous.
        """
        if mode not in {"paged", "contiguous"}:
            raise ValueError("mode must be 'paged' or 'contiguous'")
        if mode == "contiguous" and enable_prefix_caching:
            raise ValueError("contiguous mode intentionally does not share prefixes")

        trace = self._validate_requests(requests, config)
        if not trace:
            return SimulationResult(
                mode=mode,
                prefix_caching=enable_prefix_caching,
                requests_completed=0,
                total_output_tokens=0,
                elapsed_steps=0,
                throughput_output_tokens_per_step=0.0,
                mean_queue_delay_steps=0.0,
                p95_queue_delay_steps=0.0,
                mean_ttft_steps=0.0,
                p95_ttft_steps=0.0,
                mean_end_to_end_latency_steps=0.0,
                p95_end_to_end_latency_steps=0.0,
                mean_pool_utilization_pct=0.0,
                peak_pool_utilization_pct=0.0,
                mean_committed_capacity_pct=0.0,
                peak_active_requests=0,
                peak_committed_blocks=0,
                cached_prefix_evictions=0,
                reused_prefix_blocks=0,
            )

        free_blocks = list(range(config.pool_blocks))
        active: Dict[str, _RequestState] = {}
        waiting: List[KVRequest] = []
        prefixes: Dict[str, _PrefixEntry] = {}
        active_private_reservation = 0
        arrivals_index = 0
        current_step = trace[0].arrival_step
        final_step = current_step
        total_output_tokens = 0
        queue_delays: List[int] = []
        ttft: List[int] = []
        end_to_end: List[int] = []
        used_blocks_samples: List[int] = []
        committed_blocks_samples: List[int] = []
        peak_active = 0
        peak_committed = 0
        cached_prefix_evictions = 0
        reused_prefix_blocks = 0

        def cache_block_count() -> int:
            return sum(len(entry.block_ids) for entry in prefixes.values())

        def committed_blocks() -> int:
            if mode == "contiguous":
                return sum(state.reserved_private_blocks for state in active.values())
            return active_private_reservation + cache_block_count()

        def physical_used_blocks() -> int:
            if mode == "contiguous":
                return committed_blocks()
            return config.pool_blocks - len(free_blocks)

        def evict_idle_prefixes(required_headroom: int, protected_prefix: Optional[str]) -> bool:
            """Evict LRU prefixes only when no active request references them."""
            nonlocal cached_prefix_evictions
            while required_headroom > config.pool_blocks - active_private_reservation - cache_block_count():
                candidates = [
                    (entry.last_used_step, prefix_id, entry)
                    for prefix_id, entry in prefixes.items()
                    if entry.active_references == 0 and prefix_id != protected_prefix
                ]
                if not candidates:
                    return False
                _, prefix_id, entry = min(candidates, key=lambda item: (item[0], item[1]))
                self._release_blocks(free_blocks, entry.block_ids)
                del prefixes[prefix_id]
                cached_prefix_evictions += 1
            return True

        def can_admit(request: KVRequest) -> bool:
            if config.max_active_requests is not None and len(active) >= config.max_active_requests:
                return False
            if mode == "contiguous":
                reserve = _ceil_div(config.max_model_tokens, config.block_size_tokens)
                return committed_blocks() + reserve <= config.pool_blocks

            private_blocks = self._max_private_blocks(request, config, enable_prefix_caching)
            prefix_blocks = 0
            protected_prefix: Optional[str] = None
            if enable_prefix_caching and request.prefix_id:
                protected_prefix = request.prefix_id
                if request.prefix_id not in prefixes:
                    prefix_blocks = _ceil_div(request.prefix_tokens, config.block_size_tokens)
            return evict_idle_prefixes(private_blocks + prefix_blocks, protected_prefix)

        def admit(request: KVRequest) -> _RequestState:
            nonlocal active_private_reservation, reused_prefix_blocks
            if mode == "contiguous":
                reserve = _ceil_div(config.max_model_tokens, config.block_size_tokens)
                state = _RequestState(
                    request=request,
                    admitted_step=current_step,
                    reserved_private_blocks=reserve,
                    prefix_entry=None,
                )
                active_private_reservation += reserve
                return state

            prefix_entry: Optional[_PrefixEntry] = None
            if enable_prefix_caching and request.prefix_id:
                prefix_entry = prefixes.get(request.prefix_id)
                if prefix_entry is None:
                    prefix_blocks = _ceil_div(request.prefix_tokens, config.block_size_tokens)
                    allocated = [free_blocks.pop(0) for _ in range(prefix_blocks)]
                    prefix_entry = _PrefixEntry(block_ids=allocated, last_used_step=current_step)
                    prefixes[request.prefix_id] = prefix_entry
                else:
                    reused_prefix_blocks += len(prefix_entry.block_ids)
                prefix_entry.active_references += 1
                prefix_entry.last_used_step = current_step

            reserve = self._max_private_blocks(request, config, enable_prefix_caching)
            active_private_reservation += reserve
            return _RequestState(
                request=request,
                admitted_step=current_step,
                reserved_private_blocks=reserve,
                prefix_entry=prefix_entry,
            )

        def grow_private_blocks(state: _RequestState, private_tokens: int) -> None:
            if mode == "contiguous":
                return
            required = _ceil_div(private_tokens, config.block_size_tokens)
            missing = required - len(state.private_block_ids)
            if missing <= 0:
                return
            if missing > len(free_blocks):
                raise RuntimeError("admission commitment invariant was violated")
            state.private_block_ids.extend(free_blocks.pop(0) for _ in range(missing))

        def release(state: _RequestState) -> None:
            nonlocal active_private_reservation
            active_private_reservation -= state.reserved_private_blocks
            if mode == "paged":
                self._release_blocks(free_blocks, state.private_block_ids)
                if state.prefix_entry is not None:
                    state.prefix_entry.active_references -= 1
                    state.prefix_entry.last_used_step = current_step

        while arrivals_index < len(trace) or waiting or active:
            while arrivals_index < len(trace) and trace[arrivals_index].arrival_step <= current_step:
                waiting.append(trace[arrivals_index])
                arrivals_index += 1

            while waiting and can_admit(waiting[0]):
                request = waiting.pop(0)
                state = admit(request)
                active[request.request_id] = state
                queue_delays.append(current_step - request.arrival_step)

            complete_ids: List[str] = []
            for request_id, state in list(active.items()):
                request = state.request
                private_prompt = self._private_prompt_tokens(request, enable_prefix_caching and mode == "paged")
                if state.private_prefill_tokens < private_prompt:
                    next_prefill = min(
                        private_prompt,
                        state.private_prefill_tokens + config.prefill_tokens_per_step,
                    )
                    grow_private_blocks(state, next_prefill)
                    state.private_prefill_tokens = next_prefill
                    continue

                if state.generated_tokens < request.output_tokens:
                    state.generated_tokens += 1
                    private_tokens = private_prompt + state.generated_tokens
                    grow_private_blocks(state, private_tokens)
                    total_output_tokens += 1
                    if state.first_token_step is None:
                        state.first_token_step = current_step

                if state.generated_tokens == request.output_tokens:
                    complete_ids.append(request_id)

            for request_id in complete_ids:
                state = active.pop(request_id)
                if state.first_token_step is not None:
                    ttft.append(state.first_token_step - state.request.arrival_step)
                end_to_end.append(current_step - state.request.arrival_step + 1)
                release(state)

            used = physical_used_blocks()
            committed = committed_blocks()
            used_blocks_samples.append(used)
            committed_blocks_samples.append(committed)
            peak_active = max(peak_active, len(active))
            peak_committed = max(peak_committed, committed)
            final_step = current_step

            if not active and not waiting and arrivals_index < len(trace):
                current_step = trace[arrivals_index].arrival_step
            else:
                current_step += 1

        elapsed = final_step - trace[0].arrival_step + 1
        mean_used = sum(used_blocks_samples) / len(used_blocks_samples)
        mean_committed = sum(committed_blocks_samples) / len(committed_blocks_samples)
        return SimulationResult(
            mode=mode,
            prefix_caching=enable_prefix_caching,
            requests_completed=len(end_to_end),
            total_output_tokens=total_output_tokens,
            elapsed_steps=elapsed,
            throughput_output_tokens_per_step=total_output_tokens / elapsed if elapsed else 0.0,
            mean_queue_delay_steps=sum(queue_delays) / len(queue_delays) if queue_delays else 0.0,
            p95_queue_delay_steps=_percentile(queue_delays, 95.0),
            mean_ttft_steps=sum(ttft) / len(ttft) if ttft else 0.0,
            p95_ttft_steps=_percentile(ttft, 95.0),
            mean_end_to_end_latency_steps=sum(end_to_end) / len(end_to_end) if end_to_end else 0.0,
            p95_end_to_end_latency_steps=_percentile(end_to_end, 95.0),
            mean_pool_utilization_pct=100.0 * mean_used / config.pool_blocks,
            peak_pool_utilization_pct=100.0 * max(used_blocks_samples) / config.pool_blocks,
            mean_committed_capacity_pct=100.0 * mean_committed / config.pool_blocks,
            peak_active_requests=peak_active,
            peak_committed_blocks=peak_committed,
            cached_prefix_evictions=cached_prefix_evictions,
            reused_prefix_blocks=reused_prefix_blocks,
        )

    def make_synthetic_requests(
        self,
        num_requests: int,
        seed: int,
        prefix_reuse: bool = False,
    ) -> List[KVRequest]:
        """Create a deterministic synthetic trace for unit-scale simulation only."""
        if num_requests <= 0:
            raise ValueError("num_requests must be positive")
        rng = random.Random(seed)
        requests: List[KVRequest] = []
        prefix_count = 8
        prefix_tokens = self.block_size * 16
        for index in range(num_requests):
            output_tokens = rng.randint(64, 256)
            if prefix_reuse:
                prefix_id = "prefix-%d" % (index % prefix_count)
                prompt_tokens = prefix_tokens + rng.randint(64, 256)
                requests.append(
                    KVRequest(
                        request_id="request-%04d" % index,
                        arrival_step=index // 5,
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        prefix_id=prefix_id,
                        prefix_tokens=prefix_tokens,
                    )
                )
            else:
                requests.append(
                    KVRequest(
                        request_id="request-%04d" % index,
                        arrival_step=index // 5,
                        prompt_tokens=rng.randint(128, 512),
                        output_tokens=output_tokens,
                    )
                )
        return requests

    def run_paging_benchmark(self, num_requests: int = 50, seed: int = 7) -> Dict[str, Any]:
        """Compare paged exact-demand admission with static contiguous reserve.

        This is a deterministic synthetic capacity model. It deliberately does
        not label any quantity as vLLM, GPU throughput, or DRAM page faults.
        """
        config = PagingConfig(
            block_size_tokens=self.block_size,
            pool_blocks=self.total_blocks,
            max_model_tokens=1024,
            prefill_tokens_per_step=128,
        )
        trace = self.make_synthetic_requests(num_requests=num_requests, seed=seed)
        paged = self.simulate(trace, config, mode="paged")
        contiguous = self.simulate(trace, config, mode="contiguous")
        latency_reduction = 100.0 * (
            1.0 - paged.p95_end_to_end_latency_steps / contiguous.p95_end_to_end_latency_steps
        ) if contiguous.p95_end_to_end_latency_steps else 0.0
        return {
            "scenario": "synthetic-capacity-trace-no-prefix-reuse",
            "seed": seed,
            "num_requests": num_requests,
            "block_size_tokens": self.block_size,
            "kv_block_pool": self.total_blocks,
            "paged": paged.to_dict(),
            "contiguous": contiguous.to_dict(),
            "throughput_improvement_factor": (
                paged.throughput_output_tokens_per_step / contiguous.throughput_output_tokens_per_step
                if contiguous.throughput_output_tokens_per_step
                else 0.0
            ),
            "p95_end_to_end_latency_reduction_pct": latency_reduction,
        }

    def run_prefix_reuse_benchmark(self, num_requests: int = 50, seed: int = 7) -> Dict[str, Any]:
        """Isolate prefix reuse against the same paged allocator and trace."""
        config = PagingConfig(
            block_size_tokens=self.block_size,
            pool_blocks=self.total_blocks,
            max_model_tokens=1024,
            prefill_tokens_per_step=128,
        )
        trace = self.make_synthetic_requests(num_requests=num_requests, seed=seed, prefix_reuse=True)
        without_prefix_cache = self.simulate(trace, config, mode="paged", enable_prefix_caching=False)
        with_prefix_cache = self.simulate(trace, config, mode="paged", enable_prefix_caching=True)
        return {
            "scenario": "synthetic-block-aligned-prefix-reuse",
            "seed": seed,
            "num_requests": num_requests,
            "block_size_tokens": self.block_size,
            "kv_block_pool": self.total_blocks,
            "paged_without_prefix_cache": without_prefix_cache.to_dict(),
            "paged_with_prefix_cache": with_prefix_cache.to_dict(),
        }


if __name__ == "__main__":
    scheduler = LLMKVCacheScheduler()
    print(json.dumps(scheduler.run_paging_benchmark(), indent=2, sort_keys=True))
