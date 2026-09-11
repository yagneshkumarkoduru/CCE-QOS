"""Tests for the deterministic KV-cache capacity simulator.

The simulator models safe GPU admission and completion freeing. It must not
invent host-DRAM page faults for active KV blocks, because ordinary vLLM GPU
serving queues work when the cache is full instead of paging active sequences.
"""

import pytest

from implementations.v3_llm_kvcache_continuous_batching.llm_kvcache_paging_scheduler import (
    KVRequest,
    LLMKVCacheScheduler,
    PagingConfig,
)


def test_paged_exact_demand_beats_static_reservation_under_burst_pressure():
    """Exact-length paged admission should reduce queueing versus max-length reserve."""
    scheduler = LLMKVCacheScheduler(block_size_tokens=16, num_gpu_blocks=1600)
    result = scheduler.run_paging_benchmark(num_requests=50, seed=7)
    paged = result["paged"]
    contiguous = result["contiguous"]

    assert result["num_requests"] == 50
    assert paged["requests_completed"] == 50
    assert contiguous["requests_completed"] == 50
    assert paged["total_output_tokens"] == contiguous["total_output_tokens"]
    assert paged["mean_queue_delay_steps"] < contiguous["mean_queue_delay_steps"]
    assert paged["throughput_output_tokens_per_step"] > contiguous["throughput_output_tokens_per_step"]
    assert result["throughput_improvement_factor"] > 1.0


def test_prefix_reuse_reduces_committed_capacity_without_losing_requests():
    """Only complete, block-aligned prefixes are shared and all work completes."""
    scheduler = LLMKVCacheScheduler(block_size_tokens=16, num_gpu_blocks=1600)
    result = scheduler.run_prefix_reuse_benchmark(num_requests=50, seed=7)
    without_cache = result["paged_without_prefix_cache"]
    with_cache = result["paged_with_prefix_cache"]

    assert with_cache["requests_completed"] == 50
    assert with_cache["total_output_tokens"] == without_cache["total_output_tokens"]
    assert with_cache["reused_prefix_blocks"] > 0
    assert with_cache["peak_committed_blocks"] < without_cache["peak_committed_blocks"]
    assert with_cache["mean_ttft_steps"] < without_cache["mean_ttft_steps"]


def test_completion_frees_capacity_for_waiting_requests():
    """A small pool must queue safely and subsequently release completed KV blocks."""
    scheduler = LLMKVCacheScheduler(block_size_tokens=16, num_gpu_blocks=4)
    config = PagingConfig(
        block_size_tokens=16,
        pool_blocks=4,
        max_model_tokens=64,
        prefill_tokens_per_step=16,
    )
    trace = [
        KVRequest("a", 0, prompt_tokens=16, output_tokens=16),
        KVRequest("b", 0, prompt_tokens=16, output_tokens=16),
        KVRequest("c", 0, prompt_tokens=16, output_tokens=16),
    ]

    result = scheduler.simulate(trace, config, mode="paged")

    assert result.requests_completed == 3
    assert result.peak_committed_blocks <= config.pool_blocks
    assert result.mean_queue_delay_steps > 0.0
    assert result.peak_pool_utilization_pct <= 100.0


def test_prefix_reuse_requires_block_alignment():
    """Partial prefix blocks need copy-on-write and are rejected by this model."""
    scheduler = LLMKVCacheScheduler(block_size_tokens=16, num_gpu_blocks=16)
    config = PagingConfig(block_size_tokens=16, pool_blocks=16, max_model_tokens=64)
    trace = [KVRequest("bad", 0, 32, 8, prefix_id="p", prefix_tokens=17)]

    with pytest.raises(ValueError, match="block-aligned"):
        scheduler.simulate(trace, config, mode="paged", enable_prefix_caching=True)


def test_replay_is_deterministic():
    """The same synthetic trace and configuration produce identical metrics."""
    scheduler = LLMKVCacheScheduler()
    first = scheduler.run_paging_benchmark(num_requests=20, seed=3)
    second = scheduler.run_paging_benchmark(num_requests=20, seed=3)
    assert first == second
