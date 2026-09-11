"""Tests for the dependency-free vLLM benchmark trace and dispatch protocol."""

import json

import pytest

from benchmarks.vllm.benchmark_protocol import (
    BenchmarkRequest,
    BoundedDispatchQueue,
    load_trace,
    trace_sha256,
    write_trace,
)
from benchmarks.vllm.make_prefix_trace import build_trace


def test_trace_round_trip_has_stable_logical_hash(tmp_path):
    requests = [
        BenchmarkRequest("b", 10, "prompt b", 8, "group-b"),
        BenchmarkRequest("a", 0, "prompt a", 8, "group-a"),
    ]
    path = tmp_path / "trace.jsonl"

    digest = write_trace(path, requests)
    reloaded = load_trace(path)

    assert [request.request_id for request in reloaded] == ["a", "b"]
    assert digest == trace_sha256(reloaded)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["request_id"] == "a"


def test_prefix_bounded_dispatch_reuses_last_prefix_when_fair():
    queue = BoundedDispatchQueue(policy="prefix_bounded", max_reorder_wait_ms=100)
    first = BenchmarkRequest("a", 0, "a", 4, "red")
    other = BenchmarkRequest("b", 0, "b", 4, "blue")
    matching = BenchmarkRequest("c", 1, "c", 4, "red")

    queue.push(first)
    assert queue.pop_next(0).request_id == "a"
    queue.push(other)
    queue.push(matching)

    assert queue.pop_next(1).request_id == "c"
    assert queue.pop_next(1).request_id == "b"


def test_prefix_bounded_dispatch_never_skips_an_overdue_request():
    queue = BoundedDispatchQueue(policy="prefix_bounded", max_reorder_wait_ms=20)
    queue.push(BenchmarkRequest("a", 0, "a", 4, "red"))
    assert queue.pop_next(0).request_id == "a"
    queue.push(BenchmarkRequest("b", 0, "b", 4, "blue"))
    queue.push(BenchmarkRequest("c", 5, "c", 4, "red"))

    assert queue.pop_next(21).request_id == "b"


def test_trace_generator_is_deterministic_and_explicitly_grouped():
    first = build_trace(12, 3, 10, 8, 7)
    second = build_trace(12, 3, 10, 8, 7)

    assert first == second
    assert {request.prefix_id for request in first} == {"group-00", "group-01", "group-02"}
    assert all("bounded technical analysis" in request.prompt for request in first)


def test_trace_loader_rejects_unrecognized_fields(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"request_id":"a","arrival_offset_ms":0,"prompt":"p","max_tokens":4,"unknown":1}\n')

    with pytest.raises(ValueError, match="unknown trace fields"):
        load_trace(path)
