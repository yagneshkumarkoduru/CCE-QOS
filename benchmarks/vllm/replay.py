"""Asynchronous, bounded-fairness client for a local vLLM OpenAI server.

The client measures request time from the trace's intended arrival timestamp,
so policy queueing is included in end-to-end latency. It streams completions
to capture time-to-first-token (TTFT) and writes a JSON record for every
request, including explicit errors rather than dropping failed work.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .benchmark_protocol import BenchmarkRequest, BoundedDispatchQueue, trace_sha256, validate_trace
except ImportError:  # Direct script execution from this directory.
    from benchmark_protocol import BenchmarkRequest, BoundedDispatchQueue, trace_sha256, validate_trace


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass
class _RequestOutcome:
    request: BenchmarkRequest
    dispatched_ms: float
    first_token_ms: Optional[float]
    completed_ms: float
    output_tokens: Optional[int]
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        arrival = float(self.request.arrival_offset_ms)
        return {
            "request_id": self.request.request_id,
            "prefix_id": self.request.prefix_id,
            "arrival_offset_ms": self.request.arrival_offset_ms,
            "dispatched_ms": round(self.dispatched_ms, 3),
            "queue_delay_ms": round(max(0.0, self.dispatched_ms - arrival), 3),
            "first_token_ms": round(self.first_token_ms, 3) if self.first_token_ms is not None else None,
            "ttft_ms": round(max(0.0, self.first_token_ms - arrival), 3)
            if self.first_token_ms is not None
            else None,
            "completed_ms": round(self.completed_ms, 3),
            "end_to_end_latency_ms": round(max(0.0, self.completed_ms - arrival), 3),
            "output_tokens": self.output_tokens,
            "status": self.status,
            "error": self.error,
        }


async def _stream_completion(
    client: Any,
    endpoint: str,
    model: str,
    request: BenchmarkRequest,
    dispatched_ms: float,
    started_at: float,
    seed: int,
) -> _RequestOutcome:
    """Run one OpenAI-compatible streamed completion and retain only metrics."""
    payload = {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first_token_ms: Optional[float] = None
    output_tokens: Optional[int] = None
    try:
        async with client.stream("POST", endpoint.rstrip("/") + "/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices", [])
                if choices and choices[0].get("text") and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started_at) * 1000.0
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
    except Exception as exc:  # Network/server errors must remain in the record.
        completed_ms = (time.perf_counter() - started_at) * 1000.0
        return _RequestOutcome(
            request=request,
            dispatched_ms=dispatched_ms,
            first_token_ms=first_token_ms,
            completed_ms=completed_ms,
            output_tokens=output_tokens,
            status="error",
            error="%s: %s" % (type(exc).__name__, exc),
        )

    completed_ms = (time.perf_counter() - started_at) * 1000.0
    return _RequestOutcome(
        request=request,
        dispatched_ms=dispatched_ms,
        first_token_ms=first_token_ms,
        completed_ms=completed_ms,
        output_tokens=output_tokens,
        status="ok",
    )


def _summarize(
    outcomes: Iterable[_RequestOutcome],
    trace: List[BenchmarkRequest],
    policy: str,
    max_reorder_wait_ms: int,
    duration_ms: float,
) -> Dict[str, Any]:
    records = [outcome.to_dict() for outcome in outcomes]
    successes = [record for record in records if record["status"] == "ok"]
    queue_delays = [float(record["queue_delay_ms"]) for record in records]
    e2e = [float(record["end_to_end_latency_ms"]) for record in successes]
    ttft = [float(record["ttft_ms"]) for record in successes if record["ttft_ms"] is not None]
    known_tokens = [int(record["output_tokens"]) for record in successes if record["output_tokens"] is not None]
    all_usage_available = len(known_tokens) == len(successes)

    summary: Dict[str, Any] = {
        "workload_sha256": trace_sha256(trace),
        "workload_requests": len(trace),
        "dispatch_policy": policy,
        "max_reorder_wait_ms": max_reorder_wait_ms,
        "duration_ms": round(duration_ms, 3),
        "requests_succeeded": len(successes),
        "requests_failed": len(records) - len(successes),
        "mean_queue_delay_ms": round(sum(queue_delays) / len(queue_delays), 3) if queue_delays else 0.0,
        "p95_queue_delay_ms": round(_percentile(queue_delays, 95.0), 3),
        "mean_end_to_end_latency_ms": round(sum(e2e) / len(e2e), 3) if e2e else None,
        "p95_end_to_end_latency_ms": round(_percentile(e2e, 95.0), 3) if e2e else None,
        "mean_ttft_ms": round(sum(ttft) / len(ttft), 3) if ttft else None,
        "p95_ttft_ms": round(_percentile(ttft, 95.0), 3) if ttft else None,
        "completion_tokens_observed": sum(known_tokens) if all_usage_available else None,
        "output_tokens_per_second": round(sum(known_tokens) / (duration_ms / 1000.0), 3)
        if all_usage_available and duration_ms > 0
        else None,
        "usage_available_for_all_successes": all_usage_available,
        "request_records": records,
    }
    return summary


async def run_replay(
    endpoint: str,
    model: str,
    requests: Iterable[BenchmarkRequest],
    policy: str,
    max_inflight: int,
    max_reorder_wait_ms: int,
    request_timeout_s: float,
    seed: int = 20260911,
) -> Dict[str, Any]:
    """Replay a trace against one already-running vLLM server.

    The import stays local so protocol/unit tests run without HTTP dependencies.
    ``endpoint`` must be the server root, for example ``http://127.0.0.1:8000``.
    """
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required in the vLLM benchmark environment") from exc

    trace = validate_trace(requests)
    dispatcher = BoundedDispatchQueue(policy=policy, max_reorder_wait_ms=max_reorder_wait_ms)
    arrival_index = 0
    outcomes: List[_RequestOutcome] = []
    inflight: Dict[asyncio.Task[_RequestOutcome], BenchmarkRequest] = {}
    started_at = time.perf_counter()

    timeout = httpx.Timeout(timeout=request_timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while arrival_index < len(trace) or len(dispatcher) or inflight:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            while arrival_index < len(trace) and trace[arrival_index].arrival_offset_ms <= elapsed_ms:
                dispatcher.push(trace[arrival_index])
                arrival_index += 1

            while len(inflight) < max_inflight and len(dispatcher):
                dispatched_ms = (time.perf_counter() - started_at) * 1000.0
                request = dispatcher.pop_next(dispatched_ms)
                task = asyncio.create_task(
                    _stream_completion(client, endpoint, model, request, dispatched_ms, started_at, seed)
                )
                inflight[task] = request

            next_arrival_timeout: Optional[float] = None
            if arrival_index < len(trace):
                next_arrival_ms = trace[arrival_index].arrival_offset_ms
                current_ms = (time.perf_counter() - started_at) * 1000.0
                next_arrival_timeout = max(0.0, (next_arrival_ms - current_ms) / 1000.0)

            if inflight:
                timeout_seconds = next_arrival_timeout if next_arrival_timeout is not None else None
                completed, _ = await asyncio.wait(
                    inflight,
                    timeout=timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in completed:
                    inflight.pop(task)
                    outcomes.append(await task)
            elif next_arrival_timeout is not None:
                await asyncio.sleep(next_arrival_timeout)

    duration_ms = (time.perf_counter() - started_at) * 1000.0
    return _summarize(outcomes, trace, policy, max_reorder_wait_ms, duration_ms)


def write_result(path: Path, result: Dict[str, Any]) -> None:
    """Write one canonical, machine-readable run record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
