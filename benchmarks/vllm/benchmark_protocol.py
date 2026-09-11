"""Trace and bounded-dispatch primitives shared by the vLLM benchmark tools.

The module intentionally has no vLLM, HTTP, Torch, or GPU dependency. Unit
tests can therefore validate fairness and trace integrity on every platform.
The actual requester lives in :mod:`replay` and records measured timestamps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class BenchmarkRequest:
    """One immutable completion request and its intended arrival time."""

    request_id: str
    arrival_offset_ms: int
    prompt: str
    max_tokens: int
    prefix_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_offset_ms < 0:
            raise ValueError("arrival_offset_ms must be non-negative")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "BenchmarkRequest":
        allowed = {"request_id", "arrival_offset_ms", "prompt", "max_tokens", "prefix_id"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("unknown trace fields: %s" % sorted(unknown))
        return cls(
            request_id=str(value["request_id"]),
            arrival_offset_ms=int(value["arrival_offset_ms"]),
            prompt=str(value["prompt"]),
            max_tokens=int(value["max_tokens"]),
            prefix_id=(str(value["prefix_id"]) if value.get("prefix_id") is not None else None),
        )


def validate_trace(requests: Iterable[BenchmarkRequest]) -> List[BenchmarkRequest]:
    """Return a deterministic trace order and reject duplicate identifiers."""
    ordered = sorted(list(requests), key=lambda request: (request.arrival_offset_ms, request.request_id))
    identifiers = set()
    for request in ordered:
        if request.request_id in identifiers:
            raise ValueError("request IDs must be unique")
        identifiers.add(request.request_id)
    return ordered


def trace_sha256(requests: Iterable[BenchmarkRequest]) -> str:
    """Hash canonical request records so runs expose their exact workload."""
    canonical = [request.to_dict() for request in validate_trace(requests)]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_trace(path: Path, requests: Iterable[BenchmarkRequest]) -> str:
    """Write canonical JSONL and return its logical trace hash."""
    ordered = validate_trace(requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for request in ordered:
            handle.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")
    return trace_sha256(ordered)


def load_trace(path: Path) -> List[BenchmarkRequest]:
    """Load a canonical benchmark JSONL trace with line-specific errors."""
    requests: List[BenchmarkRequest] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at %s:%d: %s" % (path, line_number, exc)) from exc
            if not isinstance(decoded, dict):
                raise ValueError("trace row at %s:%d must be an object" % (path, line_number))
            try:
                requests.append(BenchmarkRequest.from_dict(decoded))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid trace row at %s:%d: %s" % (path, line_number, exc)) from exc
    return validate_trace(requests)


class BoundedDispatchQueue:
    """FIFO or prefix-local dispatch with a strict maximum queue-wait bound.

    A prefix-aware decision is permitted only while no queued request has
    waited for ``max_reorder_wait_ms``. Once any request reaches that bound,
    the oldest overdue request wins. This makes policy-induced waiting visible
    and bounded rather than silently cherry-picking a favourable order.
    """

    def __init__(self, policy: str, max_reorder_wait_ms: int) -> None:
        if policy not in {"fifo", "prefix_bounded"}:
            raise ValueError("policy must be 'fifo' or 'prefix_bounded'")
        if max_reorder_wait_ms < 0:
            raise ValueError("max_reorder_wait_ms must be non-negative")
        self.policy = policy
        self.max_reorder_wait_ms = max_reorder_wait_ms
        self._pending: List[BenchmarkRequest] = []
        self._last_prefix_id: Optional[str] = None

    def __len__(self) -> int:
        return len(self._pending)

    def push(self, request: BenchmarkRequest) -> None:
        self._pending.append(request)
        self._pending.sort(key=lambda item: (item.arrival_offset_ms, item.request_id))

    def pop_next(self, now_ms: float) -> BenchmarkRequest:
        if not self._pending:
            raise IndexError("cannot dispatch an empty queue")

        selected_index = 0
        if self.policy == "prefix_bounded":
            overdue = [
                (index, request)
                for index, request in enumerate(self._pending)
                if now_ms - request.arrival_offset_ms >= self.max_reorder_wait_ms
            ]
            if overdue:
                selected_index = min(overdue, key=lambda item: (item[1].arrival_offset_ms, item[1].request_id))[0]
            elif self._last_prefix_id is not None:
                matching = [
                    (index, request)
                    for index, request in enumerate(self._pending)
                    if request.prefix_id == self._last_prefix_id
                ]
                if matching:
                    selected_index = matching[0][0]

        selected = self._pending.pop(selected_index)
        self._last_prefix_id = selected.prefix_id
        return selected

    def snapshot(self) -> Sequence[BenchmarkRequest]:
        return tuple(self._pending)
