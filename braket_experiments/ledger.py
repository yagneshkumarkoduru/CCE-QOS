"""Task ledger for Braket submissions.

Every submitted task is recorded immediately so a submission is never
lost if a session is interrupted: task ARN, device, instance, depth,
shots, angles, and status. The collect step updates entries with measured
results.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "quantum_braket"
LEDGER_PATH = RESULTS_DIR / "task_ledger.json"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        return {"tasks": []}
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def save_ledger(ledger: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def record_task(entry: dict) -> dict:
    ledger = load_ledger()
    entry = {**entry, "submitted_at": _now()}
    ledger["tasks"].append(entry)
    save_ledger(ledger)
    return entry


def update_task(task_arn: str, fields: dict) -> dict:
    ledger = load_ledger()
    for entry in ledger["tasks"]:
        if entry.get("task_arn") == task_arn:
            entry.update(fields)
            save_ledger(ledger)
            return entry
    raise KeyError(f"task {task_arn} not found in the ledger")


def find_task(task_arn: str) -> dict | None:
    for entry in load_ledger()["tasks"]:
        if entry.get("task_arn") == task_arn:
            return entry
    return None


def pending_tasks() -> list[dict]:
    return [
        entry
        for entry in load_ledger()["tasks"]
        if entry.get("status") == "SUBMITTED"
    ]


def has_submission(device_key: str, instance: str, depth: int, shots: int) -> bool:
    """True when this exact configuration was already submitted."""

    for entry in load_ledger()["tasks"]:
        if (
            entry.get("device_key") == device_key
            and entry.get("instance") == instance
            and entry.get("depth") == depth
            and entry.get("shots") == shots
        ):
            return True
    return False


def has_ahs_submission(instance: str, schedule: str, shots: int) -> bool:
    """True when this exact Aquila program configuration was submitted."""

    for entry in ledger_tasks():
        if (
            entry.get("device_key") == "aquila"
            and entry.get("instance") == instance
            and entry.get("schedule") == schedule
            and entry.get("shots") == shots
        ):
            return True
    return False


def ledger_tasks() -> list[dict]:
    return load_ledger()["tasks"]


def estimate_spend(entries: list[dict]) -> float:
    """Estimated QPU spend in USD from the published per-task/per-shot rates.

    IonQ Forte Enterprise 1: $0.30/task + $0.08/shot.
    QuEra Aquila: $0.30/task + $0.01/shot.
    Managed simulators bill per minute and are not estimated here.
    """

    total = 0.0
    for entry in entries:
        shots = int(entry.get("shots", 0))
        if entry.get("device_key") == "ionq":
            total += 0.30 + 0.08 * shots
        elif entry.get("device_key") == "aquila":
            total += 0.30 + 0.01 * shots
    return total
