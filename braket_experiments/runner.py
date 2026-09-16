"""Braket device submission and result collection.

Device ARNs were verified live during the experiment session with the
Braket ``SearchDevices`` API (us-east-1). Costs follow the AWS Pricing API
values read in the same session: IonQ Forte Enterprise 1 charges
$0.30 per task plus $0.08 per shot; QuEra Aquila charges $0.30 per task
plus $0.01 per shot; managed simulators bill per minute of execution.
"""

from . import attribution  # noqa: F401  (attribution must precede SDK use)

import time

from braket.aws import AwsDevice, AwsQuantumTask
from braket.devices import LocalSimulator

from . import ledger
from .circuits import build_qaoa_circuit, energy_stats
from .instances import get_instance

DEVICE_ARNS = {
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "ionq": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
    "aquila": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
}

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


def run_local_simulator(solver, gamma, beta, shots: int = 4000) -> dict:
    """Exact local simulation (free) for pipeline validation."""

    circuit = build_qaoa_circuit(solver, gamma, beta)
    task = LocalSimulator().run(circuit, shots=shots)
    counts = task.result().measurement_counts
    return energy_stats(solver, counts)


def submit(device_key: str, instance: str, depth: int, solver, gamma, beta, shots: int) -> str:
    """Submit one QAOA task and record it in the ledger."""

    if device_key not in DEVICE_ARNS:
        raise ValueError(f"unknown device key {device_key!r}")
    device = AwsDevice(DEVICE_ARNS[device_key])
    circuit = build_qaoa_circuit(solver, gamma, beta)
    task = device.run(circuit, shots=shots)
    ledger.record_task(
        {
            "task_arn": task.id,
            "device_key": device_key,
            "device_arn": DEVICE_ARNS[device_key],
            "instance": instance,
            "depth": depth,
            "shots": shots,
            "gamma": [float(value) for value in gamma],
            "beta": [float(value) for value in beta],
            "status": "SUBMITTED",
        }
    )
    return task.id


def collect(task_arn: str, max_wait_seconds: float = 600.0, poll_seconds: float = 10.0) -> dict:
    """Poll one submitted task and, when finished, evaluate its energies."""

    entry = ledger.find_task(task_arn)
    if entry is None:
        raise KeyError(f"task {task_arn} not found in the ledger")
    task = AwsQuantumTask(task_arn)
    deadline = time.time() + max_wait_seconds
    state = task.state()
    while state not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(poll_seconds)
        state = task.state()
    if state not in TERMINAL_STATES:
        return {"task_arn": task_arn, "status": "STILL_QUEUED", "state": state}
    if state != "COMPLETED":
        metadata = task.metadata()
        ledger.update_task(
            task_arn,
            {"status": state, "failure_reason": metadata.get("failureReason")},
        )
        return {"task_arn": task_arn, "status": state}

    result = task.result()
    if entry.get("kind") == "ahs":
        counts = result.get_counts()
    else:
        counts = result.measurement_counts
    solver = _rebuild_solver(entry)
    stats = energy_stats(solver, counts)
    ledger.update_task(task_arn, {"status": "COLLECTED", "stats": stats})
    return {"task_arn": task_arn, "status": "COLLECTED", "stats": stats}


def collect_all(max_wait_seconds: float = 600.0, poll_seconds: float = 10.0) -> list[dict]:
    return [
        collect(entry["task_arn"], max_wait_seconds, poll_seconds)
        for entry in ledger.pending_tasks()
    ]


def task_states() -> list[dict]:
    """Non-blocking state check for every ledger entry."""

    rows = []
    for entry in ledger.ledger_tasks():
        state = "UNKNOWN"
        try:
            state = AwsQuantumTask(entry["task_arn"]).state()
        except Exception as exc:  # noqa: BLE001 - report, keep going
            state = f"ERROR: {exc}"
        rows.append({**entry, "aws_state": state})
    return rows


def sweep_collect(max_wait_seconds: float = 900.0, poll_seconds: float = 20.0) -> list[dict]:
    """Collect with one global deadline across all pending tasks.

    Every sweep checks task states with cheap API calls; terminal tasks are
    collected, and the loop exits when nothing is pending or the deadline
    is reached. No per-task blocking waits, so a large batch cannot stretch
    into hours of serial polling.
    """

    deadline = time.time() + max_wait_seconds
    summaries: list[dict] = []
    while True:
        pending = ledger.pending_tasks()
        if not pending:
            break
        collected_now = []
        for entry in pending:
            task = AwsQuantumTask(entry["task_arn"])
            state = task.state()
            if state in TERMINAL_STATES:
                collected_now.append(entry["task_arn"])
        for task_arn in collected_now:
            summaries.append(collect(task_arn, max_wait_seconds=5.0, poll_seconds=1.0))
        if time.time() >= deadline:
            break
        time.sleep(poll_seconds)
    return summaries


def _rebuild_solver(entry: dict):
    from QAOA_solver import QAOASolver

    return QAOASolver(get_instance(entry["instance"]), p=int(entry["depth"]))
