"""QuEra Aquila analog Ising experiment.

Atoms in a chain are driven through a Rydberg detuning sweep. This is an
exploration of the analog device as a sampler for the repository's chain
Ising models, whose coupling graphs are chains. It is explicitly NOT an
exact Ising embedding: the Rydberg interaction is always positive and
distance dependent, while the models have mixed-sign couplings. The
measured atom configurations are therefore evaluated on the original
models' energy landscapes and compared with the gate-model results. Raw
counts stay in the ledger for auditability.

Measurement convention: site i maps to bit i of ``int(bitstring[::-1], 2)``
(the same convention verified for the gate-model devices); the raw counts
make the assumption auditable.

Aquila supports at most 1000 shots per quantum task.
"""

from . import attribution  # noqa: F401  (attribution must precede SDK use)

import time

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.aws import AwsDevice, AwsQuantumTask

from . import ledger
from .circuits import energy_stats
from .instances import get_instance

AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}
MAX_SHOTS = 1000

SCHEDULES = {
    "slow_sweep": {
        "rabi_max_rad_s": 5.0e6,
        "detuning_min_rad_s": -12.0e6,
        "detuning_max_rad_s": 14.0e6,
        "duration_s": 2.0e-6,
    },
    "fast_sweep": {
        "rabi_max_rad_s": 8.0e6,
        "detuning_min_rad_s": -16.0e6,
        "detuning_max_rad_s": 16.0e6,
        "duration_s": 1.5e-6,
    },
    "soft_drive": {
        "rabi_max_rad_s": 3.0e6,
        "detuning_min_rad_s": -10.0e6,
        "detuning_max_rad_s": 10.0e6,
        "duration_s": 2.5e-6,
    },
}


def _snap_to_nanoseconds(value_s: float) -> float:
    """Snap a time value to an integer nanosecond (Braket AHS requirement)."""

    return round(value_s / 1e-9) * 1e-9


def build_chain_program(
    num_atoms: int = 3,
    spacing_m: float = 6.0e-6,
    rabi_max_rad_s: float = 5.0e6,
    detuning_min_rad_s: float = -12.0e6,
    detuning_max_rad_s: float = 14.0e6,
    duration_s: float = 2.0e-6,
) -> AnalogHamiltonianSimulation:
    """Chain of atoms swept through the Rydberg transition."""

    if num_atoms < 2:
        raise ValueError("num_atoms must be >= 2")
    register = AtomArrangement()
    for index in range(num_atoms):
        register.add((index * spacing_m, 0.0))
    quarter = 0.25 * duration_s
    times = [_snap_to_nanoseconds(t) for t in (0.0, quarter, duration_s - quarter, duration_s)]
    amplitudes = [0.0, rabi_max_rad_s, rabi_max_rad_s, 0.0]
    detunings = [
        detuning_min_rad_s,
        0.5 * detuning_min_rad_s,
        0.5 * detuning_max_rad_s,
        detuning_max_rad_s,
    ]
    phases = [0.0, 0.0, 0.0, 0.0]
    drive = DrivingField.from_lists(times, amplitudes, detunings, phases)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def submit_aquila(
    shots: int = 1000,
    num_atoms: int = 3,
    schedule: str = "slow_sweep",
    spacing_m: float = 6.0e-6,
) -> str:
    """Submit one AHS program and record it in the ledger."""

    if shots > MAX_SHOTS:
        raise ValueError(f"Aquila supports at most {MAX_SHOTS} shots per task")
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}")
    instance = f"chain{num_atoms}"
    get_instance(instance)  # fail fast if no matching Ising model exists
    program_params = {"num_atoms": num_atoms, "spacing_m": spacing_m, **SCHEDULES[schedule]}
    program = build_chain_program(**program_params)
    device = AwsDevice(AQUILA_ARN)
    task = device.run(program, shots=shots)
    ledger.record_task(
        {
            "task_arn": task.id,
            "device_key": "aquila",
            "device_arn": AQUILA_ARN,
            "kind": "ahs",
            "instance": instance,
            "depth": 0,
            "shots": shots,
            "schedule": schedule,
            "program": program_params,
            "status": "SUBMITTED",
        }
    )
    return task.id


def collect_aquila(task_arn: str, max_wait_seconds: float = 600.0, poll_seconds: float = 10.0) -> dict:
    """Poll and evaluate one AHS task (runner.collect also handles these)."""

    task = AwsQuantumTask(task_arn)
    deadline = time.time() + max_wait_seconds
    state = task.state()
    while state not in TERMINAL_STATES and time.time() < deadline:
        time.sleep(poll_seconds)
        state = task.state()
    if state not in TERMINAL_STATES:
        return {"task_arn": task_arn, "status": "STILL_QUEUED", "state": state}
    if state != "COMPLETED":
        ledger.update_task(task_arn, {"status": state})
        return {"task_arn": task_arn, "status": state}
    entry = ledger.find_task(task_arn) or {}
    counts = task.result().get_counts()
    solver = _energy_solver(entry.get("instance", "chain3"))
    stats = energy_stats(solver, counts)
    ledger.update_task(task_arn, {"status": "COLLECTED", "stats": stats, "counts": dict(counts)})
    return {"task_arn": task_arn, "status": "COLLECTED", "stats": stats}


def _energy_solver(instance: str):
    from QAOA_solver import QAOASolver

    return QAOASolver(get_instance(instance), p=1)
