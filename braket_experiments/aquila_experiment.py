"""QuEra Aquila analog Ising experiment.

A three-atom Rydberg chain is driven through a detuning sweep. This is an
exploration of the analog device as a sampler for the repository's
three-variable reference Ising model, whose coupling graph is a chain.
It is explicitly NOT an exact Ising embedding: the Rydberg interaction is
always positive and distance dependent, while the reference model has
mixed-sign couplings. The measured atom configurations are therefore
evaluated on the original model's energy landscape and compared with the
classical ground state and with the gate-model results. Raw counts stay
in the ledger for auditability.

Measurement convention: site i maps to bit i of ``int(bitstring, 2)``
(leftmost character is the highest site index), matching the gate-model
convention; the raw counts make the assumption auditable.
"""

from . import attribution  # noqa: F401  (attribution must precede SDK use)

import time

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.aws import AwsDevice, AwsQuantumTask

from . import ledger
from .circuits import energy_stats
from .instances import reference_chain_model

AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


def build_chain_program(
    num_atoms: int = 3,
    spacing_m: float = 6.0e-6,
    rabi_max_rad_s: float = 5.0e6,
    detuning_min_rad_s: float = -12.0e6,
    detuning_max_rad_s: float = 14.0e6,
    duration_s: float = 2.0e-6,
) -> AnalogHamiltonianSimulation:
    """Three-atom chain swept through the Rydberg transition."""

    if num_atoms < 2:
        raise ValueError("num_atoms must be >= 2")
    register = AtomArrangement()
    for index in range(num_atoms):
        register.add((index * spacing_m, 0.0))
    quarter = 0.25 * duration_s
    times = [0.0, quarter, duration_s - quarter, duration_s]
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


def submit_aquila(shots: int = 200, **program_kwargs) -> str:
    """Submit the AHS program and record it in the ledger."""

    program = build_chain_program(**program_kwargs)
    device = AwsDevice(AQUILA_ARN)
    task = device.run(program, shots=shots)
    ledger.record_task(
        {
            "task_arn": task.id,
            "device_key": "aquila",
            "device_arn": AQUILA_ARN,
            "kind": "ahs",
            "instance": "chain3",
            "depth": 0,
            "shots": shots,
            "program": {
                "num_atoms": program_kwargs.get("num_atoms", 3),
                "spacing_m": program_kwargs.get("spacing_m", 6.0e-6),
                "rabi_max_rad_s": program_kwargs.get("rabi_max_rad_s", 5.0e6),
                "detuning_min_rad_s": program_kwargs.get("detuning_min_rad_s", -12.0e6),
                "detuning_max_rad_s": program_kwargs.get("detuning_max_rad_s", 14.0e6),
                "duration_s": program_kwargs.get("duration_s", 2.0e-6),
            },
            "status": "SUBMITTED",
        }
    )
    return task.id


def collect_aquila(task_arn: str, max_wait_seconds: float = 600.0, poll_seconds: float = 10.0) -> dict:
    """Poll and evaluate the AHS task (kept for direct use; runner.collect
    handles ledger entries generically)."""

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
    counts = task.result().get_counts()
    solver = _energy_solver()
    stats = energy_stats(solver, counts)
    ledger.update_task(task_arn, {"status": "COLLECTED", "stats": stats, "counts": dict(counts)})
    return {"task_arn": task_arn, "status": "COLLECTED", "stats": stats}


def _energy_solver():
    from QAOA_solver import QAOASolver

    return QAOASolver(reference_chain_model(), p=1)
