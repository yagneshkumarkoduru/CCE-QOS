# CCE-QOS on Amazon Braket

Version: 2026-09-15
Author: Koduru Yagnesh Kumar, Independent Researcher

## Purpose

The QAOA pipeline in this repository previously ran only as local
statevector simulation. This module executes the same optimized circuits
on Amazon Braket managed simulators and prepares physical runs on an IonQ
gate QPU (Forte Enterprise 1) and the QuEra Aquila analog Rydberg device.
Task ARNs are retained in a ledger under `results/quantum_braket/` as
citable evidence.

## Instances

| Instance | Variables | Definition |
|---|---|---|
| `chain3` | 3 | Repository reference model: h={0.5,-0.8,0.3}, J={(0,1):-1.2,(1,2):0.9}, offset=2.0 |
| `random11` | 11 | Deterministic all-to-all instance, `np.random.default_rng(42)`, h in U(-0.5,0.5), J in U(-1,1) |

Angles are optimized locally with the repository's seeded COBYLA solver
(seed 12345, 60 evaluations per depth) and then frozen for all devices.

## Devices

Verified online in us-east-1 during the session (`SearchDevices`):

| Device | ARN | Status |
|---|---|---|
| SV1 simulator | `arn:aws:braket:::device/quantum-simulator/amazon/sv1` | ONLINE |
| dm1 simulator | `arn:aws:braket:::device/quantum-simulator/amazon/dm1` | ONLINE |
| IonQ Forte Enterprise 1 | `arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1` | ONLINE |
| QuEra Aquila | `arn:aws:braket:us-east-1::device/qpu/quera/Aquila` | ONLINE |

Costs read from the AWS Pricing API (us-east-1, this session): IonQ Forte
Enterprise 1 = $0.30 per task + $0.08 per shot; QuEra Aquila = $0.30 per
task + $0.01 per shot; managed simulators bill per minute of execution.

## Method

1. `braket_experiments.circuits.build_qaoa_circuit` constructs the exact
   QAOA unitary, including the single-qubit Z rotations for the linear h
   terms that the legacy OpenQASM 2 export omitted (that export is now
   fixed in `QAOA_solver.to_openqasm`).
2. Circuit parity is enforced by tests: the Braket local statevector
   expectation equals `QAOASolver.compute_expectation` to 1e-8, and the
   measurement bitstring convention is locked by a test.
3. Bit-order convention (verified empirically, not assumed): qubit 0 is
   the leftmost character (`X(0)` on two qubits returns `'10'` on both the
   local simulator and managed SV1; SV1 calibration task ARN
   `.../81df2400-c8ad-4f5b-92b6-a45ad4c91d51`). Energies are evaluated
   with the string reversed accordingly.
4. Submissions are written to a JSON ledger immediately; `collect` polls
   and evaluates them, so no task is lost across sessions.

## Results (managed simulator, 4000 shots)

| Instance | p | Device | Mean energy | Best energy | Ratio (mean) | Ratio (best) | Ground prob | Task (suffix) |
|---|---|---|---|---|---|---|---|---|
| chain3 | 1 | local-sim | 1.1605 | -0.7000 | 0.6792 | 1.0000 | 0.2880 | - |
| chain3 | 2 | local-sim | 0.4326 | -0.7000 | 0.8047 | 1.0000 | 0.3802 | - |
| chain3 | 3 | local-sim | -0.3458 | -0.7000 | 0.9389 | 1.0000 | 0.8367 | - |
| chain3 | 1 | sv1 | 1.1644 | -0.7000 | 0.6786 | 1.0000 | 0.2920 | ...25af42a06109 |
| chain3 | 2 | sv1 | 0.4448 | -0.7000 | 0.8026 | 1.0000 | 0.3698 | ...7e5363e964b5 |
| chain3 | 3 | sv1 | -0.3401 | -0.7000 | 0.9379 | 1.0000 | 0.8365 | ...f23383f5da51 |
| random11 | 1 | local-sim | -0.0678 | -10.6412 | 0.5365 | 0.9887 | 0.0000 | - |
| random11 | 2 | local-sim | -5.9065 | -10.9051 | 0.7862 | 1.0000 | 0.0080 | - |
| random11 | 3 | local-sim | -0.3385 | -10.7108 | 0.5481 | 0.9917 | 0.0000 | - |
| random11 | 1 | sv1 | -0.1016 | -10.6412 | 0.5380 | 0.9887 | 0.0000 | ...f1eb364f5010 |
| random11 | 2 | sv1 | -5.8761 | -10.9051 | 0.7849 | 1.0000 | 0.0075 | ...b92626c8da5a |
| random11 | 3 | sv1 | -0.3288 | -10.9051 | 0.5477 | 1.0000 | 0.0003 | ...37f765996315 |

The managed simulator reproduces the local statevector expectation to
shot-noise level on every instance and depth, confirming that the
OpenQASM/circuit path executes correctly on the service.

Honest notes: at p=3 the 60-evaluation COBYLA budget is the limiting
factor for random11 (ratio 0.548 < p=2), and the same is visible in the
local column; a larger optimization budget is future work.

## QPU runs (pending user agreement)

QPU task creation currently returns ``User agreement has not been
accepted for 969739653654``. Accept the Braket user agreement once at

    https://us-east-1.console.aws.amazon.com/braket/home?region=us-east-1#/permissions?tab=general

then run:

```powershell
python -m braket_experiments.run_experiments submit --device ionq --shots 100 --depths 1 2
python -m braket_experiments.run_experiments aquila --shots 200
python -m braket_experiments.run_experiments collect --max-wait 3600
python -m braket_experiments.run_experiments report
```

All submissions are ledgered first, so no task can be lost. The Aquila
program is an exploratory analog sampling experiment on the three-atom
chain (its coupling structure matches the reference model, but the
Rydberg interaction is not an exact embedding of the mixed-sign Ising
model); measured configurations are evaluated on the original energy
landscape and the raw counts stay in the ledger.

## Reproduce

```powershell
python -m braket_experiments.run_experiments local
python -m braket_experiments.run_experiments submit --device sv1 --shots 4000
python -m braket_experiments.run_experiments collect --max-wait 600
python -m braket_experiments.run_experiments report
```

## Evidence boundaries

Established: circuit parity with the exact statevector solver (tests);
managed-simulator results matching local simulation to shot noise;
task ARNs and dimensions for every collected run; the OpenQASM 2 export
fix including linear terms.

Not established: any QPU result (pending agreement acceptance); hardware
noise comparisons; energy or runtime advantage on real quantum hardware;
advantage of QAOA over the repository's classical solvers at these sizes.
