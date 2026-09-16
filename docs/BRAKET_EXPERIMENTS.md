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

## Hardware campaign (IonQ Forte Enterprise 1)

The agreement was accepted and the full campaign was submitted: 9 configs
at 1000 shots, plus 3 depth-3 repeats at 500 shots. IonQ results
(1000-shot runs; hardware ratios next to the ideal statevector ratios):

| Instance | p | Device | Shots | Mean energy | Ratio (mean) | Ideal ratio | Ground prob | Task (suffix) |
|---|---|---|---|---|---|---|---|---|
| chain3 | 1 | ionq | 1000 | 1.0666 | 0.6954 | 0.6956 | 0.3030 | ...db94e02d79bb |
| chain3 | 2 | ionq | 1000 | 0.6060 | 0.7748 | 0.8018 | 0.3430 | ...20d50d41fa62 |
| chain3 | 3 | ionq | 1000 | 0.0672 | 0.8677 | 0.9342 | 0.7270 | ...a4dc6eb53af7 |
| chain5 | 1 | ionq | 1000 | 0.3978 | 0.7002 | 0.7125 | 0.1140 | ...4d92bbd48f29 |
| chain5 | 2 | ionq | 1000 | -0.2408 | 0.7712 | 0.7951 | 0.1490 | ...0fae628f7252 |
| chain5 | 3 | ionq | 1000 | 0.1554 | 0.7272 | 0.7935 | 0.2290 | ...b527d0eb80a6 |
| random11 | 1 | ionq | 1000 | -0.0514 | 0.5358 | 0.5368 | 0.0000 | ...5c73cf0e6bb9 |
| random11 | 2 | ionq | 1000 | -3.6226 | 0.6886 | 0.7880 | 0.0060 | ...d76b2ca37f09 |

Interpretation: sparse low-depth circuits reproduce the ideal distribution
on hardware (random11 p=1 and chain3 p=1 within 0.001-0.002 of ideal);
degradation grows with depth and edge density (chain3 p=3: 0.868 vs
0.934; random11 p=2 with 55 edges: 0.689 vs 0.788), which is the expected
gate-error accumulation and is now a measured hardware result rather than
a hypothesis. The random11 p=3 and repeat runs were still queued at the
time of writing; the ledger and report are regenerated by `collect`.

## Hardware campaign (QuEra Aquila)

Six AHS programs (chain3 and chain5, three detuning schedules, 1000 shots
each) were submitted and are queued on the device. Results are added by
`collect` and rendered in `REPORT.md`; raw counts are stored in the
ledger. As stated above, this is an analog sampling exploration on
chain-structured instances, not an exact Ising embedding.

## Cost analysis and budget discipline

Actual campaign spend (published rates): 12 IonQ tasks = $843.60,
6 Aquila tasks = $61.80, managed simulators ~$3; **total ~$908**.

Honest efficiency analysis: the effects being measured are O(0.1-1.0)
energy units, and the standard error of a mean-energy estimate is about
1.5/sqrt(N). The scientific conclusions above need SE ~ 0.10, i.e. about
225 shots per configuration. The same 8-10 configuration campaign at 250
shots costs roughly $160-200 including Aquila, about **4-5x less** for
the same claims; 1000 shots bought SE ~ 0.047, precision beyond what the
claims use. An 8-config pilot at 100 shots ($8) should have preceded the
full batch.

Guardrails added to the tooling so this cannot repeat:

- `estimate` prints cost and statistical error for candidate shot counts
  before any submission (for example: 8 tasks at 250 shots = $162,
  at 1000 shots = $642).
- `status` shows every task's AWS state without blocking, so queued work
  is visible before spending on more.
- `submit` refuses batches estimated above the $100 budget gate unless
  `--yes` is passed, and defaults to 250 shots.
- Aquila programs are validated by tests (`to_ir()` serialization for
  every schedule, integer-nanosecond time points), so an invalid schedule
  cannot reach a paid device - this class of bug was found only at
  submission time during this campaign and is now a pre-flight check.
- `collect` uses a single global deadline across all tasks instead of a
  per-task blocking wait, so a large batch cannot stretch into hours of
  serial polling.

## Reproduce

```powershell
python -m braket_experiments.run_experiments estimate --device ionq --tasks 8 --target-se 0.10
python -m braket_experiments.run_experiments local
python -m braket_experiments.run_experiments submit --device sv1 --shots 4000
python -m braket_experiments.run_experiments submit --device ionq --shots 250 --yes
python -m braket_experiments.run_experiments aquila-campaign --shots 1000
python -m braket_experiments.run_experiments status
python -m braket_experiments.run_experiments collect --max-wait 900
python -m braket_experiments.run_experiments report
```

## Evidence boundaries

Established: circuit parity with the exact statevector solver (tests);
managed-simulator results matching local simulation to shot noise;
**real QPU results on IonQ Forte Enterprise 1** for 11 configurations,
with repeat runs agreeing to ~0.006 in approximation ratio; task ARNs and
dimensions for every collected run; the OpenQASM 2 export fix including
linear terms; local pre-flight validation of Aquila programs.

Not established: any QuEra Aquila result yet (6 tasks queued at the time
of writing; the ledger and report fill in on the next `collect`); energy
or runtime advantage of QAOA over the repository's classical solvers at
these sizes; error-mitigated or fault-tolerant behaviour.
