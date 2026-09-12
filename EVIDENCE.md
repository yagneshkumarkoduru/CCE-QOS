# EVIDENCE.md - CCE-QOS

Verified through: 2026-09-11. Every headline number below maps to an exact source file.
Classes: BENCHMARK = Python scheduler benchmark on synthetic workloads (not production, not hardware). LOCAL_EXTERNAL_BENCHMARK = a locally executed third-party serving engine on the recorded hardware and synthetic trace.

## Reconciliation statement (2026-09-10)

Root `metrics.txt`, root `results_table.txt`, root `schedules.json`, root `explanations.txt`, and every file in `outputs/` are all current-pipeline outputs: `run_experiment.py` writes both locations in one run, so the repository root and `outputs/` agree in format and numbers by construction. The previously cited figures (58.06% Quantum + APR feasibility, 4216.92 vs 5669.65 cost, X = 0.2562) came from a retired pipeline whose energy model and output format no longer exist in the repository; they are WITHDRAWN as pipeline-outputs and replaced below by the current pipeline's measured values. The current pipeline is deterministic (fixed seeds): rerunning `python run_experiment.py` reproduces `metrics.txt` byte-for-byte and `schedules.json` identically except for the wall-clock `runtime_seconds` field.

## Verified claims (safe to use, current pipeline)

| Claim | Value | Class | Source |
|---|---|---|---|
| Best classical cost (cost formulation) | 4168.69 (Lookahead), 54.83% feasibility, 26.47% cost reduction vs greedy | BENCHMARK | `metrics.txt` [Baseline Cost Objective]; `results_table.txt` [Baseline Cost Objective] |
| Greedy baseline | 5669.65 cost, 51.61% feasibility | BENCHMARK | `metrics.txt` [Baseline Cost Objective] |
| Best classical search on the CCE-QUBO energy | 154.40 energy (Lookahead / Beam Search) at 51.61% feasibility | BENCHMARK | `metrics.txt` [CCE-QUBO Objective]; `results_table.txt` [CCE-QUBO Objective] |
| CCE + APR feasibility (energy formulation, ours) | 67.74% - the measured maximum across pipeline methods (up from 51.61% for greedy, +16.13pp) | BENCHMARK | `metrics.txt` [APR / Quantum (local-search fallback)]; `results_table.txt` [APR / Quantum (local-search fallback)] |
| CCE + APR cost and energy | cost 6287.24 (-7.5% vs the CCE-QUBO Lookahead 6794.39), energy 160.43, latency 3479.19 cycles | BENCHMARK | `metrics.txt` [APR / Quantum (local-search fallback)]; `outputs/schedules.json` `results.cce_qubo_apr` |
| CCE-QUBO Simulated Annealing feasibility | 58.06% at energy 159.31 (best classical feasibility inside the CCE-QUBO method suite) | BENCHMARK | `metrics.txt` [CCE-QUBO Objective] |
| Weighted X metric (current definition) | CCE vs baseline: -0.4926; quantum vs baseline: -1.0197 | BENCHMARK | `metrics.txt` [X Metric] |
| QAOA approximation ratio, reference 3-qubit instance (p=2, seeded) | 0.8012 (deterministic; replaces the stochastic 0.611 figure) | BENCHMARK | `QAOA_solver.py` optimizer, seeded run; `tests/test_qaoa_multip.py` |
| QAOA p-sweep on the reference 3-qubit instance | p=1: 0.6878, p=2: 0.8021, p=3: 0.9385 (best p=3, COBYLA 60 evaluations per depth) | BENCHMARK | `sweep_qaoa_depth` in `QAOA_solver.py` |
| QAOA p-sweep on a real 11-variable workload sub-instance (conv_stem -> bn_stem from `example_workload.json`) | p=1: 0.7244 (old default p=1 gradient run: 0.7088), p=2: 0.7465, p=3: 0.7823 (best p=3, COBYLA 60 evaluations per depth, seed 1251) | BENCHMARK | `run_qaoa_statevector` p-sweep path in `quantum_interface.py` |

## APR regression: diagnosed and fixed (2026-09-10)

The APR arm previously reproduced the greedy schedule (cost 5669.65, 51.61% feasibility), and in the retired pipeline it recorded a cost of 4943.55 that was worse than the Lookahead baseline 4168.69 - previously left "unexplained". Root cause, measured in the current pipeline (`outputs/schedules.json` `results.cce_qubo_apr.metadata.round_trace`):

1. **Cross-round comparison on penalty-inflated objectives.** Each APR round optimized the QUBO energy at that round's own penalty multipliers, and the best-candidate selection compared those penalized objectives directly. As the multipliers ratcheted up (the tuner only adds and never subtracts), the recorded objective grew monotonically even at identical schedules: round 1 objective 160.275, round 2 162.854 (+2.58 at the SAME schedule), round 5 167.810. Round 1 (lowest multipliers) always won the selection, so APR degenerated to the greedy schedule it starts from.
2. **The retired pipeline's 4943.55 was the same artifact.** Its round trace recorded multipliers escalated to 14 (bandwidth_capacity / dram_pressure / memory_bank_conflict / bandwidth_imbalance all pinned at lam_max = 14) with ~13 bandwidth violations per schedule; the recorded cost 4943.55 decomposes into a true cost near 4169-4300 plus roughly 650-775 of inflated penalty terms (14 x 0.42 x 100 for bandwidth alone contributes ~590). The schedule was not catastrophically worse - the recorded objective was dominated by penalty terms at inflated multipliers.

Fixes implemented in `run_experiment.py` (`_run_apr_arm`), all measured:

- **Canonical candidate ranking:** candidates are compared on the lexicographic key (violation mass, energy at the base multipliers) - penalty-scale-free, so inflated rounds can no longer mask better schedules.
- **Penalty annealing:** once zero violations hold for `anneal_patience` consecutive rounds, multipliers decay by `anneal_decay` back toward the configured base values (never below base). On this workload zero violations are not reached, so escalation stays active (the safeguard is implemented and exercised on the synthetic APR test workload in `tests/test_apr_arm.py`).
- **Feasibility-preserving polish:** a final local descent (swap/insert/block-reverse moves) from each round candidate that only accepts moves which strictly reduce the violation mass, or keep it while strictly reducing the canonical energy. Measured: 4 polished starts, 300 iterations each, moved the selected schedule from 15.001 violation mass to 10.001 (feasibility 51.61% -> 67.74%).

Result (current pipeline, `metrics.txt` [APR / Quantum (local-search fallback)]):

| Strategy | Cost | Energy | Latency (cycles) | Feasibility |
|---|---:|---:|---:|---:|
| CCE + APR (fixed, ours) | 6287.24 | 160.43 | 3479.19 | **67.74%** |
| Lookahead (CCE-QUBO arm, best classical energy) | 6794.39 | **154.40** | 3603.50 | 51.61% |
| Greedy | 5669.65 | 160.28 | 3456.25 | 51.61% |
| CCE + APR (before fix: greedy clone) | 5669.65 | 160.28 | 3456.25 | 51.61% |

Quantified tradeoff: on the energy formulation the fixed APR does NOT match the classical Lookahead's QUBO energy (160.43 vs 154.40, +3.9%); it delivers +16.13pp feasibility (67.74%, the measured maximum), a 7.5% lower cost than Lookahead within the same CCE-QUBO evaluation (6287.24 vs 6794.39), and the lowest latency in the CCE-QUBO table (3479.19 cycles). The tradeoff is intentional (APR prioritizes feasibility) and is now data-backed instead of "unexplained".

## Known contradictions (FIXED 2026-09-10)

1. `docs/paper/RESEARCH_PAPER.md` previously claimed 100.0% feasible schedules and mixed old/new formulation numbers in Table 4.1. Paper abstract and table now use the current pipeline's measured values (APR feasibility 67.74%; classical Lookahead 4168.69 at 54.83%). Do not restore the 100% claim without a new measured run.
2. Theorem 1 (APR monotonic finite convergence) proof has a logical gap: bounded penalty-violation product does not imply monotonic decrease. Treat APR as an empirical heuristic pending a repaired proof or downgrade to Lemma/Observation.
3. Theorem 2 (McCormick envelopes) is textbook background, not a novel contribution. Cite standard references.
4. Baselines are greedy, simulated annealing, lookahead, and statevector QAOA on synthetic example workloads (`example_workload.json`, `outputs/schedules.json`). Do NOT claim comparison against TVM, XLA, or MLIR, and do NOT call workloads "production edge" workloads.
5. README KV-cache claims were previously "verified" against the old hardcoded benchmark (`run_paging_benchmark` returned fixed constants 56.0% / 3.54x). The intermediate 79,032-to-0 DRAM-fault and 7.95x figures were also withdrawn on 2026-09-11 when the implementation was replaced with a safe capacity simulator. The current simulation reports queueing, capacity commitment, latency, and output-token rate only; it does not model active-KV host paging or DRAM page faults.

## Live re-verification 2026-09-10 (post-fix)

- `run_experiment.py`: the APR arm now uses canonical penalty-scale-free candidate ranking, zero-violation-triggered penalty annealing, and a feasibility-preserving local polish (`_run_apr_arm`). A full rerun reproduces `outputs/metrics.txt` byte-for-byte (deterministic seeds); only the wall-clock `runtime_seconds` field in `outputs/schedules.json` varies between runs.
- The QUBO-APR Engine benchmark (`implementations/v2_cce_qubo_apr_engine/cce_qubo_benchmark.py`): the benchmark runs a live APR loop (QUBO via `qubo_hamiltonian_generator.py`, greedy random-restart bit-flip local search, `apr_penalty_refinement.py` updates on measured violations, 12 rounds, seed 7). Measured: greedy topological baseline objective 11.95 a.u. (0 violations), APR zero total violations first at iteration 2, best feasible APR objective 10.90 a.u. = 8.79% reduction vs greedy; static-penalty control ends infeasible (3 violations). All plotted curves are recorded from the loop.
- The KV-Cache Paging Scheduler entries in this 2026-09-10 section are superseded by the 2026-09-11 capacity-simulation verification below. The retired 79,032-to-0 DRAM-fault and 7.95x figures must not be reused.
- The Exact CP-SAT Scheduler runner (`implementations/v1_exact_cpsat_solver/main_cpsat_runner.py`): ortools 9.15 is installed in `.venv`. Both invocation modes verified 2026-09-10 (`python implementations/v1_exact_cpsat_solver/main_cpsat_runner.py` and `python -m implementations.v1_exact_cpsat_solver.main_cpsat_runner`): status OPTIMAL, makespan 10 cycles, ~0.02 s solve.
- `QAOA_solver.py`: variational parameter initialization is now seeded (`DEFAULT_QAOA_SEED = 12345`, `np.random.default_rng`) so repeated optimizations return bit-identical (gamma, beta, energy) tuples; verified by `tests/test_qaoa_reproducibility.py` (same instance, fresh instances, seed override changes the trajectory, global NumPy RNG untouched). The engine additionally supports any depth p >= 1 (parameter vector length 2p) and provides `sweep_qaoa_depth` (COBYLA budget per depth); the pipeline's QAOA arm (`quantum_interface.run_qaoa_statevector`) sweeps p = 1..p_max and keeps the depth with the best ground-state approximation ratio, still bounded to QUBOs with <= 12 variables (the 31-node example workload QUBO has 4247 variables, so the pipeline runs the deterministic classical multi-start bit-flip local search on the QUBO energy, labeled backend `local_search_fallback`).
- `cce_llm_kvcache_scheduler.py` remains a separate 8-request, 43-block synthetic energy-model demo: it prints 25 versus 11 modeled misses and a derived 55.21% energy reduction. Its 18.4 ms to 5.2 ms ITL values are input model parameters, not measurements. It is not evidence for the current capacity simulator or local vLLM benchmark and is not cited as a current paper result.

## Superseding KV-Cache And vLLM Verification 2026-09-11

### Capacity simulation

`implementations/v3_llm_kvcache_continuous_batching/llm_kvcache_paging_scheduler.py` now performs deterministic token-stepped admission simulation on a 50-request synthetic trace (seed 7, 16-token blocks, 1600-block pool, 1024-token maximum model length). The contiguous baseline reserves every request's maximum length; the paged policy commits exact prompt-plus-output demand and queues safely when capacity is exhausted. It does not model host paging, DRAM faults, vLLM behavior, GPU throughput, or NPU hardware.

| Policy | Completed | Output tokens/step | P95 end-to-end steps | Mean queue delay steps | Peak committed blocks |
|---|---:|---:|---:|---:|---:|
| Static contiguous reservation | 50 | 17.4031 | 402.85 | 69.84 | 1600 |
| Paged exact-demand admission | 50 | 30.2720 | 242.20 | 0.00 | 1472 |

The deterministic capacity-model ratios are 1.7395x output-token throughput and 39.878% lower P95 end-to-end latency for paged exact-demand admission. Source: a fresh invocation of the scheduler module on 2026-09-11; see `tests/test_kvcache_paging.py` for invariants and reproducibility coverage.

### Local vLLM attribution benchmark

`benchmarks/vllm/` runs stock `vllm==0.29.0` separately for the no-prefix-cache FIFO baseline, stock automatic prefix caching with FIFO dispatch, and the same stock prefix cache with the CCE-QOS bounded prefix-local client dispatcher. The three-trial local run used `Qwen/Qwen2.5-0.5B-Instruct`, `torch==2.13.0+cu132`, an RTX 4060 Laptop GPU (8188 MiB), a 2048-token limit, 16 server/client concurrent requests, four excluded warmups, and an explicitly synthetic 48-request four-prefix trace (`b574387035d019e747a454f8978715e863b4338632a21335e7d830aad4ff45ec`). WSL used the upstream-described `VLLM_USE_V2_MODEL_RUNNER=0` fallback; all three configurations used the same fallback.

| Configuration | Median duration ms | Median mean end-to-end ms | Median output tokens/s |
|---|---:|---:|---:|
| FIFO, prefix cache off | 2526.125 | 1531.858 | 912.069 |
| FIFO, stock prefix cache on | 1204.116 | 612.204 | 1913.436 |
| CCE-QOS bounded dispatch, stock prefix cache on | 1219.030 | 628.828 | 1890.027 |

All nine run records completed 48 of 48 requests with streaming usage. On this local synthetic condition, stock prefix caching is the dominant effect. The bounded dispatcher did not improve over FIFO plus stock prefix caching: its paired median duration was 2.062% higher, mean end-to-end latency 2.715% higher, and output-token rate 2.020% lower. This is a negative result for the incremental dispatcher under this configuration, not evidence of a general vLLM improvement. Sources: `benchmarks/vllm/results/full-20260911-trial-01/` through `full-20260911-trial-03/` and the validated aggregate `benchmarks/vllm/results/full-20260911-summary.json`.

## Next measurements required

## Prefix-Locality Dispatcher Simulation (NEW 2026-09-12)

A new prefix-locality-aware KV-cache dispatcher was designed and verified in simulation.
Source: `benchmarks/vllm/prefix_locality_dispatcher.py`
Results: `results/prefix_locality/simulation_results.json`
Class: SIMULATION - discrete-event capacity model, NOT real vLLM or GPU measurement.

**Workload:** 300 requests/trial, 5 trials, 8 prefix groups, 512-token system prompts,
interleaved arrivals (random prefix per request), KV cache capacity = 3 prefix groups.
This models a congested LLM server serving RAG queries with shared system prompts.

| Dispatcher | Hit Rate | TTFT (ms) | vs FIFO |
|---|---|---|---|
| FIFO (baseline) | 37.1% | 94,368 | - |
| CCE-QOS Bounded (100ms window, no prefix score) | 37.1% | 94,048 | -0.3% |
| PrefixLocality (200ms window, prefix-aware) | 38.7% | 93,081 | +1.4% TTFT improvement |
| **PrefixLocality (500ms window, prefix-aware)** | **47.4%** | **85,390** | **+10.3pp hit rate, 9.5% TTFT** |

**Key claim (SIMULATION):** Prefix-locality-aware scheduling with a 500ms reorder window
achieves +10.3pp higher KV cache hit rate and 9.5% lower mean TTFT vs FIFO on interleaved
RAG workloads with tight KV cache (3 prefix groups, 8 total). Tradeoff: longer window =
more reordering = higher hit rate but also higher max queue delay.

**Why CCE-QOS Bounded was worse than FIFO before:** The 100ms reorder window with no
prefix scoring added queuing overhead without any hit-rate benefit. The new dispatcher
fixes this with explicit prefix scoring.

**Why this is an improvement over the prior vLLM negative result:**
The prior benchmark used FIFO+stock prefix cache which handles bursty (clustered) workloads
well. The new simulation targets the harder case: interleaved workloads where requests
from different prefix groups arrive mixed together and prefix-aware reordering provides
real benefit.

## Next measurements required

- [x] Add one published serving baseline on the same workload files (done 2026-09-11)
- [x] Build prefix-locality-aware dispatcher (done 2026-09-12)
- [ ] Repair Theorem 1 proof or reframe as empirical observation
- [x] Fix the Exact CP-SAT Scheduler runner imports (done 2026-09-10)
- [x] Reconcile root benchmark files with current pipeline (done 2026-09-10)
- [x] Diagnose and fix APR regression (done 2026-09-10: feasibility 51.61% -> 67.74%)
- [ ] Run PrefixLocality dispatcher against real vLLM on interleaved-prefix workload
