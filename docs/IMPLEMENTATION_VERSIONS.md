# Component Architecture Guide

**CCE-QOS: Compiler Optimization, QUBO Energy Scheduling & KV-Cache Paging**

The repository ships three standalone component implementations under `implementations/`. The directory names on disk carry version-style prefixes for historical import-compatibility reasons; in all prose and documentation each component is referred to by its descriptive name below.

---

## 1. Architectural Comparison

| Feature / Metric | Exact CP-SAT Scheduler | QUBO-APR Engine | KV-Cache Paging Scheduler |
| :--- | :--- | :--- | :--- |
| **Directory** | [`implementations/v1_exact_cpsat_solver/`](../implementations/v1_exact_cpsat_solver/) | [`implementations/v2_cce_qubo_apr_engine/`](../implementations/v2_cce_qubo_apr_engine/) | [`implementations/v3_llm_kvcache_continuous_batching/`](../implementations/v3_llm_kvcache_continuous_batching/) |
| **Target Engine** | Google OR-Tools CP-SAT | Classical local search / quantum annealers | Deterministic KV-cache capacity simulation |
| **Implementation Language** | Python / C++ (OR-Tools) | Python / Vectorized NumPy | Python / Paged Cache Manager |
| **Optimality Guarantee** | **Exact Global Optimal (MIP)** | Approximate (local search) | Dynamic Heuristic |
| **Runtime Scaling** | Exponential in horizon | Polynomial ($O(N^2)$ per evaluation) | Token-stepped simulation, not serving latency |
| **Measured Result (this repo)** | Optimal makespan 10 cycles on the 6-node demo DAG, ~0.03 s solve | Zero violations from iteration 2; 8.79% objective-energy cut vs greedy on the 32-var demo | 1.7395x output tokens per simulated step and 39.878% lower P95 latency versus static reservation on a 50-request trace |
| **LLM Inference Speedup** | N/A (Static DAG) | N/A (Static DAG) | Capacity-model output only; see local vLLM attribution separately |
| **DRAM Page Faults** | N/A | N/A | Not modeled |

---

## 2. Directory Structure & File Map

```text
CCE-QOS/
├── implementations/
│   ├── v1_exact_cpsat_solver/               # Exact CP-SAT Scheduler
│   │   ├── ortools_cpsat_engine.py          # Google OR-Tools CP-SAT exact integer scheduler
│   │   ├── cce_dag_parser.py                # DAG JSON workload parser & memory model
│   │   └── main_cpsat_runner.py             # Exact optimal solver benchmark runner
│   ├── v2_cce_qubo_apr_engine/              # QUBO-APR Engine
│   │   ├── qubo_hamiltonian_generator.py    # QUBO matrix formulation H = x^T Q x
│   │   ├── apr_penalty_refinement.py        # Adaptive Penalty Refinement (APR) update rule
│   │   └── cce_qubo_benchmark.py            # Live APR convergence & energy benchmark
│   └── v3_llm_kvcache_continuous_batching/  # KV-Cache Paging Scheduler
│       ├── llm_kvcache_paging_scheduler.py  # Block-level KV cache paging simulation
│       └── qaoa_variational_circuit.py      # Mean-field variational QAOA energy model
```

---

## 3. Execution Instructions

### 3.1 Run the Exact CP-SAT Scheduler Benchmark
```bash
python -m implementations.v1_exact_cpsat_solver.main_cpsat_runner
```
Verified output: solver status OPTIMAL, makespan 10 cycles on the 6-node synthetic NPU DAG (OR-Tools 9.15).

### 3.2 Run the QUBO-APR Engine Convergence Benchmark
```bash
python -m implementations.v2_cce_qubo_apr_engine.cce_qubo_benchmark
```
Measured on the 32-variable demo workload (4 tasks x 4 slots x 2 banks, DAG edges 0→2, 1→2, 2→3): greedy topological baseline objective 11.95 a.u. with 0 violations; APR track reaches zero total violations from iteration 2 and a best feasible objective of 10.90 a.u. (8.79% reduction). The static-penalty control track (frozen low penalties) ends at objective 1.00 a.u. with 3 violations - illustrating that adaptive refinement, not a fixed penalty, drives feasibility.

### 3.3 Run the KV-Cache Paging Scheduler Simulation
```bash
python -m implementations.v3_llm_kvcache_continuous_batching.llm_kvcache_paging_scheduler
```
Measured on the 50-request synthetic capacity trace (seed 7, 1600-block pool of 16-token pages, 1024-token maximum model length): static contiguous reservation completes all requests at 17.4031 output tokens per simulated step with P95 end-to-end latency 402.85 steps and mean queue delay 69.84 steps. Paged exact-demand admission completes the same work at 30.2720 output tokens per step with P95 latency 242.20 and zero mean queue delay: 1.7395x modeled throughput and 39.878% lower P95 latency. The simulator queues active work safely and does not model vLLM, hardware, host paging, or DRAM faults.
