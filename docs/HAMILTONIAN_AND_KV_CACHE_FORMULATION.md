# Constraint-Coupled Energy (CCE) QUBO Hamiltonian, APR & KV-Cache Paging

**CCE-QOS: Compiler Optimization & Combinatorial NPU Scheduling Series**  
*Independent Research Technical Report | Compiler Architecture & Quantum Optimization*

---

## 1. Mathematical Problem Formulation

Modern Neural Processing Units (NPUs) and multi-core spatial accelerators feature multi-bank on-chip SRAM hierarchies. Unoptimized compiler task graph scheduling induces concurrent bank contentions, cache thrashing, and uncoordinated DRAM spills.

### 1.1 Objective Formulation
We define the Constraint-Coupled Energy (CCE) cost over a Directed Acyclic Graph (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

$$\mathcal{J}(x) = \sum_{i \in \mathcal{V}} E_{\text{compute}}(i) + \sum_{(u, v) \in \mathcal{E}} E_{\text{comm}}(u, v) \cdot \mathbb{I}(\text{Bank}(u) \ne \text{Bank}(v)) + \sum_{t=1}^T \sum_{b=1}^B E_{\text{leak}}(b) \cdot \mathbb{I}(\text{Bank } b \text{ active})$$

### 1.2 Binary Decision Variables
To map the problem to an unconstrained binary optimization landscape, we define binary indicators:

$$x_{i, t, b} \in \{0, 1\} \iff \text{Task } i \text{ is scheduled at time slot } t \text{ using SRAM bank } b$$

The complete decision vector $x \in \{0, 1\}^N$ has dimension $N = |\mathcal{V}| \times T \times B$.

---

## 2. QUBO Hamiltonian Construction

The constrained scheduling problem is converted to an unconstrained Quadratic Unconstrained Binary Optimization (QUBO) problem:

$$\min_{x \in \{0, 1\}^N} H(x) = x^T Q x = H_{\text{cost}}(x) + \sum_{k=1}^K \lambda_k H_{\text{constraint}, k}(x)$$

### 2.1 One-Hot Task Scheduling Constraint
Every task must execute exactly once across all time slots and banks:

$$H_{\text{one-hot}} = \sum_{i \in \mathcal{V}} \left( \sum_{t=1}^T \sum_{b=1}^B x_{i, t, b} - 1 \right)^2$$

### 2.2 Precedence Dependency Constraint
For every edge $(u, v) \in \mathcal{E}$, task $u$ must complete before task $v$ begins ($t_v \ge t_u + d_u$):

$$H_{\text{precedence}} = \sum_{(u, v) \in \mathcal{E}} \sum_{t_u = 1}^T \sum_{t_v \le t_u + d_u - 1} \left( \sum_{b=1}^B x_{u, t_u, b} \right) \left( \sum_{b'=1}^B x_{v, t_v, b'} \right)$$

### 2.3 Bank Capacity & Contention Constraint
At any time slot $t$, total allocated SRAM buffer memory must not exceed bank capacity $C_{\text{bank}}$:

$$H_{\text{capacity}} = \sum_{t=1}^T \sum_{b=1}^B \max\left(0, \sum_{i \in \mathcal{V}_{\text{active}}(t)} M_i x_{i, t, b} - C_{\text{bank}}\right)^2$$

---

## 3. Adaptive Penalty Refinement (APR) Algorithm

Static penalty coefficients $\lambda_k$ fail across varying DAG topologies: under-penalization yields invalid schedules with dependency violations, while over-penalization distorts the energy landscape, trapping solvers in sub-optimal local minima.

### APR Feasibility Threshold (Scope Proposition)
> **Proposition.** Let the feasible set be nonempty, let violations be counted in whole units, and let $H_{\text{cost}}(x) \ge 0$ on the finite search space. If the inner solver returns an exact global minimizer of the penalized Hamiltonian and every multiplier exceeds the minimum cost of a feasible schedule, then the returned minimizer is feasible; the APR growth law therefore reaches a feasible solve in finitely many rounds under exact minimization.
>
> **Proof sketch.** A feasible schedule pays zero penalty, while any infeasible schedule pays at least one whole violation unit at its violating multiplier; once every multiplier exceeds the feasible minimum cost, no infeasible schedule can minimize the penalized Hamiltonian.
>
> **Scope.** The deployed pipeline uses approximate solvers (CP-SAT with time limits, simulated annealing, QAOA sampling), so the exact-minimizer assumption does not hold in general and APR is evaluated empirically (51.61% to 67.74% measured feasibility on the energy formulation). The earlier draft's "zero-violation convergence" theorem is withdrawn; the full statement and proof are in `docs/paper/RESEARCH_PAPER.md` section 2.2.

---

## 4. LLM KV-Cache Paging & Continuous Prefix Batching

In Large Language Model (LLM) serving, Key-Value (KV) attention caches grow dynamically ($O(L \cdot B \cdot d)$ per sequence). Static contiguous reservation can create internal fragmentation by reserving maximum sequence length for work that completes earlier. The repository simulator models this capacity-management tradeoff; it does not model host paging, DRAM faults, or a physical NPU page-table implementation.

We formulate block-level paged attention scheduling:
* Physical memory is partitioned into uniform blocks of size $B_{\text{tok}} = 16\text{ tokens}$.
* An abstract block map assigns logical KV tokens to an abstract finite block pool non-contiguously.
* Prompt prefix tokens (e.g., shared system instructions) are shared across concurrent requests via copy-on-write referencing.

On the repository's 50-request synthetic capacity trace (`implementations/v3_llm_kvcache_continuous_batching/llm_kvcache_paging_scheduler.py`, 1600-block pool, seed 7), static contiguous reservation completes all work at 17.4031 output tokens per simulated step with P95 end-to-end latency 402.85 steps. Exact-demand paged admission completes the same output-token count at 30.2720 output tokens per step with P95 latency 242.20 steps: 1.7395x modeled throughput and 39.878% lower P95 latency. These figures are deterministic simulation outputs, not vLLM, GPU, NPU, or hardware measurements (see `EVIDENCE.md`).
