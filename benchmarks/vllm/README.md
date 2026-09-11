# CCE-QOS and vLLM Benchmark Harness

This directory provides a reproducible **local** comparison protocol. It does
not claim to reproduce vLLM's SOSP 2023 A100 measurements and does not replace
vLLM's CUDA runtime. CCE-QOS contributes a bounded prefix-local admission
policy layered above the same stock vLLM server.

## What Is Compared

Every run uses the same model, GPU, server limits, trace, request cap, and
seed. The harness executes three configurations separately:

| Configuration | vLLM automatic prefix cache | Client dispatch |
|---|---:|---|
| `fifo_no_prefix_cache` | off | FIFO |
| `fifo_prefix_cache` | on | FIFO |
| `cce_prefix_bounded` | on | CCE-QOS prefix-local, bounded wait |

The second row isolates the stock vLLM cache feature. The third row measures
only the additional scheduling policy relative to the second row. Request
latency starts at the trace arrival timestamp, so any policy queue delay is
included. A request that reaches `max_reorder_wait_ms` is dispatched ahead of
prefix locality, preventing starvation.

## Environment

The verified local environment is WSL2/Linux with an RTX 4060 Laptop GPU
(8 GB), CUDA-visible compute capability 8.9, Python 3.12, and `vllm==0.29.0`.
Use a fresh environment because vLLM bundles a CUDA/PyTorch stack:

```bash
uv venv --python 3.12 --seed ~/.venvs/cce-qos-vllm
uv pip install --python ~/.venvs/cce-qos-vllm/bin/python -r requirements.txt --torch-backend=auto
```

The WSL2 image also needs the host compiler and Python headers used by
Triton and FlashInfer JIT compilation:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev
```

The pinned CUDA compiler packages match torch's CUDA 13.2 runtime. When that
toolkit is installed beside the Python environment, `run_local.py` exposes it
only to the vLLM child process and creates a private cache symlink for the
versioned `libcudart` ABI name required by FlashInfer's linker. The effective
child environment is recorded in every result JSON.

## Run

Generate the explicitly synthetic prefix-reuse workload. It is not ShareGPT,
Alpaca, or a paper-workload reproduction.

```bash
python make_prefix_trace.py \
  --output traces/prefix-reuse.jsonl \
  --requests 48 \
  --prefix-groups 4 \
  --arrival-interval-ms 10 \
  --max-tokens 48 \
  --seed 20260911
```

Run the full attribution set from the same vLLM environment:

```bash
python run_local.py \
  --trace traces/prefix-reuse.jsonl \
  --results-dir results/local-run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.72 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-inflight 16 \
   --max-reorder-wait-ms 100
```

Run at least three fresh-server trials and summarize only the matched,
error-free records:

```bash
python summarize.py \
  --trial results/full-trial-01 \
  --trial results/full-trial-02 \
  --trial results/full-trial-03 \
  --output results/full-summary.json
```

Each JSON result records the trace SHA-256, model, server command, child
environment, installed versions, GPU identity, dirty Git status, request-level
queue delay, TTFT, end-to-end latency, and completion-token throughput when the
server supplies OpenAI streaming usage. Server logs are kept beside results.
Generated traces and results are ignored by Git. Four requests with a distinct
warmup prompt run before timing by default; they are excluded from result
metrics.

vLLM 0.29.0 has a documented WSL2 Model Runner V2 UVA startup defect
([vLLM #54652](https://github.com/vllm-project/vllm/issues/54652)). On WSL2,
the harness sets `VLLM_USE_V2_MODEL_RUNNER=0` only for the vLLM child process,
selecting the upstream-described V1 fallback. The override appears in each
result JSON and must be retained when comparing runs.

## Interpretation Rules

- Compare `fifo_prefix_cache` with `fifo_no_prefix_cache` for vLLM automatic
  prefix-cache impact.
- Compare `cce_prefix_bounded` with `fifo_prefix_cache` for the CCE-QOS
  dispatcher impact.
- Do not compare local RTX 4060 values numerically with vLLM paper A100 values.
- Do not call a synthetic prefix-reuse trace a production or ShareGPT workload.
- A failed or incomplete trial is rejected by `summarize.py` and invalidates a performance claim.
