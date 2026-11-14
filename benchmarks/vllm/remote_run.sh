#!/usr/bin/env bash
# CCE-QOS measured server-class GPU paired comparison.
# Runs on a fresh Deep Learning AMI GPU instance. Installs vLLM, builds the
# Poisson interleaved trace, calibrates the constrained KV cache block budget,
# then runs 3 seeds x 2 arms (fifo_prefix_cache, cce_prefix_bounded) on
# identical traces. Writes results and uploads to S3.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SEEDS="${SEEDS:-42 123 2026}"
REQUESTS="${REQUESTS:-300}"
PREFIX_GROUPS="${PREFIX_GROUPS:-8}"
MEAN_IA_MS="${MEAN_IA_MS:-50}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_REORDER_WAIT_MS="${MAX_REORDER_WAIT_MS:-500}"
MAX_INFLIGHT="${MAX_INFLIGHT:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
# Constrained KV cache block budget (16-token blocks). Must be >= the minimum
# to serve one max_model_len request (about 66 blocks at 1024 tokens for
# Qwen2.5-1.5B), and small enough to hold only a few of the 8 prefix groups.
NUM_BLOCKS="${NUM_BLOCKS:-80}"
S3_BUCKET="${S3_BUCKET:-s3://esthien-atlas-eda-969739653654/engineering/cce-qos}"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
WORK="/opt/cce-qos-bench"
RESULTS="$WORK/results"
TRACE_DIR="$WORK/traces"

echo "=== CCE-QOS measured run $RUN_ID ==="
echo "model=$MODEL seeds=[$SEEDS] requests=$REQUESTS groups=$PREFIX_GROUPS mean_ia_ms=$MEAN_IA_MS"

# --- Python environment -------------------------------------------------
sudo mkdir -p "$WORK"
sudo chown "$USER":"$USER" "$WORK"
cd "$WORK"

if [ ! -d "$WORK/.venv" ]; then
  python3 -m venv "$WORK/.venv"
fi
# shellcheck disable=SC1091
source "$WORK/.venv/bin/activate"
pip install --upgrade pip >/dev/null
echo "installing vllm (this resolves a compatible torch wheel)..."
pip install "vllm==0.29.0" httpx >/dev/null
python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__); print('cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0))"

mkdir -p "$RESULTS" "$TRACE_DIR"

# --- Trace ---------------------------------------------------------------
python make_poisson_trace.py \
  --output "$TRACE_DIR/poisson-interleaved.jsonl" \
  --requests "$REQUESTS" \
  --prefix-groups "$PREFIX_GROUPS" \
  --mean-inter-arrival-ms "$MEAN_IA_MS" \
  --max-tokens "$MAX_TOKENS" \
  --seed 20261114 | tee "$RESULTS/trace_build.log"

# --- Constrained KV cache -------------------------------------------------
# NUM_BLOCKS is fixed above (default 80). It holds only a few of the 8 prefix
# groups, keeping the prefix cache smaller than the working set.
echo "using NUM_BLOCKS=$NUM_BLOCKS" | tee -a "$RESULTS/run.log"

# --- Arms ----------------------------------------------------------------
run_arm () {
  local seed="$1"; shift
  echo "--- seed $seed ---"
  for cfg in fifo_prefix_cache cce_prefix_bounded; do
    local rdir="$RESULTS/seed-${seed}/${cfg}"
    mkdir -p "$rdir"
    echo "running $cfg (seed $seed)"
    python run_local.py \
      --trace "$TRACE_DIR/poisson-interleaved.jsonl" \
      --results-dir "$rdir" \
      --model "$MODEL" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-inflight "$MAX_INFLIGHT" \
      --max-reorder-wait-ms "$MAX_REORDER_WAIT_MS" \
      --num-blocks "$NUM_BLOCKS" \
      --seed "$seed" \
      --warmup-requests 4 \
      --only "$cfg" 2>&1 | tee "$rdir/driver.log"
  done
}

for s in $SEEDS; do
  run_arm "$s"
done

# --- Package + upload ----------------------------------------------------
ENV_JSON="$RESULTS/environment.json"
{
  echo "{"
  echo "  \"run_id\": \"$RUN_ID\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"seeds\": [$(echo $SEEDS | sed 's/ /, /g')],"
  echo "  \"requests\": $REQUESTS,"
  echo "  \"prefix_groups\": $PREFIX_GROUPS,"
  echo "  \"mean_inter_arrival_ms\": $MEAN_IA_MS,"
  echo "  \"num_blocks\": $NUM_BLOCKS,"
  echo "  \"max_reorder_wait_ms\": $MAX_REORDER_WAIT_MS,"
  echo "  \"gpu_memory_utilization\": $GPU_MEM_UTIL,"
  echo "  \"max_num_seqs\": $MAX_NUM_SEQS,"
  echo "  \"max_inflight\": $MAX_INFLIGHT,"
  echo "  \"instance_info\": \"$(ec2metadata --instance-type 2>/dev/null || echo unknown)\","
  echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo unknown)\""
  echo "}"
} > "$ENV_JSON"

aws s3 cp "$RESULTS" "$S3_BUCKET/measured-$RUN_ID/" --recursive || echo "S3 upload failed (non-fatal)"
echo "=== run $RUN_ID complete ==="
echo "RESULTS_DIR=$RESULTS"
