#!/usr/bin/env bash
# =============================================================================
# Serve a model with vLLM (single node) and run the agentic MATH-500 eval
# (scripts/eval_math500_agentic.py) against it. Used for BOTH the baseline (base
# model) and the RL checkpoint, with identical settings -> the accuracy delta is
# the held-out benchmark result.
#
#   EVAL_MODEL_PATH=/Volumes/.../models/Qwen3.5-35B-A3B  bash scripts/serve_and_eval.sh
# =============================================================================
set -xeuo pipefail

command -v vllm >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"
export OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS=0
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${EVAL_MODEL_PATH:-/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B}"
TP="${EVAL_TP:-8}"
PORT="${EVAL_PORT:-8000}"
SERVED="${EVAL_MODEL:-eval}"
SERVE_LEN="${EVAL_SERVE_LEN:-8192}"      # prompt + multi-turn response budget
GPU_UTIL="${EVAL_GPU_UTIL:-0.85}"
HEALTH_TIMEOUT="${EVAL_HEALTH_TIMEOUT:-1800}"
LOCAL_CACHE="${EVAL_LOCAL_CACHE:-/local_disk0/eval_model}"
STAGE="${EVAL_STAGE:-1}"                  # bulk-copy UC->NVMe first (FUSE random-read is slow)

# --- optional NVMe pre-stage (UC FUSE mmap/random-read is slow; bulk cp is fast) --
SERVE_PATH="${MODEL_PATH}"
if [ "${STAGE}" = "1" ] && [ -d "${MODEL_PATH}" ]; then
  echo "[eval] staging ${MODEL_PATH} -> ${LOCAL_CACHE} (bulk copy)"
  mkdir -p "${LOCAL_CACHE}"
  # parallel copy of the shard files; -n so a partial re-run doesn't refetch.
  ls -1 "${MODEL_PATH}" | xargs -P 8 -I{} cp -rn "${MODEL_PATH}/{}" "${LOCAL_CACHE}/" || true
  SERVE_PATH="${LOCAL_CACHE}"
fi
# tokenizer for the eval client reads the (small) original path — same files.
export MODEL_PATH

echo "[eval] starting vLLM serve: ${SERVE_PATH} (TP=${TP}, len=${SERVE_LEN}) on :${PORT}"
vllm serve "${SERVE_PATH}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${SERVE_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --disable-custom-all-reduce \
  --trust-remote-code \
  ${EVAL_SERVE_EXTRA_ARGS:-} &
SERVE_PID=$!
cleanup() { echo "[eval] stopping vLLM (pid ${SERVE_PID})"; kill "${SERVE_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# --- wait for /health --------------------------------------------------------
echo "[eval] waiting for /health (<=${HEALTH_TIMEOUT}s) ..."
ok=0
for _ in $(seq 1 "${HEALTH_TIMEOUT}"); do
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "[eval] FATAL: vLLM server exited before becoming healthy" >&2
    wait "${SERVE_PID}" || true
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done
[ "${ok}" = "1" ] || { echo "[eval] FATAL: /health not ready within ${HEALTH_TIMEOUT}s" >&2; exit 1; }
echo "[eval] server healthy; running eval"

export EVAL_BASE_URL="http://127.0.0.1:${PORT}/v1"
export EVAL_MODEL="${SERVED}"
# EVAL_SCRIPT is the use case's eval.py as an ABSOLUTE path (the job sets it, e.g.
# ${CODE_SOURCE_PATH}/usecases/agentic-search/eval.py). Put its directory on PYTHONPATH
# so the eval can `import reward` / `import tool` -- the SAME modules training uses.
EVAL_SCRIPT="${EVAL_SCRIPT:?set EVAL_SCRIPT to the use case eval.py (absolute path)}"
export PYTHONPATH="$(dirname "${EVAL_SCRIPT}")${PYTHONPATH:+:${PYTHONPATH}}"
set +e
echo "[eval] running ${EVAL_SCRIPT}"
python3 "${EVAL_SCRIPT}"
RC=$?
set -e
echo "[eval] eval finished rc=${RC}"
exit "${RC}"
