#!/usr/bin/env bash
# =============================================================================
# Serve ONE model with vLLM (single node) and run a use case's eval.py against it.
# Used for BOTH the baseline (base model) and an RL checkpoint, with identical
# settings, so the difference between the two artifacts is the result.
#
#   EVAL_MODEL_PATH=<base model dir | <run>/global_step_N[/actor/model/huggingface]> \
#   EVAL_SCRIPT='${CODE_SOURCE_PATH}/usecases/<uc>/eval.py' \
#     bash engine/serve/serve_and_eval.sh
#
# Everything checkable without a GPU is checked BEFORE staging weights or starting
# vLLM: the eval script exists, and the model is complete -- for a training
# checkpoint, verl's completion manifest plus every indexed shard. The model's
# identity is written to EVAL_MODEL_IDENTITY_FILE for the eval artifact.
# =============================================================================
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # engine/serve
# shellcheck source=../lib/paths.sh
source "${HERE}/../lib/paths.sh"
VERIFY_CKPT="${HERE}/../lib/verify_checkpoint.py"

command -v vllm >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"
export OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS=0
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- preflight: fail in seconds, not after staging ~70 GB and starting vLLM ------
# EVAL_SCRIPT comes from env_variables, where ${CODE_SOURCE_PATH} is NOT expanded
# (engine/lib/paths.sh). Resolve it, then require the file.
EVAL_SCRIPT="$(resolve_code_path "${EVAL_SCRIPT:?set EVAL_SCRIPT to the use case eval.py}")"
if [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "[eval] FATAL: EVAL_SCRIPT does not exist: ${EVAL_SCRIPT}" >&2
  exit 2
fi
# An eval artifact is evidence: never replace an earlier one (EVAL_OVERWRITE=1 to force).
for _out in "${EVAL_OUT:-}" "${EVAL_TRACE_OUT:-}"; do
  if [ -n "${_out}" ] && [ -e "${_out}" ] && [ "${EVAL_OVERWRITE:-0}" != "1" ]; then
    echo "[eval] FATAL: ${_out} already exists -- give this eval its own EVAL_OUT /" \
         "EVAL_TRACE_OUT, or set EVAL_OVERWRITE=1 to replace it." >&2
    exit 2
  fi
done

# The model is REQUIRED -- a default would let an eval of "the checkpoint" quietly
# evaluate something else (a fresh run never produces the step a stale default names).
if [ -z "${EVAL_MODEL_PATH:-}" ]; then
  echo "[eval] FATAL: set EVAL_MODEL_PATH: the base model dir, or a checkpoint --" \
       "e.g. make search-eval CKPT=<run>/global_step_N" >&2
  if [ -n "${EVAL_CKPT_ROOT:-}" ] && [ -d "${EVAL_CKPT_ROOT}" ]; then
    echo "[eval] complete checkpoints under ${EVAL_CKPT_ROOT}:" >&2
    python3 "${VERIFY_CKPT}" --list-complete "${EVAL_CKPT_ROOT}" >&2 || echo "  (none)" >&2
  fi
  exit 2
fi
EVAL_MODEL_IDENTITY_FILE="${EVAL_MODEL_IDENTITY_FILE:-$(mktemp -t eval_model_identity.XXXXXX)}"
if ! MODEL_HF_DIR="$(python3 "${VERIFY_CKPT}" "${EVAL_MODEL_PATH}" --print-hf-dir \
                       --json-out "${EVAL_MODEL_IDENTITY_FILE}")"; then
  echo "[eval] FATAL: ${EVAL_MODEL_PATH} is not a complete, servable model (reason above)." >&2
  exit 2
fi
export EVAL_MODEL_IDENTITY_FILE
# The eval client loads its tokenizer from MODEL_PATH; it must be the model being served.
if [ -n "${MODEL_PATH:-}" ] && [ "$(realpath -m "${MODEL_PATH}")" != "$(realpath -m "${MODEL_HF_DIR}")" ]; then
  echo "[eval] FATAL: MODEL_PATH=${MODEL_PATH} is not the served model ${MODEL_HF_DIR}." \
       "Set only EVAL_MODEL_PATH." >&2
  exit 2
fi
MODEL_PATH="${MODEL_HF_DIR}"

TP="${EVAL_TP:-8}"
PORT="${EVAL_PORT:-8000}"
SERVED="${EVAL_MODEL:-eval}"
SERVE_LEN="${EVAL_SERVE_LEN:-8192}"      # prompt + multi-turn response budget
GPU_UTIL="${EVAL_GPU_UTIL:-0.85}"
HEALTH_TIMEOUT="${EVAL_HEALTH_TIMEOUT:-1800}"
LOCAL_CACHE="${EVAL_LOCAL_CACHE:-/local_disk0/eval_model}"
STAGE="${EVAL_STAGE:-1}"                  # bulk-copy UC->NVMe first (FUSE random-read is slow)

# --- optional NVMe pre-stage (UC FUSE mmap/random-read is slow; bulk cp is fast) --
# The local copy is keyed by the model's IDENTITY (verify_checkpoint.py), copied into
# a temp dir, verified against the source, and only then renamed into place -- so a
# reused /local_disk0 can never serve a previous model's weights under this model's
# name, and a partial copy is never mistaken for a finished one.
SERVE_PATH="${MODEL_HF_DIR}"
if [ "${STAGE}" = "1" ]; then
  IDENT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["identity"])' "${EVAL_MODEL_IDENTITY_FILE}")"
  CACHE="${LOCAL_CACHE%/}/${IDENT}"
  if [ -f "${CACHE}/.voa_complete" ] && python3 "${VERIFY_CKPT}" "${CACHE}" --matches "${EVAL_MODEL_IDENTITY_FILE}"; then
    echo "[eval] reusing the verified local copy ${CACHE}"
  else
    echo "[eval] staging ${MODEL_HF_DIR} -> ${CACHE} (bulk copy)"
    rm -rf "${CACHE}" "${CACHE}.partial"
    mkdir -p "${CACHE}.partial"
    # parallel copy of the shard files; any failed cp fails the job (no `|| true`)
    find "${MODEL_HF_DIR}" -mindepth 1 -maxdepth 1 -not -name '.*' -print0 \
      | xargs -0 -P 8 -I{} cp -r {} "${CACHE}.partial/"
    python3 "${VERIFY_CKPT}" "${CACHE}.partial" --matches "${EVAL_MODEL_IDENTITY_FILE}"
    touch "${CACHE}.partial/.voa_complete"
    mv "${CACHE}.partial" "${CACHE}"
  fi
  SERVE_PATH="${CACHE}"
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
# EVAL_SCRIPT was resolved + checked in the preflight. Put its directory on PYTHONPATH
# so the eval can `import reward` / `import tool` -- the same modules training uses.
PYTHONPATH="$(dirname "${EVAL_SCRIPT}")${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH
set +e
echo "[eval] running ${EVAL_SCRIPT}"
python3 "${EVAL_SCRIPT}"
RC=$?
set -e
echo "[eval] eval finished rc=${RC}"
exit "${RC}"
