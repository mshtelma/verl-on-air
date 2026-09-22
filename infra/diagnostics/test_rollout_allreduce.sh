#!/usr/bin/env bash
# =============================================================================
# 1-NODE DE-RISK (8xH100): reproduce the air/53 training-rollout crash and test
# the two graphs-preserving fixes, BEFORE committing another 32-GPU run.
#
# air/53 died at vLLM CUDA-graph capture: `custom_all_reduce.cuh:455 'invalid
# argument'` (a known vLLM bug on H100 -- see memory vllm-custom-allreduce-h100
# + vllm#42609/#43923/#40812). This spins up JUST the Qwen3.5-35B-A3B rollout
# engine at the SAME shapes that crashed (GEN_TP=8, max_model_len=8192, graphs
# ON) under the SAME env the launcher sets, and runs three variants:
#
#   V1 baseline_repro   : expandable_segments:True + custom AR  -> EXPECT CRASH
#   V2 no_expandable_seg: (no expandable_segments) + custom AR   -> lever A: keeps
#                         the FAST custom kernel if the alloc-conf was the trigger
#   V3 disable_custom_ar: expandable_segments:True + --disable-custom-all-reduce
#                         -> lever B: NCCL fallback (what a patched vLLM auto-does)
#
# Faithful env (from run_grpo_fully_async.sh): VLLM_USE_V1=1,
# VLLM_ALLREDUCE_USE_SYMM_MEM=0, CUDA_DEVICE_MAX_CONNECTIONS=1, and
# PYTORCH_CUDA_ALLOC_CONF toggled PER VARIANT here. Graphs stay ON (no
# --enforce-eager) for every variant -- the whole point is to keep graphs.
#
#   air run --file air/54_test_rollout_allreduce.yaml -p df1 --watch
# =============================================================================
set -uo pipefail   # NOT -e: we WANT to survive a crashing variant and test the next

export OPENSSL_FORCE_FIPS_MODE=0 OPENSSL_FIPS=0
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Match the launcher's rollout env (everything EXCEPT PYTORCH_CUDA_ALLOC_CONF,
# which each variant sets/unsets itself).
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="${ROLLOUT_MODEL_PATH:-/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B}"
TP="${ROLLOUT_TP:-8}"
MAX_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
MEM="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
LOCAL_CACHE="${ROLLOUT_LOCAL_CACHE:-/local_disk0/rollout_cache}"
HEALTH_TIMEOUT="${ROLLOUT_HEALTH_TIMEOUT:-600}"
PORT="${ROLLOUT_PORT:-8100}"

command -v vllm >/dev/null 2>&1 || { echo "FATAL: vllm not on PATH." >&2; exit 127; }

# --- pre-stage weights to NVMe once (FUSE random-read is slow; bulk cp is fast) --
if [[ "${MODEL}" == /Volumes/* ]]; then
    dst="${LOCAL_CACHE%/}/$(basename "${MODEL}")"
    if [ -f "${dst}/.stage_complete" ]; then
        echo "[test] local cache already complete: ${dst}"
    else
        echo "[test] staging ${MODEL} -> ${dst} ..."
        mkdir -p "${dst}"; t0=$(date +%s)
        find "${MODEL}" -maxdepth 1 -mindepth 1 -type f -printf '%P\0' \
            | xargs -0 -P 8 -I {} cp -f "${MODEL}/{}" "${dst}/{}"
        find "${MODEL}" -maxdepth 1 -mindepth 1 -type d -printf '%P\0' \
            | xargs -0 -r -I {} cp -rf "${MODEL}/{}" "${dst}/{}"
        echo "[test] staged in $(( $(date +%s) - t0 ))s"
        touch "${dst}/.stage_complete"
    fi
    MODEL="${dst}"
fi

wait_gpu_free() {   # block until the busiest GPU is nearly empty (frees between variants)
    local used
    for _ in $(seq 1 30); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
        used="${used// /}"
        [ -n "${used}" ] && [ "${used}" -lt 3000 ] && return 0
        sleep 5
    done
    echo "[test] WARN: GPU mem not freed before next variant (busiest=${used:-?}MiB)"
}

kill_vllm() {   # SERVER_PID + any lingering vLLM worker/enginecore procs
    kill "$1" 2>/dev/null || true; sleep 3
    pkill -9 -f 'vllm serve'  2>/dev/null || true
    pkill -9 -f 'VllmWorker'  2>/dev/null || true
    pkill -9 -f 'EngineCore'  2>/dev/null || true
    sleep 3
}

declare -A RESULT
ORDER=()

try_variant() {   # try_variant <label> <alloc_conf ('' => unset)> <extra_flags>
    local label="$1" alloc="$2" flags="$3"
    local log="/tmp/rollout_${label}.log"
    ORDER+=("${label}")
    echo ""
    echo "======================================================================"
    echo "[test] VARIANT ${label}"
    echo "       PYTORCH_CUDA_ALLOC_CONF='${alloc:-<unset>}'  extra_flags='${flags:-<none>}'  graphs=ON"
    echo "======================================================================"
    if [ -n "${alloc}" ]; then export PYTORCH_CUDA_ALLOC_CONF="${alloc}"; else unset PYTORCH_CUDA_ALLOC_CONF; fi

    # graphs ON (NO --enforce-eager); core args mirror verl's rollout.
    # shellcheck disable=SC2086
    vllm serve "${MODEL}" \
        --served-model-name rollout_test \
        --tensor-parallel-size "${TP}" \
        --host 0.0.0.0 --port "${PORT}" \
        --gpu-memory-utilization "${MEM}" \
        --max-model-len "${MAX_LEN}" \
        --max-num-batched-tokens "${MAX_LEN}" \
        --dtype bfloat16 \
        --enable-chunked-prefill \
        --trust-remote-code \
        ${flags} > "${log}" 2>&1 &
    local pid=$! deadline=$(( $(date +%s) + HEALTH_TIMEOUT )) verdict=""
    while true; do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            verdict="PASS (graphs captured, /health OK)"; break
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then verdict="CRASH (engine exited)"; break; fi
        [ "$(date +%s)" -ge "${deadline}" ] && { verdict="TIMEOUT (${HEALTH_TIMEOUT}s)"; break; }
        sleep 5
    done
    # classify: was it the custom-all-reduce graph-capture bug, or something else?
    local sig
    sig="$(grep -aoE "custom_all_reduce.cuh:[0-9]+ '?(invalid argument|an illegal memory access[^']*)'?|Engine core initialization failed|died unexpectedly" "${log}" 2>/dev/null | head -1)"
    kill_vllm "${pid}"; wait_gpu_free
    RESULT["${label}"]="${verdict}${sig:+  <-  ${sig}}"
    echo "[test] ${label} RESULT: ${RESULT[${label}]}"
    if [[ "${verdict}" != PASS* ]]; then
        echo "----- ${label}: first error/traceback lines -----"
        grep -naiE 'custom_all_reduce|invalid argument|illegal memory|Traceback|Error|assert|died unexpectedly|Engine core' "${log}" 2>/dev/null | head -20
    fi
}

echo "############ ROLLOUT CUSTOM-ALL-REDUCE DE-RISK ############"
echo "model=${MODEL} TP=${TP} max_model_len=${MAX_LEN} mem=${MEM} (graphs ON, symm_mem OFF)"

try_variant "V1_baseline_repro"    "expandable_segments:True" ""
try_variant "V2_no_expandable_seg" ""                          ""
try_variant "V3_disable_custom_ar" "expandable_segments:True" "--disable-custom-all-reduce"

echo ""
echo "==================== ROLLOUT ALL-REDUCE TEST SUMMARY ===================="
for k in "${ORDER[@]}"; do printf '  %-22s %s\n' "${k}" "${RESULT[${k}]:-<not run>}"; done
echo "========================================================================"
echo "Interpretation:"
echo "  V1 CRASH + V2 PASS -> expandable_segments is the trigger; fix = alloc-conf override (keeps custom kernel = fastest)."
echo "  V1 CRASH + V3 PASS -> --disable-custom-all-reduce fixes it (NCCL fallback; rollout-scoped, no trainer impact)."
echo "  V1 PASS            -> standalone harness did NOT reproduce; the crash needs verl's async path (re-plan)."
