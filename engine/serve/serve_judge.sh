#!/usr/bin/env bash
# =============================================================================
# Serve the LLM-as-judge as an OpenAI-compatible endpoint (vLLM or SGLang).
#
# The judge is DECOUPLED from the verl training stack: it runs in its OWN image
# (a RECENT engine build, independent of the training image's pinned vLLM 0.24.0)
# and the training job reaches it over HTTP via JUDGE_BASE_URL
# (usecases/math/reward.py). So the judge model can be far newer than
# anything the training rollout could run.
#
# ENGINE: JUDGE_ENGINE=sglang (default) | vllm. Both expose an OpenAI /v1 API and
# a /health endpoint, so the reward client is engine-agnostic. For GLM-5.3-Flash
# the mature lane (2026-09) is the official SGLang image lmsysorg/sglang:glm-5.3-flash
# on x86/H100; the FP8 base checkpoint zai-org/GLM-5.3-Flash (~226 GB) fits one
# 8xH100 node comfortably. Engine-specific flags go through JUDGE_EXTRA_ARGS
# (e.g. --reasoning-parser glm45 for vLLM GLM, or SGLang's --reasoning-parser).
#
# Serve forever. If JUDGE_RENDEZVOUS is set, publish "http://<ip>:<port>/v1" there so
# training reward workers on other nodes can discover this endpoint. The training job's
# dispatcher (engine/train/dispatch_agentic.sh) runs this on the judge nodes.
#
#   MODEL: JUDGE_MODEL_PATH (a UC/local dir) OR JUDGE_MODEL_ID (an HF repo id).
# =============================================================================
set -xeuo pipefail

export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

JUDGE_ENGINE="${JUDGE_ENGINE:-sglang}"          # sglang | vllm
MODEL="${JUDGE_MODEL_PATH:-${JUDGE_MODEL_ID:?set JUDGE_MODEL_PATH (a dir) or JUDGE_MODEL_ID (an HF repo id)}}"
SERVED_NAME="${JUDGE_SERVED_NAME:-judge}"
PORT="${JUDGE_PORT:-8000}"
TP="${JUDGE_TP:-8}"
MEM_FRACTION="${JUDGE_GPU_MEM_UTIL:-0.90}"      # vLLM --gpu-memory-utilization / SGLang --mem-fraction-static
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"   # judge sees prompt + trajectory; 16k is ample for GSM8K
HEALTH_TIMEOUT="${JUDGE_HEALTH_TIMEOUT:-2400}"  # allow for a large first-load / HF pull (~226 GB fp8)
# Optional engine-specific passthrough, e.g. JUDGE_EXTRA_ARGS="--reasoning-parser glm45".
read -r -a EXTRA_ARGS <<< "${JUDGE_EXTRA_ARGS:-}"

# --- multi-node judge (tensor-parallel across >1 node) -----------------------
# A large FP8 model (e.g. GLM-5.3, 744B ~= 744 GB) does not fit one 8xH100 (640 GB)
# and df1 has no H200, so the judge serves at TP=8*NNODES across several nodes.
# AI Runtime runs this script once PER NODE; the caller maps the injected topology
# to these vars (JUDGE_NODE_RANK=POD_RANK, JUDGE_HEAD_ADDR=MASTER_ADDR, NNODES=...).
# Rank 0 = Ray head + vLLM API server (+ rendezvous publish); rank>0 = a Ray worker
# that joins the head's cluster and idles until the head's Ray port goes away.
# Default to the AI Runtime injected topology (identity map: every node is a judge
# node), so a single-role judge job needs no command wrapper. The dispatcher
# sets the JUDGE_* vars explicitly to override this when only a SUBSET of the job's
# nodes are judges (the rest run training).
JUDGE_NNODES="${JUDGE_NNODES:-${NUM_NODES:-1}}"
JUDGE_NODE_RANK="${JUDGE_NODE_RANK:-${POD_RANK:-${NODE_RANK:-0}}}"
JUDGE_HEAD_ADDR="${JUDGE_HEAD_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
JUDGE_RAY_PORT="${JUDGE_RAY_PORT:-6380}"        # distinct from verl training's 6379
if [ "${JUDGE_NNODES}" -gt 1 ]; then
    TP="${JUDGE_TP:-$(( 8 * JUDGE_NNODES ))}"   # fill every judge-node GPU by default
fi

# --- optional: pre-stage weights from slow UC FUSE onto fast local NVMe -------
# Serving a large checkpoint straight off a /Volumes UC FUSE mount is bottlenecked
# by FUSE read bandwidth. Measured on 8xH100 with GLM-5.3-Flash (306 GiB, 62 fp8
# shards): SGLang's multi-thread loader ran at ~450 MiB/s aggregate = ~92 s/shard
# = ~95 min for the full model, which blows any sane health deadline. Copying the
# checkpoint once to local NVMe hits the SAME ~450 MiB/s FUSE ceiling (~12 min),
# but then SGLang loads from NVMe at GB/s (a couple of minutes) and the endpoint is
# far more robust. Enable by pointing JUDGE_LOCAL_CACHE at an NVMe dir (/local_disk0).
# STAGE_ONLY=1 does just the copy (prints throughput) and exits — a cheap way to
# validate the copy path / measure FUSE read speed without launching the server.
STAGE_ONLY="${STAGE_ONLY:-0}"
if [ -n "${JUDGE_LOCAL_CACHE:-}" ] && [[ "${MODEL}" == /Volumes/* ]]; then
    dst="${JUDGE_LOCAL_CACHE%/}/$(basename "${MODEL}")"
    if [ -f "${dst}/.stage_complete" ]; then
        echo "[judge] local cache already complete: ${dst}"
    else
        echo "[judge] staging ${MODEL} -> ${dst} (fast local NVMe)..."
        mkdir -p "${dst}"
        src_bytes=$(du -sb "${MODEL}" | awk '{print $1}')
        echo "[judge] source size: $(( src_bytes / 1024 / 1024 )) MiB; target filesystem:"
        df -h "${JUDGE_LOCAL_CACHE}" || true
        t0=$(date +%s)
        # Parallel copy of the flat top-level files saturates the FUSE read ceiling;
        # GLM checkpoints are flat, but copy any subdirs too, just in case.
        find "${MODEL}" -maxdepth 1 -mindepth 1 -type f -printf '%P\0' \
            | xargs -0 -P "${JUDGE_STAGE_PARALLEL:-8}" -I {} cp -f "${MODEL}/{}" "${dst}/{}"
        find "${MODEL}" -maxdepth 1 -mindepth 1 -type d -printf '%P\0' \
            | xargs -0 -r -I {} cp -rf "${MODEL}/{}" "${dst}/{}"
        t1=$(date +%s)
        # Integrity: staged shard count must match the source.
        src_n=$(find "${MODEL}" -maxdepth 1 -name '*.safetensors' | wc -l)
        dst_n=$(find "${dst}"   -maxdepth 1 -name '*.safetensors' | wc -l)
        if [ "${src_n}" != "${dst_n}" ]; then
            echo "FATAL: staged shard count mismatch: src=${src_n} dst=${dst_n}" >&2; exit 1
        fi
        secs=$(( t1 - t0 )); [ "${secs}" -lt 1 ] && secs=1
        echo "[judge] staged ${dst_n} shards, $(( src_bytes / 1024 / 1024 )) MiB in ${secs}s (~$(( src_bytes / 1024 / 1024 / secs )) MiB/s)"
        touch "${dst}/.stage_complete"
    fi
    MODEL="${dst}"
fi
if [ "${STAGE_ONLY}" = "1" ]; then
    echo "[judge] STAGE_ONLY=1: pre-stage complete, exiting before server launch."
    exit 0
fi

# --- multi-node: form a private Ray cluster across the judge nodes (TP > 8) ----
# vLLM shards a TP>8 model across a Ray cluster (8 GPUs/node, one API server on the
# head). Each node has already pre-staged its OWN NVMe copy above. NCCL wants each
# node's own routable IP, so export VLLM_HOST_IP.
if [ "${JUDGE_NNODES}" -gt 1 ]; then
    VLLM_HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"

    # vLLM 0.24's Ray executor breaks on ray>=2.55 (vllm#45318: ActorHandleNotFoundError,
    # "not valid across Ray sessions", at EngineCore init). Our image ships ray 2.58 for
    # verl, but JUDGE nodes run ONLY vLLM (never verl), so pin ray core to the vLLM-CI
    # version here -- on BOTH head and worker, before any ray usage. --no-deps keeps every
    # other package untouched (surgical core swap). PyPI is reachable from df1 jobs.
    if [ -n "${JUDGE_RAY_VERSION:-}" ]; then
        cur="$(python3 -c 'import ray; print(ray.__version__)' 2>/dev/null || echo none)"
        if [ "${cur}" != "${JUDGE_RAY_VERSION}" ]; then
            echo "[judge] pinning ray ${cur} -> ${JUDGE_RAY_VERSION} (vllm#45318 multi-node fix)..."
            pip install --no-cache-dir --no-deps "ray==${JUDGE_RAY_VERSION}" >/tmp/ray_pin.log 2>&1 \
                || { echo "FATAL: ray pin to ${JUDGE_RAY_VERSION} failed:" >&2; tail -25 /tmp/ray_pin.log >&2; exit 1; }
            echo "[judge] ray now: $(python3 -c 'import ray; print(ray.__version__)' 2>&1)"
        else
            echo "[judge] ray already ${cur}; no pin needed."
        fi
    fi

    if [ "${JUDGE_NODE_RANK}" != "0" ]; then
        # WORKER: wait for the head's Ray port, join, then idle until it closes.
        echo "[judge] node ${JUDGE_NODE_RANK}: waiting for Ray head ${JUDGE_HEAD_ADDR}:${JUDGE_RAY_PORT}..."
        wdeadline=$(( $(date +%s) + 3000 ))
        until (exec 3<>"/dev/tcp/${JUDGE_HEAD_ADDR}/${JUDGE_RAY_PORT}") 2>/dev/null; do
            exec 3>&- 3<&- 2>/dev/null || true
            [ "$(date +%s)" -ge "${wdeadline}" ] && { echo "FATAL: Ray head not up after wait." >&2; exit 1; }
            sleep 5
        done
        exec 3>&- 3<&- 2>/dev/null || true
        ray start --address="${JUDGE_HEAD_ADDR}:${JUDGE_RAY_PORT}" --num-gpus 8
        echo "[judge] node ${JUDGE_NODE_RANK}: joined Ray; idling until head closes."
        misses=0
        while true; do
            if (exec 3<>"/dev/tcp/${JUDGE_HEAD_ADDR}/${JUDGE_RAY_PORT}") 2>/dev/null; then
                exec 3>&- 3<&- 2>/dev/null || true; misses=0
            else
                misses=$(( misses + 1 ))
                [ "${misses}" -ge 4 ] && { echo "[judge] head Ray gone; worker exiting."; break; }
            fi
            sleep 15
        done
        ray stop 2>/dev/null || true
        exit 0
    fi

    # HEAD (rank 0): start the Ray head, wait for all judge-node GPUs to register.
    echo "[judge] node 0: Ray head :${JUDGE_RAY_PORT}, expecting ${TP} GPUs across ${JUDGE_NNODES} nodes..."
    ray start --head --port "${JUDGE_RAY_PORT}" --num-gpus 8 --dashboard-host 0.0.0.0
    jdeadline=$(( $(date +%s) + 3000 ))
    until [ "$(ray status 2>/dev/null | sed -n 's#.*/\([0-9]\+\)\.0 GPU.*#\1#p' | tail -1)" -ge "${TP}" ] 2>/dev/null; do
        have="$(ray status 2>/dev/null | sed -n 's#.*/\([0-9]\+\)\.0 GPU.*#\1#p' | tail -1)"
        echo "[judge] Ray GPUs: ${have:-0}/${TP}..."
        [ "$(date +%s)" -ge "${jdeadline}" ] && { echo "FATAL: only ${have:-0}/${TP} GPUs joined." >&2; ray status || true; exit 1; }
        sleep 10
    done
    echo "[judge] Ray cluster complete (${TP} GPUs). Launching vLLM (ray backend)."
fi

case "${JUDGE_ENGINE}" in
  vllm)
    command -v vllm >/dev/null 2>&1 || { echo "FATAL: vllm not on PATH in this image." >&2; exit 127; }
    # --disable-custom-all-reduce: on H100 the intra-node 8-way custom all-reduce
    # crashes vLLM CUDA-graph capture (custom_all_reduce.cuh:455) -- see memory
    # vllm-custom-allreduce-h100; NCCL is the safe path and fine for a judge.
    LAUNCH=(vllm serve "${MODEL}"
        --served-model-name "${SERVED_NAME}"
        --tensor-parallel-size "${TP}"
        --host 0.0.0.0 --port "${PORT}"
        --gpu-memory-utilization "${MEM_FRACTION}"
        --max-model-len "${MAX_MODEL_LEN}"
        --disable-custom-all-reduce
        --trust-remote-code)
    # TP>8 shards across the Ray cluster formed above.
    if [ "${JUDGE_NNODES}" -gt 1 ]; then
        LAUNCH+=(--distributed-executor-backend ray)
    fi
    ;;
  sglang)
    python3 -c 'import sglang' 2>/dev/null || { echo "FATAL: sglang not importable in this image." >&2; exit 127; }
    LAUNCH=(python3 -m sglang.launch_server --model-path "${MODEL}"
        --served-model-name "${SERVED_NAME}"
        --tp-size "${TP}"
        --host 0.0.0.0 --port "${PORT}"
        --mem-fraction-static "${MEM_FRACTION}"
        --context-length "${MAX_MODEL_LEN}"
        --trust-remote-code)
    ;;
  *)
    echo "FATAL: unknown JUDGE_ENGINE='${JUDGE_ENGINE}' (want sglang|vllm)." >&2; exit 2 ;;
esac

echo "============== judge serve =============="
echo "engine     : ${JUDGE_ENGINE}"
echo "model      : ${MODEL}  (served as '${SERVED_NAME}')"
echo "endpoint   : http://0.0.0.0:${PORT}/v1   (tp=${TP}, mem=${MEM_FRACTION}, max_len=${MAX_MODEL_LEN})"
echo "mode       : SERVE_FOREVER"
echo "extra args : ${EXTRA_ARGS[*]:-<none>}"
echo "========================================="

LOG="/tmp/judge_${JUDGE_ENGINE}_${PORT}.log"

# On startup failure, surface the ROOT CAUSE, not just the shutdown tail: grep the
# engine log for the FIRST real error/traceback (the shutdown cascade at the very
# end otherwise hides it), then print a large tail for context.
dump_log() {
    echo "----- ${JUDGE_ENGINE} log: first error/traceback lines -----" >&2
    grep -naiE 'error|traceback|exception|assert|not support|unsupported|no module|out of memory|oom|\bmla\b|sparse|attention backend|flashinfer|placement group|RayWorker|raise ' "${LOG}" 2>/dev/null | head -60 >&2 || true
    echo "----- ${JUDGE_ENGINE} log: last 300 lines -----" >&2
    tail -n 300 "${LOG}" >&2 || true
}
"${LAUNCH[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "${LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    kill "${WATCHDOG_PID:-}" 2>/dev/null || true
    [ "${JUDGE_NNODES}" -gt 1 ] && ray stop 2>/dev/null || true
}
trap cleanup EXIT

# --- wait for readiness (both engines expose /health) ------------------------
echo "[judge] waiting up to ${HEALTH_TIMEOUT}s for the server to become healthy..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "FATAL: ${JUDGE_ENGINE} server exited during startup. Last log lines:" >&2
        dump_log
        exit 1
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "FATAL: server not healthy after ${HEALTH_TIMEOUT}s. Last log lines:" >&2
        dump_log
        exit 1
    fi
    sleep 5
done
echo "[judge] healthy. Models: $(curl -sf "http://127.0.0.1:${PORT}/v1/models" || true)"

# --- serve mode: publish the endpoint (for cross-node discovery) and block ----
if [ -n "${JUDGE_RENDEZVOUS:-}" ]; then
    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    IP="${IP:-127.0.0.1}"
    mkdir -p "$(dirname "${JUDGE_RENDEZVOUS}")"
    echo "http://${IP}:${PORT}/v1" > "${JUDGE_RENDEZVOUS}"
    echo "[judge] published endpoint http://${IP}:${PORT}/v1 -> ${JUDGE_RENDEZVOUS}"
fi

# Self-exit watchdog (for the co-located single-job topology). A dedicated judge
# on its OWN nodes must NOT idle its GPUs to the job timeout once training finishes.
# The dispatcher signals completion by creating JUDGE_EXIT_SENTINEL (on the shared UC
# rendezvous dir); we poll for it and stop the server. JUDGE_MAX_LIFETIME is a backstop
# for the pathological case where training dies WITHOUT writing the sentinel (e.g. a
# node OOM-kill skips the trap) -- else the judge would idle until the hard job kill.
# When the head server stops, cleanup() runs `ray stop`, the head's Ray port closes,
# and each worker node (which polls that port) exits on its own.
if [ -n "${JUDGE_EXIT_SENTINEL:-}" ] || [ "${JUDGE_MAX_LIFETIME:-0}" -gt 0 ]; then
    ( wstart=$(date +%s)
      while kill -0 "${SERVER_PID}" 2>/dev/null; do
          if [ -n "${JUDGE_EXIT_SENTINEL:-}" ] && [ -f "${JUDGE_EXIT_SENTINEL}" ]; then
              echo "[judge] exit sentinel seen (${JUDGE_EXIT_SENTINEL}); stopping server."
              kill "${SERVER_PID}" 2>/dev/null || true; break
          fi
          if [ "${JUDGE_MAX_LIFETIME:-0}" -gt 0 ] \
             && [ "$(( $(date +%s) - wstart ))" -ge "${JUDGE_MAX_LIFETIME}" ]; then
              echo "[judge] max lifetime ${JUDGE_MAX_LIFETIME}s reached; stopping server."
              kill "${SERVER_PID}" 2>/dev/null || true; break
          fi
          sleep 15
      done ) &
    WATCHDOG_PID=$!
    echo "[judge] self-exit watchdog ${WATCHDOG_PID} (sentinel=${JUDGE_EXIT_SENTINEL:-none} max_life=${JUDGE_MAX_LIFETIME:-0}s)."
fi

echo "[judge] serving; waiting on ${JUDGE_ENGINE} (pid ${SERVER_PID})."
wait "${SERVER_PID}"
