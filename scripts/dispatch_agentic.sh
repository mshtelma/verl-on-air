#!/usr/bin/env bash
# =============================================================================
# air/53 RANK DISPATCHER — one single job, one image, two roles.
#
# df1 has NO cross-JOB connectivity and one docker image per job, so the full
# agentic run (fully-async GRPO training + a self-hosted LLM judge) must live in
# ONE job. AI Runtime runs this `command` ONCE PER NODE with the topology injected
# (NUM_NODES / POD_RANK / MASTER_ADDR / MASTER_PORT). This script reads POD_RANK
# and sends each node to its role:
#
#   ranks [0 .. TRAINING_NODES-1]      -> run_grpo_fully_async.sh (GRPO training)
#   ranks [TRAINING_NODES .. NUM_NODES-1] -> serve_judge.sh (GLM-5.3 judge, TP=8*JUDGE_NODES)
#
# The two halves form SEPARATE Ray clusters (training head = global rank 0 on
# port 6379 with ray 2.58; judge head = first judge node on port 6380 with ray
# pinned to 2.48 — see serve_judge.sh). They talk only over HTTP: the judge head
# publishes its OpenAI endpoint to a shared UC rendezvous file, and the training
# reward-loop workers read it as JUDGE_BASE_URL.
#
# WHY NNODES is overridden for training: verl's Ray head waits for NNODES nodes.
# The injected NUM_NODES is 4 (2 train + 2 judge), but the training cluster must
# wait for only its 2 nodes, so we export NNODES=TRAINING_NODES; the judge nodes
# never join verl's cluster (they run serve_judge.sh), so verl sees exactly 2.
#
# RENDEZVOUS (shared UC dir, job-unique by MASTER_ADDR:MASTER_PORT):
#   judge_ray_head  <- judge head IP        (judge worker reads it to join Ray)
#   judge_endpoint  <- http://ip:port/v1    (training reads it as JUDGE_BASE_URL;
#                                             written by serve_judge.sh when healthy)
#   training_done   <- sentinel             (training rank 0 writes it on exit;
#                                             judge head's watchdog stops on it)
# =============================================================================
set -xeuo pipefail

export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES="${NUM_NODES:-1}"
POD_RANK="${POD_RANK:-${NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-0}"

TRAINING_NODES="${TRAINING_NODES:-2}"           # nodes running GRPO (rest serve the judge)
JUDGE_NODES=$(( NUM_NODES - TRAINING_NODES ))
JUDGE_WAIT_TIMEOUT="${JUDGE_WAIT_TIMEOUT:-2400}" # how long a role waits on a rendezvous file

if [ "${JUDGE_NODES}" -lt 1 ]; then
    echo "FATAL: TRAINING_NODES(${TRAINING_NODES}) >= NUM_NODES(${NUM_NODES}); no judge node left." >&2
    exit 1
fi

# Job-unique rendezvous dir (rank-0 IP:port is stable across this job's nodes and
# effectively unique per job on df1's dynamic pod IPs).
RDV="${RENDEZVOUS_ROOT:-/Volumes/main/mshtelma/verl/rendezvous}/${MASTER_ADDR}_${MASTER_PORT}"
mkdir -p "${RDV}"

# --- rendezvous helpers ------------------------------------------------------
rdv_put() {  # rdv_put <file> <value>  (atomic: temp + mv)
    local f="$1" v="$2"
    printf '%s\n' "${v}" > "${f}.tmp.$$"
    mv -f "${f}.tmp.$$" "${f}"
}
rdv_wait() {  # rdv_wait <file> <timeout_s> -> prints value | returns 1 on timeout
    local f="$1" deadline=$(( $(date +%s) + ${2:-1800} ))
    until [ -s "${f}" ]; do
        [ "$(date +%s)" -ge "${deadline}" ] && return 1
        sleep 5
    done
    cat "${f}"
}
my_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }

echo "[dispatch] rank ${POD_RANK}/${NUM_NODES}  training_nodes=${TRAINING_NODES} judge_nodes=${JUDGE_NODES}  rdv=${RDV}"

# =============================================================================
if [ "${POD_RANK}" -lt "${TRAINING_NODES}" ]; then
    # ---------------------------- TRAINING ROLE ------------------------------
    # verl's Ray head waits for TRAINING_NODES (not the injected NUM_NODES).
    export NNODES="${TRAINING_NODES}"
    # NODE_RANK stays = POD_RANK (0..TRAINING_NODES-1); MASTER_ADDR (=global rank 0)
    # IS the training head, so the launcher's Ray bootstrap needs no change.

    # Point the LLM-judge reward at our own scripts unless the caller overrode them.
    export FUNCTION_TOOL_PATH="${FUNCTION_TOOL_PATH:-${HERE}/tools/calc_tool.py}"
    export CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${HERE}/reward/judge_reward.py}"

    # Rank 0 owns the training-done sentinel: clear any stale one, and (via an EXIT
    # trap so it fires on success, failure, OR signal) tell the judge to self-exit
    # when training ends. Only rank 0 writes it -- a training worker exiting early
    # must not prematurely kill the judge.
    if [ "${POD_RANK}" = "0" ]; then
        rm -f "${RDV}/training_done" 2>/dev/null || true
        trap 'rdv_put "${RDV}/training_done" done' EXIT
    fi

    # Hand the LLM-judge endpoint to the reward loop. We do NOT rely on this export
    # reaching the reward-loop Ray actors: run3 showed Ray does not reliably carry a
    # driver `export` into actor processes, so judge_reward.py re-resolves the URL at
    # CALL time. We give it three ways to find the judge, most-robust last:
    #   JUDGE_BASE_URL         - this export (works iff Ray propagates it)
    #   JUDGE_ENDPOINT_FILE    - the exact rendezvous file path (this export)
    #   RENDEZVOUS_ROOT/MASTER_ADDR_MASTER_PORT/judge_endpoint  - reconstructed by
    #       the reward fn from container-level vars that reach EVERY process (set
    #       before Ray starts), so it survives even if both exports are dropped.
    echo "[dispatch] rank ${POD_RANK} TRAINING: waiting for judge endpoint..."
    JUDGE_BASE_URL="$(rdv_wait "${RDV}/judge_endpoint" "${JUDGE_WAIT_TIMEOUT}")" || {
        echo "FATAL: judge endpoint not published within ${JUDGE_WAIT_TIMEOUT}s." >&2; exit 1; }
    export JUDGE_BASE_URL
    export JUDGE_ENDPOINT_FILE="${RDV}/judge_endpoint"
    export RENDEZVOUS_ROOT     # ensure the reconstruction fallback matches this RDV
    export JUDGE_MODEL="${JUDGE_MODEL:-judge}"
    echo "[dispatch] rank ${POD_RANK} TRAINING: JUDGE_BASE_URL=${JUDGE_BASE_URL} JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE}"

    # NOT exec: keep this process as parent so the rank-0 EXIT trap fires after the
    # launcher returns (or if it is killed).
    bash "${HERE}/run_grpo_fully_async.sh"
    exit $?
fi

# =============================================================================
# ------------------------------- JUDGE ROLE ----------------------------------
# serve_judge.sh forms a TP=8*JUDGE_NODES Ray cluster across the judge nodes and
# serves the OpenAI endpoint. Map the global topology to serve_judge.sh's JUDGE_*:
export JUDGE_NNODES="${JUDGE_NODES}"
export JUDGE_NODE_RANK=$(( POD_RANK - TRAINING_NODES ))   # judge-local rank (head = 0)
export VALIDATE_ONLY=0                                     # serve forever
export JUDGE_RENDEZVOUS="${RDV}/judge_endpoint"            # serve_judge.sh publishes here when healthy
export JUDGE_EXIT_SENTINEL="${RDV}/training_done"          # self-exit when training signals done

if [ "${JUDGE_NODE_RANK}" = "0" ]; then
    # JUDGE HEAD: publish own IP so the judge worker(s) can join the Ray cluster.
    IP="$(my_ip)"; IP="${IP:-127.0.0.1}"
    rm -f "${RDV}/judge_ray_head" "${RDV}/judge_endpoint" 2>/dev/null || true
    export JUDGE_HEAD_ADDR="${IP}"
    rdv_put "${RDV}/judge_ray_head" "${IP}"
    echo "[dispatch] rank ${POD_RANK} JUDGE HEAD: ip=${IP}"
else
    # JUDGE WORKER: discover the head IP from the rendezvous.
    echo "[dispatch] rank ${POD_RANK} JUDGE WORKER: waiting for judge head ip..."
    JUDGE_HEAD_ADDR="$(rdv_wait "${RDV}/judge_ray_head" "${JUDGE_WAIT_TIMEOUT}")" || {
        echo "FATAL: judge head ip not published within ${JUDGE_WAIT_TIMEOUT}s." >&2; exit 1; }
    export JUDGE_HEAD_ADDR
    echo "[dispatch] rank ${POD_RANK} JUDGE WORKER: head=${JUDGE_HEAD_ADDR}"
fi

exec bash "${HERE}/serve_judge.sh"
