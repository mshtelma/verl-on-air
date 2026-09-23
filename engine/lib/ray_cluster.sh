#!/usr/bin/env bash
# =============================================================================
# Ray cluster bootstrap across AI Runtime nodes.
#
# verl is a Ray application, NOT a torchrun application. AI Runtime runs your
# `command:` ONCE PER NODE and injects rendezvous env vars, but it does not
# start Ray and Ray does not auto-discover peers. So we form the cluster:
#
#   NODE_RANK 0      -> install the cleanup trap, `ray start --head`, wait for all
#                       GPUs to register, then fall through and launch verl (which
#                       attaches to the local head).
#   NODE_RANK != 0   -> `ray start --address=<head>`, then drain: exit when the head
#                       says it is done, stops heartbeating, or disappears.
#
# Hard-won details:
#
#  * DO NOT use `ray start --block` on workers. When the head finishes, a
#    blocked worker hangs forever, the air job stays RUNNING, and you keep
#    paying for 8 idle H100s until the timeout. Workers poll instead.
#
#  * The head's teardown MUST be a `trap ... EXIT`, installed BEFORE `ray start
#    --head`: with `set -e` + `pipefail` any failure exits at once -- including a
#    bootstrap that never saw every GPU -- and Ray daemons left running keep the
#    workers' drain loop (and the job) alive until the job timeout.
#
#  * A head killed without running its trap (OOM-kill, SIGKILL) leaves its Ray
#    daemons -- and their open port -- behind. So with a shared rendezvous dir
#    (VOA_RDV_DIR, set by dispatch_agentic.sh) the head also writes a heartbeat,
#    and a worker that sees it go stale exits 1 instead of waiting for the port.
#
# Injected by AI Runtime: NUM_NODES, LOCAL_WORLD_SIZE, WORLD_SIZE,
#                         POD_RANK (also as NODE_RANK), LOCAL_ADDR,
#                         MASTER_ADDR, MASTER_PORT.
# =============================================================================

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_JOIN_DELAY_S="${RAY_JOIN_DELAY_S:-15}"        # head start before a worker's first attempt
RAY_JOIN_ATTEMPTS="${RAY_JOIN_ATTEMPTS:-30}"      # ...then this many, RAY_JOIN_INTERVAL_S apart
RAY_JOIN_INTERVAL_S="${RAY_JOIN_INTERVAL_S:-10}"
RAY_NODES_TIMEOUT_S="${RAY_NODES_TIMEOUT_S:-900}" # head: every GPU registered within this
RAY_DRAIN_POLL_S="${RAY_DRAIN_POLL_S:-15}"        # worker: how often it checks on the head
RAY_HEARTBEAT_S="${RAY_HEARTBEAT_S:-30}"          # head: heartbeat period
RAY_HEARTBEAT_STALE_S="${RAY_HEARTBEAT_STALE_S:-600}"   # worker: head presumed dead after this

_ray_rdv() { printf '%s' "${VOA_RDV_DIR:-}"; }

# ray_worker_wait_and_exit <nnodes> <gpus_per_node> <head_addr> <node_rank>
# Joins the head, drains until training is over, then exits. Never returns:
#   exit 0  the head wrote ray_head_done, or its port closed (training finished)
#   exit 1  it never came up, or it stopped heartbeating with its port still open
ray_worker_wait_and_exit() {
  local nnodes="$1" gpus="$2" head="$3" rank="$4" rdv
  rdv="$(_ray_rdv)"
  echo "[node ${rank}] joining Ray head ${head}:${RAY_PORT}"
  sleep "${RAY_JOIN_DELAY_S}"

  local joined=0 i
  for i in $(seq 1 "${RAY_JOIN_ATTEMPTS}"); do
    if ray start --address="${head}:${RAY_PORT}" --num-gpus="${gpus}"; then
      joined=1; break
    fi
    echo "[node ${rank}] head not up yet (attempt ${i}/${RAY_JOIN_ATTEMPTS})"; sleep "${RAY_JOIN_INTERVAL_S}"
  done
  if [ "${joined}" != "1" ]; then
    echo "[node ${rank}] FATAL: could not join Ray head ${head}:${RAY_PORT}" >&2
    exit 1
  fi

  echo "[node ${rank}] joined; draining until the head is done (rdv=${rdv:-none})..."
  local miss=0 beat age
  while true; do
    if [ -n "${rdv}" ] && [ -f "${rdv}/ray_head_done" ]; then
      echo "[node ${rank}] head done ($(cat "${rdv}/ray_head_done" 2>/dev/null)); exiting worker cleanly"
      ray stop --force 2>/dev/null || true
      exit 0
    fi
    if [ -n "${rdv}" ] && beat="$(cat "${rdv}/ray_head_alive" 2>/dev/null)" && [[ "${beat}" =~ ^[0-9]+$ ]]; then
      age=$(( $(date +%s) - beat ))
      if [ "${age}" -gt "${RAY_HEARTBEAT_STALE_S}" ]; then
        echo "[node ${rank}] FATAL: the head's last heartbeat is ${age}s old (limit ${RAY_HEARTBEAT_STALE_S}s):" \
             "it died without cleaning up; exiting instead of waiting for the job timeout" >&2
        ray stop --force 2>/dev/null || true
        exit 1
      fi
    fi
    if python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(5)
sys.exit(0 if s.connect_ex(('${head}', ${RAY_PORT})) == 0 else 1)" 2>/dev/null; then
      miss=0
    else
      miss=$((miss + 1))
      echo "[node ${rank}] head unreachable (${miss}/3)"
      if [ "${miss}" -ge 3 ]; then
        echo "[node ${rank}] head gone -> training finished; exiting worker cleanly"
        ray stop --force 2>/dev/null || true
        exit 0
      fi
    fi
    sleep "${RAY_DRAIN_POLL_S}"
  done
}

# ray_start_head <nnodes> <gpus_per_node> <head_addr>
# Starts the head and blocks until every node's GPUs have registered. Install
# ray_install_cleanup_trap FIRST: a bootstrap that fails here must still stop Ray.
ray_start_head() {
  local nnodes="$1" gpus="$2" head="$3"
  local want=$((nnodes * gpus))

  echo "[head] starting Ray head on ${head}:${RAY_PORT}"
  ray start --head \
    --node-ip-address="${head}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${gpus}"

  echo "[head] waiting up to ${RAY_NODES_TIMEOUT_S}s for ${want} GPUs across ${nnodes} nodes..."
  local deadline=$(( $(date +%s) + RAY_NODES_TIMEOUT_S )) have=0
  while :; do
    have=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    echo "[head] cluster GPUs: ${have}/${want}"
    [ "${have}" -ge "${want}" ] && { echo "[head] cluster ready"; return 0; }
    [ "$(date +%s)" -ge "${deadline}" ] && break
    sleep 10
  done
  echo "[head] FATAL: only ${have}/${want} GPUs registered within ${RAY_NODES_TIMEOUT_S}s" >&2
  return 1
}

# ray_install_cleanup_trap: on the head only, AFTER the worker branch has exited and
# BEFORE ray_start_head. On exit -- success, failure or a signal -- it stops Ray and
# tells the workers (ray_head_done) so they drain at once. With a rendezvous dir it
# also starts the head heartbeat the workers watch.
ray_install_cleanup_trap() {
  local rdv
  rdv="$(_ray_rdv)"
  RAY_HEARTBEAT_PID=""
  if [ -n "${rdv}" ]; then
    mkdir -p "${rdv}"
    rm -f "${rdv}/ray_head_done" 2>/dev/null || true
    # `$$` is this launcher even inside the subshell: the heartbeat dies with it, SIGKILL included.
    ( while kill -0 "$$" 2>/dev/null; do
        date +%s > "${rdv}/ray_head_alive.tmp.$$" && mv -f "${rdv}/ray_head_alive.tmp.$$" "${rdv}/ray_head_alive"
        sleep "${RAY_HEARTBEAT_S}"
      done ) >/dev/null 2>&1 &
    RAY_HEARTBEAT_PID=$!
  fi
  cleanup() {
    local rc=$?
    echo "[head] exiting rc=${rc}; stopping Ray so workers can drain"
    [ -n "${RAY_HEARTBEAT_PID:-}" ] && kill "${RAY_HEARTBEAT_PID}" 2>/dev/null
    if [ -n "$(_ray_rdv)" ]; then
      printf 'rc=%s\n' "${rc}" > "$(_ray_rdv)/ray_head_done.tmp.$$" \
        && mv -f "$(_ray_rdv)/ray_head_done.tmp.$$" "$(_ray_rdv)/ray_head_done"
    fi
    ray stop --force 2>/dev/null || true
    exit "${rc}"
  }
  trap cleanup EXIT
}
