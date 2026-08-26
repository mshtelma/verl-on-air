#!/usr/bin/env bash
# =============================================================================
# Ray cluster bootstrap across AI Runtime nodes.
#
# verl is a Ray application, NOT a torchrun application. AI Runtime runs your
# `command:` ONCE PER NODE and injects rendezvous env vars, but it does not
# start Ray and Ray does not auto-discover peers. So we form the cluster:
#
#   NODE_RANK 0      -> `ray start --head`, wait for all GPUs to register,
#                       then fall through and launch verl (which attaches to
#                       the local head).
#   NODE_RANK != 0   -> `ray start --address=<head>`, then poll the head and
#                       exit 0 when it disappears.
#
# Two hard-won details:
#
#  * DO NOT use `ray start --block` on workers. When the head finishes, a
#    blocked worker hangs forever, the air job stays RUNNING, and you keep
#    paying for 8 idle H100s until the timeout. We poll the GCS port instead
#    and exit cleanly.
#
#  * The head's teardown MUST be a `trap ... EXIT`, not a line after the
#    launch. With `set -e` + `pipefail`, a verl failure exits immediately and
#    any post-launch cleanup line is skipped — leaving the worker's poll loop
#    alive and the job wedged.
#
# Injected by AI Runtime: NUM_NODES, LOCAL_WORLD_SIZE, WORLD_SIZE,
#                         POD_RANK (also as NODE_RANK), LOCAL_ADDR,
#                         MASTER_ADDR, MASTER_PORT.
# =============================================================================

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# ray_worker_wait_and_exit <nnodes> <gpus_per_node> <head_addr> <node_rank>
# Joins the head, blocks until training is over, then `exit 0`. Never returns.
ray_worker_wait_and_exit() {
  local nnodes="$1" gpus="$2" head="$3" rank="$4"
  echo "[node ${rank}] joining Ray head ${head}:${RAY_PORT}"
  sleep 15   # give the head a head start before the first attempt

  local joined=0 i
  for i in $(seq 1 30); do
    if ray start --address="${head}:${RAY_PORT}" --num-gpus="${gpus}"; then
      joined=1; break
    fi
    echo "[node ${rank}] head not up yet (attempt ${i}/30)"; sleep 10
  done
  if [ "${joined}" != "1" ]; then
    echo "[node ${rank}] FATAL: could not join Ray head ${head}:${RAY_PORT}" >&2
    exit 1
  fi

  echo "[node ${rank}] joined; waiting for the head to finish training..."
  local miss=0
  while true; do
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
    sleep 15
  done
}

# ray_start_head <nnodes> <gpus_per_node> <head_addr>
# Starts the head and blocks until every node's GPUs have registered.
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

  echo "[head] waiting for ${want} GPUs across ${nnodes} nodes..."
  local i have
  for i in $(seq 1 90); do
    have=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    echo "[head] cluster GPUs: ${have}/${want}"
    [ "${have}" -ge "${want}" ] && { echo "[head] cluster ready"; return 0; }
    sleep 10
  done
  echo "[head] FATAL: only ${have}/${want} GPUs registered after 15 min" >&2
  return 1
}

# Install on the head only, AFTER the worker branch has exited.
ray_install_cleanup_trap() {
  cleanup() {
    local rc=$?
    echo "[head] training exited rc=${rc}; stopping Ray so workers can drain"
    ray stop --force 2>/dev/null || true
    exit "${rc}"
  }
  trap cleanup EXIT
}
