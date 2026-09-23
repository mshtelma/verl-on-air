#!/usr/bin/env bash
# =============================================================================
# verl GRPO — FULLY ASYNC / DISAGGREGATED rollout (Megatron backend).
#
# This is a SEPARATE entrypoint from run_grpo_megatron.sh (the sync launcher,
# rungs 1-4). The fully-async path decouples the Rollouter and the Trainer into
# distinct processes on DISJOINT GPUs, with a MessageQueue between them and NCCL
# weight sync — see verl/experimental/fully_async_policy.
#
#   python -m verl.experimental.fully_async_policy.fully_async_main
#          --config-name=fully_async_ppo_megatron_trainer   <overrides...>
#
# Modeled VERBATIM on verl v0.9.0's own example:
#   verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh
# adapted only for (a) air's parameters: plumbing and (b) Qwen3.5 correctness
# (Gated-DeltaNet has no THD packing -> BSHD everywhere: use_remove_padding=False
# and use_dynamic_bsz=False; both differ from the recipe's config defaults).
#
# WHY a different mechanism than run_grpo_megatron.sh's separate_async block:
#   trainer.v1.trainer_mode=separate_async places the standalone rollout at
#   start_rank=hybrid_num_replicas; with hybrid_engine=False that is 0, so the
#   rollout collides with the trainer on the SAME GPUs (observed: vLLM OOM,
#   ~31 GiB Megatron resident on the rollout node). fully_async_policy instead
#   takes TOP-LEVEL rollout.nnodes / rollout.n_gpus_per_node and (with
#   use_dynamic_resource_scheduling=False, the default) keeps the trainer GPUs
#   rollout-free -> real disaggregation.
#
# KEY CONFIG SURFACE (from the recipe README + config):
#   trainer.nnodes / trainer.n_gpus_per_node   -> Trainer resources
#   rollout.nnodes  / rollout.n_gpus_per_node  -> Rollouter resources (TOP-LEVEL)
#   rollout.total_rollout_steps                -> total rollout SAMPLES (sizing)
#   data.train_batch_size=0, data.gen_batch_size=1   -> streaming (not effective)
#   async_training.trigger_parameter_sync_step -> local updates between syncs
#   async_training.require_batches             -> #mini-batches fetched per update
#   async_training.staleness_threshold         -> 0 sync, >0 async freshness
#   Between two syncs the Trainer consumes
#     trigger_parameter_sync_step * require_batches * ppo_mini_batch_size samples.
# =============================================================================
set -xeuo pipefail

# The v5 image puts ray/verl/torch in /opt/venv/bin; prepend only if missing.
command -v ray >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"

# air hosts run a FIPS kernel; non-FIPS crypto aborts on SSL init.
export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0
# Fully-async requires vLLM server mode (AgentLoop) -> v1 engine.
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# classic Megatron wants =1 for comm/compute overlap (this launcher is classic).
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 122B checkpoint save is memory-tight (~76 GiB reserved during training). Reduce
# allocator fragmentation so the dist-checkpoint save buffers fit (per the CUDA
# OOM hint from an observed save-time failure).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/hparams.sh
source "${HERE}/../lib/hparams.sh"
# shellcheck source=../lib/ray_cluster.sh
source "${HERE}/../lib/ray_cluster.sh"
# shellcheck source=../lib/run_identity.sh
source "${HERE}/../lib/run_identity.sh"
# shellcheck source=../lib/run_driver.sh
source "${HERE}/../lib/run_driver.sh"

hp_dump

# Every engine knob this job sets, typed and checked against what THIS launcher reads -- a knob only
# the other mode reads, or a misspelt one, stops the job here instead of being ignored
# (engine/lib/preflight.py). Booleans come back as exactly True/False.
KNOB_EXPORTS="$(python3 "${HERE}/../lib/preflight.py" knobs --mode async)" || exit 1
eval "${KNOB_EXPORTS}"

# =============================================================================
# Topology — split the node's GPUs between the Trainer and the Rollouter.
# The canonical geo3k example uses 1 node, 8 GPUs -> 4 rollout + 4 train.
# For multi-node, trainer.nnodes and rollout.nnodes are ADDITIVE (each side gets
# its own whole nodes); set N_GPUS_ROLLOUT=NGPUS_PER_NODE for whole-node splits.
# =============================================================================
NNODES="${NNODES:-${NUM_NODES:-1}}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}"
NODE_RANK="${NODE_RANK:-${POD_RANK:-0}}"
HEAD_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Two split modes:
#   WHOLE-NODE (multi-node): set ROLLOUT_NNODES>=1. The Rollouter gets
#     ROLLOUT_NNODES whole nodes and the Trainer the rest; each side uses all
#     NGPUS_PER_NODE. This is what 35B/122B want (a full trainer node).
#   WITHIN-NODE (single node): leave ROLLOUT_NNODES=0 and set N_GPUS_ROLLOUT<8;
#     the one node is split N_GPUS_ROLLOUT rollout + (rest) train.
# Either way we emit trainer.nnodes/n_gpus_per_node + rollout.nnodes/n_gpus_per_node,
# which is the same config shape verl's fully_async recipe expects.
ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"
if [ "${ROLLOUT_NNODES}" -ge 1 ]; then
  SPLIT_MODE="whole-node"
  TRAINER_NNODES=$(( NNODES - ROLLOUT_NNODES ))
  TRAINER_N_GPUS=${NGPUS_PER_NODE}
  ROLLOUT_N_GPUS=${NGPUS_PER_NODE}
  if [ "${TRAINER_NNODES}" -lt 1 ]; then
    echo "FATAL: ROLLOUT_NNODES(${ROLLOUT_NNODES}) leaves no trainer node (NNODES=${NNODES})." >&2
    exit 1
  fi
else
  SPLIT_MODE="within-node"
  TRAINER_NNODES=${NNODES}
  ROLLOUT_NNODES=${NNODES}
  N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
  ROLLOUT_N_GPUS=${N_GPUS_ROLLOUT}
  TRAINER_N_GPUS=$(( NGPUS_PER_NODE - N_GPUS_ROLLOUT ))
  if [ "${TRAINER_N_GPUS}" -lt 1 ] || [ "${ROLLOUT_N_GPUS}" -lt 1 ]; then
    echo "FATAL: within-node split needs >=1 GPU each (train=${TRAINER_N_GPUS} rollout=${ROLLOUT_N_GPUS})." >&2
    exit 1
  fi
fi
TRAINING_GPUS=$(( TRAINER_NNODES * TRAINER_N_GPUS ))

# --- Parallelism (classic Megatron; offloaded optimizer) --------------------
# 9B dense smoke: TP=2 on 4 training GPUs -> DP=2. Rollout TP=1 -> N replicas.
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-1}"
ETP="${ETP:-1}"
GEN_TP="${GEN_TP:-1}"
OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1}"

# =============================================================================
# Parameters from air
# =============================================================================
MODEL_PATH="$(hp model_name "Qwen/Qwen3.5-9B")"
TRAIN_FILES="$(hp train_files "/Volumes/main/mshtelma/verl/data/geo3k/train.parquet")"
VAL_FILES="$(hp val_files "/Volumes/main/mshtelma/verl/data/geo3k/test.parquet")"
CKPT_DIR="$(hp output_dir "/Volumes/main/mshtelma/verl/ckpt/qwen3_5-9b-fully-async")"
TOTAL_EPOCHS="$(hp total_epochs 1)"
PPO_MINI_BATCH_SIZE="$(hp ppo_mini_batch_size 16)"
ROLLOUT_N="$(hp rollout_n 4)"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}"   # rollout sampling temperature (verl default 1.0; raise >1 for more GRPO exploration -> denser reward)
ROLLOUT_PREFIX_CACHING="${ROLLOUT_PREFIX_CACHING:-False}"   # vLLM prefix cache; default OFF (prior runs unchanged). ON reuses the shared system-prompt + prior-turn KV across multi-turn -> big rollout speedup. SAFE only if verl flushes the cache on each weight sync.
MAX_PROMPT_LEN="$(hp max_prompt_length 1024)"
MAX_RESPONSE_LEN="$(hp max_response_length 2048)"
ACTOR_LR="$(hp actor_lr 1e-6)"

# Run identity: this run writes to <output_dir>/<RUN_ID>/, and resuming is explicit (RESUME).
resolve_run_identity "${CKPT_DIR}" || exit 1   # CKPT_DIR becomes <output_dir>/<RUN_ID>

# --- fully-async knobs (env-overridable; smoke defaults) --------------------
TOTAL_ROLLOUT_STEPS="$(hp total_rollout_steps 64)"   # total rollout SAMPLES
# LR horizon MUST be explicit here: fully-async streams (data.train_batch_size=0),
# so verl cannot derive total_training_steps from the dataloader the way the sync
# launcher does -> Megatron's OptimizerParamScheduler asserts lr_decay_steps>0 and
# the Trainer dies at setup. The canonical geo3k example hardcodes it
# (lr_decay_steps == total_rollout_steps); mirror that. For a smoke this keeps LR
# ~constant over the handful of optimizer steps (well inside the horizon).
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TOTAL_ROLLOUT_STEPS}}"
TRIGGER_SYNC_STEP="${TRIGGER_SYNC_STEP:-2}"          # local updates between syncs
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"              # mini-batches fetched per update
STALENESS="${STALENESS:-0.1}"                        # 0 sync, >0 async
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"

# --- agentic / multi-turn tool-calling (opt-in; default OFF => single-turn) --
# When MULTI_TURN=True the rollout runs verl's ToolAgentLoop: the model emits
# <tool_call> blocks, verl executes the tool, feeds the result back, and loops
# up to MAX_TURNS. Tools come from FUNCTION_TOOL_PATH (stateless @function_tool
# callables, offered to every agent_name="tool_agent" sample). The response the
# actor trains on is the WHOLE episode (all assistant turns + tool responses),
# so the rollout response budget must cover it: verl needs rollout.prompt_length
# and rollout.response_length sized for the whole trajectory, not one turn
# (mirrors grpo_qwen35_35b_megatron_async.sh's length arithmetic).
MULTI_TURN="${MULTI_TURN:-False}"
MAX_TURNS="${MAX_TURNS:-4}"
FUNCTION_TOOL_PATH="${FUNCTION_TOOL_PATH:-}"          # python file of @function_tool defs
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-}"              # yaml of stateful BaseTool defs (optional)
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-${HERE}/agent_loops.yaml}"  # agent-loop registry (multi-turn): default registers the role-span ToolAgentLoop
# Ray workers import engine/train modules by name (the agent-loop registry): keep it on PYTHONPATH.
case ":${PYTHONPATH:-}:" in *":${HERE}:"*) ;; *) PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" ;; esac
export PYTHONPATH

TOOL_FORMAT="${TOOL_FORMAT:-hermes}"                  # tool-call parser (Qwen3.5 = hermes)
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"           # parallel AgentLoopWorker actors
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-512}" # per tool-response token cap

# --- reward manager (opt-in; default = verl's data_source-dispatched scorer) --
# REWARD_MANAGER=rate_limited + a CUSTOM_REWARD_PATH is the LLM-judge path: the
# fully-async RewardLoopManager runs the custom (async) compute_score with
# concurrency / RPM / TPM limits suited to an external judge endpoint. The
# reward.max_* knobs are NOT in the reward schema, so add them with '+'.
# max_concurrent DEFAULTS TO 1 (serial) inside verl -> always set it for throughput.
REWARD_MANAGER="${REWARD_MANAGER:-}"                  # e.g. rate_limited | naive | dapo
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-}"
CUSTOM_REWARD_NAME="${CUSTOM_REWARD_NAME:-compute_score}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-}"
REWARD_MAX_RPM="${REWARD_MAX_RPM:-}"
REWARD_MAX_TPM="${REWARD_MAX_TPM:-}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-}"

# Episode-length arithmetic. Single-turn: response budget = MAX_RESPONSE_LEN.
# Multi-turn: the whole trajectory can be up to (prompt+response)*turns tokens;
# verl's response_length is the trajectory MINUS the initial prompt.
if [ "${MULTI_TURN}" = "True" ]; then
  EPISODE_LEN=$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * MAX_TURNS ))
  RESP_BUDGET=$(( EPISODE_LEN - MAX_PROMPT_LEN ))
  # vLLM context must hold prompt + full multi-turn response.
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${EPISODE_LEN}}"
else
  EPISODE_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))
  RESP_BUDGET="${MAX_RESPONSE_LEN}"
fi

PROJECT_NAME="${PROJECT_NAME:-$(hp project_name verl-on-air)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(hp experiment_name grpo-fully-async)}"

# Per-sync sample budget (README): trigger * require * ppo_mini.
SYNC_SAMPLES=$(( TRIGGER_SYNC_STEP * REQUIRE_BATCHES * PPO_MINI_BATCH_SIZE ))

# --- completion-certificate preflight ----------------------------------------
# Success is decided by engine/lib/run_certificate.py (see the exit guard at the
# bottom): the run must reach an EXACT planned final parameter version and leave a
# verified checkpoint there. Refuse, in seconds, any budget that could not produce
# one -- instead of discovering it after a multi-hour run.
if [ "${SYNC_SAMPLES}" -lt 1 ] || [ $(( TOTAL_ROLLOUT_STEPS % SYNC_SAMPLES )) -ne 0 ]; then
  echo "FATAL: total_rollout_steps(${TOTAL_ROLLOUT_STEPS}) must be a multiple of samples/sync" \
       "(${SYNC_SAMPLES} = TRIGGER_SYNC_STEP*REQUIRE_BATCHES*ppo_mini_batch_size): the trainer" \
       "drops a partial last batch, so the final version would not be exact." >&2
  exit 1
fi
EXPECTED_FINAL=$(( TOTAL_ROLLOUT_STEPS / SYNC_SAMPLES ))   # = weight syncs = final global_step_N
SAVE_FREQ="${SAVE_FREQ:--1}"
if ! [[ "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  if [ "${ALLOW_UNCERTIFIED:-False}" = "True" ]; then
    echo "WARNING: SAVE_FREQ=${SAVE_FREQ} -> no checkpoints; this run's success CANNOT be certified" \
         "(ALLOW_UNCERTIFIED=1): its exit code will be reported as-is."
  else
    echo "FATAL: SAVE_FREQ=${SAVE_FREQ}: a run with no checkpoint cannot be certified complete" \
         "(verl can exit 0 after a crash). Set a positive SAVE_FREQ, or ALLOW_UNCERTIFIED=1 for a" \
         "throwaway smoke." >&2
    exit 1
  fi
fi
if [ "${TEST_FREQ:--1}" = "0" ]; then
  echo "FATAL: TEST_FREQ=0 makes verl's trainer divide by zero at the end of fit(), which skips" \
       "the final checkpoint. Use -1 (off) or a positive interval." >&2
  exit 1
fi
# verl caps the run at min(total_rollout_steps, len(dataloader) * total_epochs), counted AFTER
# dropping over-long prompts. If that cap could bind, the run would end "early" by design and
# never be certified -- require 10% headroom.
TRAIN_ROWS=""
if [[ "${TRAIN_FILES}" != \[* ]]; then
  TRAIN_ROWS="$(python3 -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' \
                "${TRAIN_FILES}" 2>/dev/null || true)"
fi
if [[ "${TRAIN_ROWS}" =~ ^[0-9]+$ ]]; then
  if [ $(( TOTAL_ROLLOUT_STEPS * 10 )) -gt $(( TRAIN_ROWS * TOTAL_EPOCHS * 9 )) ]; then
    echo "FATAL: total_rollout_steps(${TOTAL_ROLLOUT_STEPS}) needs more than 90% of the" \
         "${TRAIN_ROWS} rows x ${TOTAL_EPOCHS} epochs of ${TRAIN_FILES}; after verl's over-long" \
         "prompt filter the data could run out first. Raise total_epochs or lower the budget." >&2
    exit 1
  fi
elif [ "${DRY_RUN:-0}" = "1" ] || [[ "${TRAIN_FILES}" == \[* ]]; then
  echo "[info] cannot count the rows of ${TRAIN_FILES} here -> dataset-size headroom not checked."
else
  echo "FATAL: cannot read train_files ${TRAIN_FILES} to size the run." >&2
  exit 1
fi

# The resolved geometry and budget against the model's own limits (heads, layers, experts), the
# Megatron grid and the batch split -- engine/lib/preflight.py; prints the run's plan.
python3 "${HERE}/../lib/preflight.py" plan --mode async \
    MODEL="${MODEL_PATH}" NUM_NODES="${NUM_NODES:-${NNODES}}" NODES="${NNODES}" GPUS_PER_NODE="${NGPUS_PER_NODE}" \
    TRAINER_NODES="${TRAINER_NNODES}" TRAINER_GPUS="${TRAINING_GPUS}" \
    ROLLOUT_GPUS="$(( ROLLOUT_NNODES * ROLLOUT_N_GPUS ))" \
    TP="${TP}" PP="${PP}" CP="${CP}" EP="${EP}" ETP="${ETP}" GEN_TP="${GEN_TP}" \
    PPO_MINI="${PPO_MINI_BATCH_SIZE}" ROLLOUT_N="${ROLLOUT_N}" MULTI_TURN="${MULTI_TURN}" MAX_TURNS="${MAX_TURNS}" \
    TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS}" TRIGGER_SYNC_STEP="${TRIGGER_SYNC_STEP}" \
    REQUIRE_BATCHES="${REQUIRE_BATCHES}" SAVE_FREQ="${SAVE_FREQ}" CKPT_DIR="${CKPT_DIR}" RUN_ID="${RUN_ID:-}" \
    || exit 1
TRAIN_DP=$(( TRAINING_GPUS / (TP * PP * CP) ))   # a whole number: the plan checked it

mkdir -p logs
RUN_TAG="$(date +%Y%m%d-%H%M%S)"

cat <<EOF
============== verl-on-air (FULLY ASYNC) ==============
model             : ${MODEL_PATH}
cluster           : ${NNODES} node(s) x ${NGPUS_PER_NODE} GPU   (split=${SPLIT_MODE})
trainer           : ${TRAINER_NNODES}n x ${TRAINER_N_GPUS}gpu = ${TRAINING_GPUS}  (TP=${TP} PP=${PP} EP=${EP} -> DP=${TRAIN_DP})
rollout           : ${ROLLOUT_NNODES}n x ${ROLLOUT_N_GPUS}gpu  (GEN_TP=${GEN_TP})
async             : trigger_sync=${TRIGGER_SYNC_STEP} require_batches=${REQUIRE_BATCHES} staleness=${STALENESS} partial=${PARTIAL_ROLLOUT}
batch             : ppo_mini=${PPO_MINI_BATCH_SIZE} n=${ROLLOUT_N}  -> ${SYNC_SAMPLES} samples/sync
rollout budget    : total_rollout_steps=${TOTAL_ROLLOUT_STEPS} prompt groups -> exactly ${EXPECTED_FINAL} weight syncs
checkpoints       : save_freq=${SAVE_FREQ} -> success requires a verified ${CKPT_DIR}/global_step_${EXPECTED_FINAL}
seq               : prompt<=${MAX_PROMPT_LEN} response<=${MAX_RESPONSE_LEN} (episode<=${EPISODE_LEN}, resp_budget=${RESP_BUDGET})
agentic           : multi_turn=${MULTI_TURN} max_turns=${MAX_TURNS} tool=${FUNCTION_TOOL_PATH:-none} format=${TOOL_FORMAT}
reward            : manager=${REWARD_MANAGER:-default} fn=${CUSTOM_REWARD_PATH:-builtin} src=${REWARD_SOURCE:-n/a}
node rank         : ${NODE_RANK} (head=${HEAD_ADDR})
======================================================
EOF

# =============================================================================
# Hydra config resolution.
# fully_async_main's @hydra.main(config_path="config") is module-relative, so
# --config-path points at the INSTALLED package's config dir; and the megatron
# config declares `hydra.searchpath: file://verl/trainer/config` which is
# CWD-relative, so we cd into site-packages for it to resolve. verl is
# pip-installed (no repo checkout), so neither can be left to the air CWD.
# =============================================================================
# DRY_RUN is meant to work OFF-NODE (a laptop, CI) where verl is not installed, so a
# failed import must not abort the config print under `set -e`. On a real run it stays a
# hard failure -- with a legible message instead of a bare traceback.
if ! VERL_SITE="$(python3 -c 'import os, verl; print(os.path.dirname(os.path.dirname(verl.__file__)))' 2>/dev/null)"; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        VERL_SITE="<verl-site-packages>"
        echo "[info] DRY_RUN: verl is not importable here -> placeholder site-packages path."
    else
        echo "FATAL: cannot import verl. This launcher must run inside the training image" \
             "(/opt/venv on PATH). Use DRY_RUN=1 to print the config off-node." >&2
        exit 1
    fi
fi
CONFIG_PATH="${VERL_SITE}/verl/experimental/fully_async_policy/config"
echo "[info] verl site-packages: ${VERL_SITE}"
echo "[info] fully-async config: ${CONFIG_PATH}/fully_async_ppo_megatron_trainer.yaml"

# =============================================================================
# Config assembly (arrays so DRY_RUN can print them).
# =============================================================================
DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=0                 # streaming: not effective in fully-async
    data.gen_batch_size=1                   # streaming sample production
    data.return_raw_chat=True               # required for vLLM server/AgentLoop mode
    data.max_prompt_length="${MAX_PROMPT_LEN}"
    data.max_response_length="${MAX_RESPONSE_LEN}"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False   # Qwen3.5 (per verl's 35B async recipe)
    actor_rollout_ref.model.use_remove_padding=False  # Gated-DeltaNet: no THD packing
    actor_rollout_ref.hybrid_engine=False             # disaggregated: rollout is standalone
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.lr_decay_steps="${LR_DECAY_STEPS}"   # fully-async: explicit LR horizon (streaming has no dataloader-derived step count)
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False     # BSHD (Qwen3.5)
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True   # classic path (tested by verl's Qwen3.5 recipes)
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True
    # CPU-offloaded optimizer (classic ZeRO-1 + offload), as the fully-async recipe does.
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${OFFLOAD_FRACTION}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    # Recompute (memory over compute) + MoE fusions are harmless for dense EP=1.
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

REF=(
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.ref.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.entropy_from_logits_with_chunking=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async                    # server / AgentLoop mode (required)
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.8}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP}"
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.calculate_log_probs=True      # required by fully-async
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    # Cap vLLM context (else it sizes KV for the model's 262144 config max, ~3 GiB
    # KV/request). We use prompt(<=1024)+response(2048); 8192 covers that + VL image
    # margin. Matters on memory-constrained rollout; harmless on the dedicated pool.
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-8192}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN:-8192}"
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_PREFIX_CACHING}"
    # enforce_eager=True skips vLLM CUDA-graph capture. At intra-node GEN_TP<=8 the
    # graph-capture path invokes the custom all-reduce kernel, which crashes on this
    # H100 topology ("custom_all_reduce.cuh:455 'invalid argument'") and kills every
    # rollout worker at init (observed). Cross-node GEN_TP=16 uses NCCL and
    # is unaffected. Default off; set ROLLOUT_ENFORCE_EAGER=True for co-located / GEN_TP<=8.
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-False}"
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl  # trainer->rollout weight sync
)

# H100 custom-all-reduce graph-capture crash (custom_all_reduce.cuh:455 'invalid
# argument') -- the GRAPHS-PRESERVING alternative to ROLLOUT_ENFORCE_EAGER. Verified
# on a dedicated single-node probe (infra/diagnostics/air/test_rollout_allreduce.yaml):
# the trigger is PYTORCH_CUDA_ALLOC_CONF=expandable_
# segments:True (VMM allocations can't be shared via the legacy cudaIpcGetMemHandle the
# custom kernel uses at capture; vllm#42609/#43923/#40812), and --disable-custom-all-
# reduce (NCCL fallback -- what a patched/newer vLLM auto-does) FIXES it while KEEPING
# CUDA graphs. This is ROLLOUT-SCOPED via engine_kwargs.vllm (verl plumbs it into
# AsyncEngineArgs, vllm_async_server.py:250,312) -> the Megatron trainer keeps its
# expandable_segments (unlike dropping the process-global alloc-conf). Needed at
# intra-node GEN_TP<=8 with larger capture shapes (a multi-turn run at max_model_len
# 8192 hit it; ~3072 did not). '+' because engine_kwargs.vllm is an empty {} in the
# schema. Default off (keeps the fast custom kernel); the agentic use cases set it True.
if [ "${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-False}" = "True" ]; then
    ROLLOUT+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True)
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","mlflow"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes="${TRAINER_NNODES}"
    trainer.n_gpus_per_node="${TRAINER_N_GPUS}"
    trainer.default_local_dir="${CKPT_DIR}"
    "${IDENTITY_ARGS[@]}"                   # resume_mode (+ max_actor_ckpt_to_keep)
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ:--1}"
    trainer.test_freq="${TEST_FREQ:--1}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
)

# TOP-LEVEL rollout.* — this is the Rollouter resource pool (disjoint GPUs).
ROLLOUTER=(
    rollout.nnodes="${ROLLOUT_NNODES}"
    rollout.n_gpus_per_node="${ROLLOUT_N_GPUS}"
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}"
)

ASYNC=(
    async_training.staleness_threshold="${STALENESS}"
    async_training.trigger_parameter_sync_step="${TRIGGER_SYNC_STEP}"
    async_training.require_batches="${REQUIRE_BATCHES}"
    async_training.partial_rollout="${PARTIAL_ROLLOUT}"
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# GRPO std-normalisation: divide each group's advantages by the group's reward std (verl's
# default). It keeps a graded reward's order and relative gaps; it changes how groups weigh
# against each other. Exactly True or False -- a misspelling must not silently mean "default".
case "${NORM_ADV_BY_STD_IN_GRPO:-}" in
    "") ;;
    True|False) ALGORITHM+=(algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}") ;;
    *) echo "FATAL: NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO} -- use True or False." >&2; exit 1 ;;
esac

# --- dist-checkpointing (opt-in) --------------------------------------------
# By default verl saves the `model` content as a FULL-GATHER HF export via
# mbridge, which OOMs on much larger models (_save_model_as_hf_via_bridge
# gathers all weights onto one GPU). use_dist_checkpointing=True switches the
# save to a SHARDED Megatron dist checkpoint (no gather) -- but the same flag
# ALSO switches INIT to load weights from dist_checkpointing_path instead of HF,
# so it needs a checkpoint pre-converted from HF first. Set for BOTH actor and ref
# (ref also loads its weights at init). Opt-in: default OFF preserves the HF path,
# which is what the eval jobs serve. Not needed at 35B; a seam for bigger models.
if [ "${USE_DIST_CKPT:-False}" = "True" ]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH:?set DIST_CKPT_PATH when USE_DIST_CKPT=True}"
    )
    REF+=(
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    )
fi

# --- multi-turn tool-calling overrides (appended LAST so they win) ----------
# These re-set the response-length budget for the whole trajectory and turn on
# ToolAgentLoop. Placed after DATA/ROLLOUT/ACTOR in the arg list so Hydra's
# last-wins override replaces the single-turn defaults set there.
MULTITURN=()
if [ "${MULTI_TURN}" = "True" ]; then
    MULTITURN=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}"
        actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}"
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LEN}"
        actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}"
        actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
        # Whole-episode budget (verl needs these sized for the full trajectory).
        actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}"
        actor_rollout_ref.rollout.response_length="${RESP_BUDGET}"
        # NB: do NOT emit rollout.single_turn_response_length here. That field does
        # NOT exist in verl v0.9.0's rollout schema (the v6 image is VERL_REF=v0.9.0),
        # and a PLAIN Hydra override of an absent struct key aborts the whole run at
        # config parse ("Key 'single_turn_response_length' is not in struct") -- this
        # killed a run in its first second. v0.9.0's ToolAgentLoop
        # never reads it (0 refs in agent_loop.py/tool_agent_loop.py); each turn is
        # bounded by the remaining response_length and the whole episode by
        # rollout.max_model_len. A NEWER verl added the field -- if the image is ever
        # bumped, re-add it with '+' only after confirming it is in that rollout config.
        data.max_response_length="${RESP_BUDGET}"
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${EPISODE_LEN}"
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${EPISODE_LEN}"
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${EPISODE_LEN}"
    )
    [ -n "${FUNCTION_TOOL_PATH}" ] && MULTITURN+=(actor_rollout_ref.rollout.multi_turn.function_tool_path="${FUNCTION_TOOL_PATH}")
    [ -n "${TOOL_CONFIG_PATH}" ] && MULTITURN+=(actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}")
    # Custom agent loop registry: name -> _target_ (verl agent_loop.py:548); the data's agent_name
    # column routes samples to it. Defaults to engine/train/agent_loops.yaml, which registers
    # `tool_agent` = RoleSpanToolAgentLoop (the reward's record of what the MODEL wrote).
    MULTITURN+=(actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}")
    # The registry's _target_ is imported BY MODULE NAME inside Ray's agent-loop workers, and Ray
    # actors do not reliably inherit this shell's exports: hand them PYTHONPATH (it holds
    # engine/train) explicitly through verl's ray_kwargs.ray_init.runtime_env.
    MULTITURN+=("+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'")
fi

# --- reward-manager overrides (opt-in; e.g. LLM-judge via rate_limited) ------
REWARD=()
[ -n "${REWARD_MANAGER}" ] && REWARD+=(reward.reward_manager.name="${REWARD_MANAGER}")
if [ -n "${CUSTOM_REWARD_PATH}" ]; then
    REWARD+=(
        reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}"
        reward.custom_reward_function.name="${CUSTOM_REWARD_NAME}"
    )
fi
# reward.max_* are not in the reward schema -> add with '+'.
[ -n "${REWARD_MAX_CONCURRENT}" ] && REWARD+=(+reward.max_concurrent="${REWARD_MAX_CONCURRENT}")
[ -n "${REWARD_MAX_RPM}" ] && REWARD+=(+reward.max_rpm="${REWARD_MAX_RPM}")
[ -n "${REWARD_MAX_TPM}" ] && REWARD+=(+reward.max_tpm="${REWARD_MAX_TPM}")
[ -n "${REWARD_TIMEOUT}" ] && REWARD+=(+reward.timeout="${REWARD_TIMEOUT}")

# =============================================================================
# DRY_RUN — print the resolved invocation and exit (before any Ray bootstrap).
# =============================================================================
if [ "${MULTI_TURN}" = "True" ] && [ -n "${FUNCTION_TOOL_PATH}" ] && [ ! -f "${FUNCTION_TOOL_PATH}" ]; then
    echo "FATAL: FUNCTION_TOOL_PATH does not exist: ${FUNCTION_TOOL_PATH}" >&2
    exit 1
fi
if [ -n "${CUSTOM_REWARD_PATH}" ] && [ ! -f "${CUSTOM_REWARD_PATH}" ]; then
    echo "FATAL: CUSTOM_REWARD_PATH does not exist: ${CUSTOM_REWARD_PATH}" >&2
    exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    set +x
    echo "---- resolved fully-async invocation ----"
    printf 'cd %s && python3 -m verl.experimental.fully_async_policy.fully_async_main \\\n' "${VERL_SITE}"
    printf '    --config-path=%s --config-name=fully_async_ppo_megatron_trainer \\\n' "${CONFIG_PATH}"
    for arg in "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" \
               "${ROLLOUT[@]}" "${TRAINER[@]}" "${ROLLOUTER[@]}" "${ASYNC[@]}" \
               ${MULTITURN[@]+"${MULTITURN[@]}"} ${REWARD[@]+"${REWARD[@]}"}; do
        printf '    %s \\\n' "${arg}"
    done
    echo "    # $(( ${#ALGORITHM[@]} + ${#DATA[@]} + ${#MODEL[@]} + ${#ACTOR[@]} + ${#REF[@]} \
            + ${#ROLLOUT[@]} + ${#TRAINER[@]} + ${#ROLLOUTER[@]} + ${#ASYNC[@]} \
            + ${#MULTITURN[@]} + ${#REWARD[@]} )) overrides"
    exit 0
fi

# =============================================================================
# Ray cluster (multi-node only). Single-node lets fully_async_main ray.init().
# =============================================================================
if [ "${NNODES}" -gt 1 ] && [ "${NODE_RANK}" != "0" ]; then
    ray_worker_wait_and_exit "${NNODES}" "${NGPUS_PER_NODE}" "${HEAD_ADDR}" "${NODE_RANK}"
    # never returns
fi
if [ "${NNODES}" -gt 1 ]; then
    ray_install_cleanup_trap          # first: a bootstrap that fails must still stop Ray
    ray_start_head "${NNODES}" "${NGPUS_PER_NODE}" "${HEAD_ADDR}"
fi

# =============================================================================
# Launch. cd into site-packages so the config searchpath resolves.
#
# SUCCESS IS NOT THE EXIT CODE. verl's fully-async main runs the Trainer and the
# Rollouter as concurrent components, and in v0.9.0 its exit status is wrong in
# BOTH directions:
#   * a finished run exits NON-ZERO: the component that finishes cancels the other
#     ("RuntimeError: cancelled" -> RayTaskError), and in multi-node air then marks
#     the job FAILED although training completed and the checkpoint was saved;
#   * a crashed run can exit 0: the Rollouter gathers its tasks with
#     return_exceptions=True and then sends the ordinary stop signal, so a
#     rollout/reward failure ends as a normal stop -- the Trainer force-saves the
#     version it reached and both components "complete successfully". The
#     "[ASYNC MAIN] Training completed or interrupted" line is printed from a
#     `finally:` block, i.e. after failures too.
# So after the process exits -- whatever its code -- engine/lib/run_certificate.py
# decides: the run succeeded only if THIS run wrote verl's checkpoint tracker at
# exactly the planned final version (EXPECTED_FINAL), that checkpoint verifies
# (ckpt_contents.json + a complete HF export), and nobody raised the abort channel.
# The verdict is written to run_result.json next to the checkpoints.
# =============================================================================
LOG="logs/${EXPERIMENT_NAME}-${RUN_TAG}.log"
LOG_ABS="${HERE}/../../${LOG}"
# The tee below runs AFTER `cd "${VERL_SITE}"`, so the earlier CWD-relative
# `mkdir -p logs` (run before we knew the final CWD) can miss this absolute path ->
# tee dies "No such file or directory" AND the post-mortem log scan then reads an
# empty/absent file (observed: a real Hydra parse error never reached the tee'd log).
# mkdir the ABSOLUTE dir.
mkdir -p "$(dirname "${LOG_ABS}")"
CERTIFY="${HERE}/../lib/run_certificate.py"
# Tracker state BEFORE the run: a tracker left by a previous run in the same output
# dir must not certify this one.
PRE_TRACKER="$(python3 "${CERTIFY}" snapshot "${CKPT_DIR}")"
# The abort channel (engine/lib/run_control.py): any component -- e.g. a reward worker
# whose judge failure budget is exhausted -- can request a stop by writing this file.
ABORT_FILE="$(python3 "${HERE}/../lib/run_control.py" path || true)"

# What is about to run, next to the checkpoints (run_result.json lands there at the end).
python3 "${HERE}/../lib/run_manifest.py" "${CKPT_DIR}/run_manifest.json" \
    launcher=run_grpo_fully_async.sh expected_final_version="${EXPECTED_FINAL}" -- \
    "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
    "${TRAINER[@]}" "${ROLLOUTER[@]}" "${ASYNC[@]}" ${MULTITURN[@]+"${MULTITURN[@]}"} \
    ${REWARD[@]+"${REWARD[@]}"} "$@"

cd "${VERL_SITE}"
set +e
# Its own process group, watched for the abort channel, stopped on TERM/INT/HUP -- and
# WAITED for, so a signal is handled at once (engine/lib/run_driver.sh).
run_driver "${LOG_ABS}" \
    python3 -m verl.experimental.fully_async_policy.fully_async_main \
        --config-path="${CONFIG_PATH}" \
        --config-name=fully_async_ppo_megatron_trainer \
        "${ALGORITHM[@]}" \
        "${DATA[@]}" \
        "${MODEL[@]}" \
        "${ACTOR[@]}" \
        "${REF[@]}" \
        "${ROLLOUT[@]}" \
        "${TRAINER[@]}" \
        "${ROLLOUTER[@]}" \
        "${ASYNC[@]}" \
        ${MULTITURN[@]+"${MULTITURN[@]}"} \
        ${REWARD[@]+"${REWARD[@]}"} \
        "$@"
RC="${DRIVER_RC}"

if [[ "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
    python3 "${CERTIFY}" check --ckpt-dir "${CKPT_DIR}" --expected-final "${EXPECTED_FINAL}" \
        --pre "${PRE_TRACKER}" --raw-rc "${RC}" --log "${LOG_ABS}" \
        ${ABORT_FILE:+--abort-file "${ABORT_FILE}"} --settle-s "${CERT_SETTLE_S:-120}" \
        --json-out "${CKPT_DIR}/run_result.json" --json-out "${LOG_ABS%.log}.result.json"
    RC=$?
else
    echo "[head] UNCERTIFIED run (SAVE_FREQ=${SAVE_FREQ}, ALLOW_UNCERTIFIED=1): exit ${RC} is" \
         "reported as-is and says nothing reliable about completion."
fi
set -e
# Triggers the EXIT trap (multi-node) which re-exits with this code; direct on single node.
exit "${RC}"
