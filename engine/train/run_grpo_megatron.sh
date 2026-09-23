#!/usr/bin/env bash
# =============================================================================
# verl GRPO with the Megatron/mcore backend on Databricks AI Runtime.
#
# Two backend modes, one script:
#
#   MEGATRON_MODE=fsdp     (default) Megatron-FSDP == ZeRO-3. Shards params +
#                          grads + optimizer across the DP dimension, via
#                          Megatron-Bridge (vanilla_mbridge=False is REQUIRED:
#                          verl only threads use_megatron_fsdp through the
#                          Megatron-Bridge "provider" code path).
#                          -> This is what makes offload-free 35B-A3B fit on
#                             16xH100. See docs/sizing.md.
#
#   MEGATRON_MODE=classic  Distributed optimizer == ZeRO-1 (params and grads
#                          are REPLICATED across DP) + heavy CPU offload. This
#                          is the config verl actually tested for
#                          Qwen3.5-35B-A3B on a single 8xH100 node, so it is
#                          the known-good fallback.
#
# Derived from the upstream examples, which are the reference for every flag:
#   examples/grpo_trainer/run_qwen3_5_35b_megatron.sh   (Qwen3.5 MoE, classic)
#   examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh  (Megatron-FSDP)
#
# All knobs are env vars; air passes dataset/model/output via `parameters:`.
# =============================================================================
set -xeuo pipefail

# PATH guard: air's `command:` can run with a minimal PATH that omits the venv
# bin. Observed on a MULTI-NODE job whose YAML used a plain-scalar `command:`
# (vs the `command: |` block form): `ray: command not found` -> exit 127 in the
# Ray head start, because ray/verl/torch live in /opt/venv/bin. Prepend it when
# the venv tools are not already resolvable; no-op when air provides the full
# PATH. Also keeps `python3` pointed at the venv interpreter that has verl.
command -v ray >/dev/null 2>&1 || export PATH="/opt/venv/bin:${PATH}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/hparams.sh
source "${HERE}/../lib/hparams.sh"
# shellcheck source=../lib/ray_cluster.sh
source "${HERE}/../lib/ray_cluster.sh"
# shellcheck source=../lib/run_identity.sh
source "${HERE}/../lib/run_identity.sh"

# --- FIPS ------------------------------------------------------------------
# air hosts run a FIPS kernel; non-FIPS crypto in the image aborts on SSL init.
# Set for the driver; Ray workers inherit the process environment.
# NEVER set RAY_RUNTIME_ENV_HOOK="" — Ray tries to import the empty string as a
# class path and dies with "expected a valid path like mymodule.provider_class".
export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0

hp_dump

# Every engine knob this job sets, typed and checked against what THIS launcher reads -- a knob only
# the other mode reads, or a misspelt one, stops the job here instead of being ignored
# (engine/lib/preflight.py). Booleans come back as exactly True/False.
KNOB_EXPORTS="$(python3 "${HERE}/../lib/preflight.py" knobs --mode sync)" || exit 1
eval "${KNOB_EXPORTS}"

# =============================================================================
# Mode + topology
# =============================================================================
MEGATRON_MODE="${MEGATRON_MODE:-fsdp}"      # fsdp | classic
OFFLOAD="${OFFLOAD:-auto}"                  # auto | 0 | 1

NNODES="${NNODES:-${NUM_NODES:-1}}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}"
NODE_RANK="${NODE_RANK:-${POD_RANK:-0}}"
HEAD_ADDR="${MASTER_ADDR:-127.0.0.1}"
WORLD_GPUS=$((NNODES * NGPUS_PER_NODE))

# --- trainer mode + disaggregated placement --------------------------------
# TRAINER_MODE selects verl's v1 trainer (trainer.v1.trainer_mode):
#   sync            (default) rollout+train co-located on ALL GPUs — rungs 1-4.
#   separate_async  standalone vLLM rollout on its OWN nodes; the trainer consumes
#                   bounded-stale trajectories (fully async / disaggregated).
# The Ray cluster always spans ALL air nodes (NNODES); in separate_async the
# trainer pool is the non-rollout share and verl places the standalone rollout on
# the remaining ROLLOUT_NNODES nodes. For sync, ROLLOUT_NNODES=0 so TRAINER_*
# collapse to the whole cluster and nothing downstream changes.
TRAINER_MODE="${TRAINER_MODE:-sync}"           # sync | separate_async
ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"          # standalone rollout nodes (separate_async)
TRAINER_NNODES=$(( NNODES - ROLLOUT_NNODES ))
TRAINER_GPUS=$(( TRAINER_NNODES * NGPUS_PER_NODE ))
if [ "${TRAINER_MODE}" = "separate_async" ] && { [ "${ROLLOUT_NNODES}" -lt 1 ] || [ "${TRAINER_NNODES}" -lt 1 ]; }; then
  echo "FATAL: separate_async needs ROLLOUT_NNODES>=1 and >=1 trainer node" \
       "(NNODES=${NNODES} ROLLOUT_NNODES=${ROLLOUT_NNODES} -> TRAINER_NNODES=${TRAINER_NNODES})." >&2
  exit 1
fi

# `auto`: offload only when the optimizer cannot be sharded thin enough to fit.
# classic replicates params/grads across DP, so it needs offload at <=16 GPUs;
# fsdp shards everything and only needs offload at <=8 -- but Megatron-FSDP crashes
# WITH offload (aten.is_pinned on DTensor), so fsdp below 16 GPUs has no automatic
# answer: pick classic+offload, or say OFFLOAD=0 for a model that fits (rungs 1-2).
# docs/sizing.md has the per-GPU byte budget these thresholds come from.
if [ "${OFFLOAD}" = "auto" ]; then
  if [ "${MEGATRON_MODE}" = "fsdp" ]; then
    if [ "${TRAINER_GPUS}" -ge 16 ]; then
      OFFLOAD=0
    else
      echo "FATAL: MEGATRON_MODE=fsdp on ${TRAINER_GPUS} GPUs would need CPU offload, which crashes" \
           "Megatron-FSDP. Set OFFLOAD=0 if the model fits without it, or MEGATRON_MODE=classic." >&2
      exit 1
    fi
  else
    [ "${TRAINER_GPUS}" -ge 32 ] && OFFLOAD=0 || OFFLOAD=1
  fi
fi

# --- Parallelism -----------------------------------------------------------
# Qwen3.5-35B-A3B: 40 layers, 256 experts (top-8), hidden 2048, 16Q/2KV heads.
# 92.5% of the weights are routed experts -> EP is the dominant lever.
# EP=8 gives 32 experts/rank; TP must divide 16 heads.
TP="${TP:-$([ "${MEGATRON_MODE}" = "fsdp" ] && echo 1 || echo 2)}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"
# vLLM rollout TP. Keep <= GPUs-per-node so rollout tensor parallel stays
# intra-node (NVLink); extra nodes become rollout DP replicas instead.
GEN_TP="${GEN_TP:-8}"

# --- agentic / multi-turn tool-calling (opt-in; default OFF => single-turn) ---
# Mirrors run_grpo_fully_async.sh's block (same tool, reward and agent-loop
# overrides). When MULTI_TURN=True the CO-LOCATED rollout runs verl's ToolAgentLoop
# in vLLM server mode (rollout.mode=async is required for the agent loop even though
# the TRAINER is synchronous — "async" there names the vLLM engine mode, not the
# training mode). Switching an agentic job to sync is NOT one knob: it needs its own
# node count, backend/offload, and an explicit optimizer-step budget -- see
# usecases/agentic-search/air/4_train_sync.yaml.
#
# SUPPORT STATUS (be precise; see docs/training-modes.md): the measured runs in this
# repo trained the agentic use cases on the FULLY-ASYNC launcher. Agentic sync is
# config-validated only (scripts/compose_check.py composes it against the pinned
# verl); it has not run on GPUs. A co-located judge on sync is refused by the
# dispatcher for the same reason.
MULTI_TURN="${MULTI_TURN:-False}"
MAX_TURNS="${MAX_TURNS:-4}"
FUNCTION_TOOL_PATH="${FUNCTION_TOOL_PATH:-}"          # python file of @function_tool defs
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-}"              # yaml of stateful BaseTool defs (optional)
TOOL_FORMAT="${TOOL_FORMAT:-hermes}"                  # tool-call parser (Qwen3.5 -> qwen3_coder)
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"           # parallel AgentLoopWorker actors
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-512}" # per tool-response token cap
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-${HERE}/agent_loops.yaml}"  # agent-loop registry (multi-turn): default registers the role-span ToolAgentLoop
# Ray workers import engine/train modules by name (the agent-loop registry): keep it on PYTHONPATH.
case ":${PYTHONPATH:-}:" in *":${HERE}:"*) ;; *) PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" ;; esac
export PYTHONPATH

# --- CUDA_DEVICE_MAX_CONNECTIONS ------------------------------------------
# classic Megatron wants =1 for comm/compute overlap. Megatron-FSDP requires it
# UNSET (or >1) — with =1 the FSDP all-gather/reduce-scatter streams serialise
# behind compute and you silently lose most of the overlap. This is the single
# easiest thing to get wrong when copy-pasting the upstream 35B script.
if [ "${MEGATRON_MODE}" = "fsdp" ]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS || true
else
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# =============================================================================
# Parameters from air
# =============================================================================
MODEL_PATH="$(hp model_name "Qwen/Qwen3.5-35B-A3B")"
TRAIN_FILES="$(hp train_files "/Volumes/main/mshtelma/verl/data/geo3k/train.parquet")"
VAL_FILES="$(hp val_files "/Volumes/main/mshtelma/verl/data/geo3k/test.parquet")"
CKPT_DIR="$(hp output_dir "/Volumes/main/mshtelma/verl/ckpt/default")"
TOTAL_EPOCHS="$(hp total_epochs 1)"
TOTAL_TRAIN_STEPS="$(hp total_training_steps 3)"   # 0 disables the cap
TRAIN_BATCH_SIZE="$(hp train_batch_size 32)"
PPO_MINI_BATCH_SIZE="$(hp ppo_mini_batch_size 32)"
ROLLOUT_N="$(hp rollout_n 5)"
MAX_PROMPT_LEN="$(hp max_prompt_length 1024)"
MAX_RESPONSE_LEN="$(hp max_response_length 2048)"
ACTOR_LR="$(hp actor_lr 1e-6)"
IMAGE_KEY="$(hp image_key images)"                 # "" for text-only datasets

# Run identity: this run writes to <output_dir>/<RUN_ID>/, and resuming is explicit (RESUME).
resolve_run_identity "${CKPT_DIR}" || exit 1   # CKPT_DIR becomes <output_dir>/<RUN_ID>

PROJECT_NAME="${PROJECT_NAME:-$(hp project_name verl-on-air)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(hp experiment_name grpo-megatron)}"

# The resolved geometry and budget against the model's own limits (heads, layers, experts), the
# Megatron grid, FSDP-vs-offload and the batch split -- verl would otherwise reject a bad batch
# only after the cluster spun up (engine/lib/preflight.py). Prints the run's plan.
if [ "${TRAINER_MODE}" = "separate_async" ]; then ROLLOUT_GPUS=$(( ROLLOUT_NNODES * NGPUS_PER_NODE )); else ROLLOUT_GPUS=${WORLD_GPUS}; fi
python3 "${HERE}/../lib/preflight.py" plan --mode sync \
    MODEL="${MODEL_PATH}" NUM_NODES="${NUM_NODES:-${NNODES}}" NODES="${NNODES}" GPUS_PER_NODE="${NGPUS_PER_NODE}" \
    TRAINER_NODES="${TRAINER_NNODES}" TRAINER_GPUS="${TRAINER_GPUS}" ROLLOUT_GPUS="${ROLLOUT_GPUS}" \
    TP="${TP}" PP="${PP}" CP="${CP}" EP="${EP}" ETP="${ETP}" GEN_TP="${GEN_TP}" \
    MEGATRON_MODE="${MEGATRON_MODE}" OFFLOAD="${OFFLOAD}" \
    PPO_MINI="${PPO_MINI_BATCH_SIZE}" ROLLOUT_N="${ROLLOUT_N}" MULTI_TURN="${MULTI_TURN}" MAX_TURNS="${MAX_TURNS}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" TOTAL_TRAINING_STEPS="${TOTAL_TRAIN_STEPS}" \
    SAVE_FREQ="${SAVE_FREQ:--1}" CKPT_DIR="${CKPT_DIR}" RUN_ID="${RUN_ID:-}" \
    || exit 1

# Episode-length arithmetic (same rule as the fully-async launcher). Single-turn:
# the response budget IS max_response_length. Multi-turn: the actor trains on the
# WHOLE episode (every assistant turn + every tool response), so the trajectory can
# reach (prompt+response)*turns tokens and verl's response_length is that minus the
# initial prompt. vLLM's context must hold the whole thing.
if [ "${MULTI_TURN}" = "True" ]; then
  EPISODE_LEN=$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * MAX_TURNS ))
  RESP_BUDGET=$(( EPISODE_LEN - MAX_PROMPT_LEN ))
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${EPISODE_LEN}}"
else
  EPISODE_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))
  RESP_BUDGET="${MAX_RESPONSE_LEN}"
fi

mkdir -p logs
RUN_TAG="$(date +%Y%m%d-%H%M%S)"

cat <<EOF
================ verl-on-air ================
mode              : ${MEGATRON_MODE}   (offload=${OFFLOAD})   trainer_mode=${TRAINER_MODE}
model             : ${MODEL_PATH}
topology          : ${NNODES} node(s) x ${NGPUS_PER_NODE} GPU = ${WORLD_GPUS}  (trainer=${TRAINER_NNODES}n/${TRAINER_GPUS}gpu, rollout=${ROLLOUT_NNODES}n)
parallelism       : TP=${TP} PP=${PP} CP=${CP} EP=${EP} ETP=${ETP} GEN_TP=${GEN_TP}
batch             : train=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} n=${ROLLOUT_N}
seq               : prompt<=${MAX_PROMPT_LEN} response<=${MAX_RESPONSE_LEN} (episode<=${EPISODE_LEN}, resp_budget=${RESP_BUDGET})
agentic           : multi_turn=${MULTI_TURN} max_turns=${MAX_TURNS} tool=${FUNCTION_TOOL_PATH:-none} format=${TOOL_FORMAT}
node rank         : ${NODE_RANK} (head=${HEAD_ADDR})
=============================================
EOF

# =============================================================================
# Config assembly
# =============================================================================
ALGORITHM=(
    # GRPO: group-relative advantage baseline, no value network -> no critic
    # worker at all, which is a big part of why 35B-A3B is tractable here.
    algorithm.adv_estimator=grpo
    # KL is applied as a LOSS term (below), not folded into the reward.
    algorithm.use_kl_in_reward=False
)
# GRPO std-normalisation, exactly as the async launcher takes it (True/False; unset = verl's default).
case "${NORM_ADV_BY_STD_IN_GRPO:-}" in
    "") ;;
    True|False) ALGORITHM+=(algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}") ;;
    *) echo "FATAL: NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO} -- use True or False." >&2; exit 1 ;;
esac

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LEN}"
    data.max_response_length="${MAX_RESPONSE_LEN}"
    data.filter_overlong_prompts=True
    data.truncation=error
    # The fully-async launcher leaves verl's default (shuffle on); a sync run that means to
    # match an async one must set DATA_SHUFFLE=True. The ladder keeps the deterministic order.
    data.shuffle="${DATA_SHUFFLE:-False}"
)
# geo3k is multimodal; Qwen3.5 has a vision tower. Drop image_key for text-only.
[ -n "${IMAGE_KEY}" ] && DATA+=( data.image_key="${IMAGE_KEY}" )

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    # Qwen3.5 Gated-DeltaNet has no THD (packed sequence) support in Megatron-LM,
    # so the whole pipeline must run in BSHD. This flag and the two dynamic_bsz
    # flags below are not tuning — they are correctness requirements.
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False          # required by BSHD
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl        # k3 estimator
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    # vocab is 248320; un-chunked logits+entropy is ~3 GB per micro-batch.
    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True
    # Recompute everything: activations are cheap to recompute relative to the
    # HBM they free, and this model is memory- not compute-bound here.
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    # MoE: grouped GEMM + fused permute are the difference between usable and
    # unusable throughput with 256 experts.
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.ref.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.ref.megatron.entropy_from_logits_with_chunking=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.calculate_log_probs=True
    # Cap the vLLM context. Unset, vLLM sizes KV for the model's config max
    # (Qwen3.5 = 262144), needing ~3 GiB KV/request -- which fails on a memory-
    # constrained co-located rollout ("KV cache needed > available"). We only use
    # prompt(<=1024)+response(2048); 8192 covers that
    # plus generous VL image-token margin and needs ~0.1 GiB KV/request.
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-8192}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN:-8192}"
    # Free the KV cache between rollout and training so the two peaks do not sum.
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
)
# Actor->vLLM weight sync moves tensors in fixed-size buckets. A tensor larger
# than the bucket aborts with "too large to fit in the bucket"; the embedding
# here is 248320x2048 (~970 MiB bf16), so any bucket must exceed that. Left unset
# because the config path for this moved between verl releases — see
# docs/troubleshooting.md before setting it.
[ -n "${WEIGHT_BUCKET_MB:-}" ] && ROLLOUT+=(
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB}"
)
# enforce_eager disables vLLM CUDA-graph capture, cutting several GiB off the
# awake-vLLM footprint. That footprint is the ['weights'] tag the on_step_end
# Megatron-FSDP->HF weight sync collides with on the 35B fsdp run (the full-tensor
# DTensor gather in uneven_dtensor_to_full_tensor OOMs against vLLM's re-woken
# weights). Trades rollout throughput (eager generation) for co-location headroom.
[ "${ROLLOUT_ENFORCE_EAGER:-False}" = "True" ] && ROLLOUT+=(
    actor_rollout_ref.rollout.enforce_eager=True
)

# --- mode-specific --------------------------------------------------------
if [ "${MEGATRON_MODE}" = "fsdp" ]; then
    ACTOR+=(
        # Megatron-Bridge (NOT legacy mbridge): verl only passes
        # use_megatron_fsdp through the Bridge "provider" path.
        actor_rollout_ref.actor.megatron.vanilla_mbridge=False
        actor_rollout_ref.actor.megatron.use_megatron_fsdp=True
        # ZeRO-3: shard optimizer + grads + params. Redundant-but-explicit --
        # verl already applies this whenever use_megatron_fsdp=True
        # (verl/utils/megatron_utils.py:415, setdefault). Kept so the sizing story
        # in docs/sizing.md cannot be invalidated by a future default change.
        +actor_rollout_ref.actor.megatron.override_ddp_config.data_parallel_sharding_strategy=optim_grads_params
        # Megatron-FSDP is incompatible with gradient accumulation fusion.
        ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    )
    REF+=(
        actor_rollout_ref.ref.megatron.use_mbridge=True
        actor_rollout_ref.ref.megatron.vanilla_mbridge=False
        actor_rollout_ref.ref.megatron.use_megatron_fsdp=True
        ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False
    )
else
    ACTOR+=(
        # Legacy mbridge is what upstream's tested Qwen3.5-35B script uses.
        actor_rollout_ref.actor.megatron.vanilla_mbridge=True
        ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    )
    REF+=( actor_rollout_ref.ref.megatron.use_mbridge=True
           actor_rollout_ref.ref.megatron.vanilla_mbridge=True )
fi

# --- offload --------------------------------------------------------------
# use_precision_aware_optimizer: CLASSIC MODE ONLY.
#
# It halves Adam state (12 -> 8 bytes/param), so enabling it everywhere looked
# like free headroom. It is not: with use_megatron_fsdp=True it SEGFAULTS inside
# Transformer Engine, in Megatron's gradient-clipping path:
#
#   !!!!!!! Segfault encountered !!!!!!!
#     transformer_engine::multi_tensor_scale::multi_tensor_scale_tensor_cuda(...)
#     nvte_multi_tensor_scale_tensor_cuda
#
# all 8 ranks, immediately after the first rollout completed.
#
# Ground truth in verl: precision-aware appears ONLY in the classic 35B script
# (examples/grpo_trainer/run_qwen3_5_35b_megatron.sh), bundled with CPU offload
# (optimizer_cpu_offload + optimizer_offload_fraction + overlap_cpu_optimizer).
# verl's Megatron-FSDP reference (run_qwen2-7b_math_megatron_fsdp.sh) does NOT set
# it, and verl's own default is False (config/optim/megatron.yaml:54).
#
# Cost of not having it, from docs/sizing.py: 16-GPU fsdp goes 39.2 -> 47.9 GB/GPU,
# still comfortable. clip_grad defaults to 1.0, which is what reaches
# multi_tensor_scale, so this is not avoidable by luck.
if [ "${MEGATRON_MODE}" != "fsdp" ]; then
    ACTOR+=( +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True )
fi

if [ "${OFFLOAD}" = "1" ]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${OFFLOAD_FRACTION:-1}"
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    )
    REF+=( actor_rollout_ref.ref.megatron.param_offload=True )
    # ~390 GB of host RAM per node at offload_fraction=1 for this model. If the
    # node has less, the job dies in the optimizer build with an opaque OOM.
    echo "[warn] OFFLOAD=1: expect ~400-500 GB host RAM per node. Host has: \
$(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}' || echo unknown)"
fi

# --- dist-checkpointing (opt-in) --------------------------------------------
# Default verl saves `model` as a FULL-GATHER HF export via mbridge, which is what
# the eval jobs consume -- but it gathers every weight onto one GPU, so it OOMs on
# much larger models (measured at 122B). use_dist_checkpointing=True switches the
# save to a SHARDED Megatron dist checkpoint AND switches INIT to load from
# dist_checkpointing_path, so it needs a checkpoint pre-converted from HF. Neither
# shipped use case needs this at 35B; left as a documented seam. Set for actor+ref.
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

TRAINER=(
    trainer.critic_warmup=0                  # GRPO has no critic to warm up
    trainer.logger='["console","mlflow"]'    # air injects the MLflow context
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${TRAINER_NNODES}"
    trainer.default_local_dir="${CKPT_DIR}"
    "${IDENTITY_ARGS[@]}"                   # resume_mode (+ max_actor_ckpt_to_keep)
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
    trainer.save_freq="${SAVE_FREQ:--1}"
    trainer.test_freq="${TEST_FREQ:--1}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
)
[ "${TOTAL_TRAIN_STEPS}" != "0" ] && TRAINER+=( trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" )

# --- separate_async: standalone (disaggregated) rollout --------------------
# Stand up vLLM on its OWN nodes; the trainer streams fresh weights to it via an
# nccl broadcast (checkpoint_engine) and consumes bounded-stale trajectories.
# Hard asserts in verl/trainer/ppo/v1/trainer_separate_async.py:
#   * train_batch_size == parameter_sync_step * ppo_mini_batch_size  (-> 32==1*32)
#   * rollout.checkpoint_engine.backend != "naive"  (use nccl)
#   * rollout.nnodes > 0 and rollout.n_gpus_per_node > 0
# hybrid_engine=False so the trainer GPUs do NOT also host rollout (fully
# disaggregated); the standalone pool is the only rollout. The rule-based geo3k
# reward has no reward model, so the separate-mode reward-pool assert is skipped.
if [ "${TRAINER_MODE}" = "separate_async" ]; then
    PARAM_SYNC_STEP="${PARAM_SYNC_STEP:-$(( TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE ))}"
    TRAINER+=(
        trainer.v1.trainer_mode=separate_async
        trainer.v1.separate_async.parameter_sync_step="${PARAM_SYNC_STEP}"
        trainer.v1.separate_async.num_warmup_batches="${ASYNC_WARMUP_BATCHES:-1}"
    )
    # bounded staleness: max model-versions a trajectory may span before it is dropped
    [ -n "${MAX_OFF_POLICY:-}" ] && TRAINER+=(
        trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY}" )
    ROLLOUT+=(
        actor_rollout_ref.hybrid_engine=False
        actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}"
        actor_rollout_ref.rollout.n_gpus_per_node="${NGPUS_PER_NODE}"
        actor_rollout_ref.rollout.checkpoint_engine.backend="${CKPT_ENGINE_BACKEND:-nccl}"
    )
    echo "[info] separate_async: trainer=${TRAINER_NNODES}n rollout=${ROLLOUT_NNODES}n" \
         "param_sync_step=${PARAM_SYNC_STEP} ckpt_backend=${CKPT_ENGINE_BACKEND:-nccl}"
fi

# --- reward ---------------------------------------------------------------
# Default: verl's built-in rule-based scorer, dispatched on the parquet's
# `data_source` column. For geo3k that is verl/utils/reward_score/geo3k.py:
#   0.9 * (boxed answer graded correct) + 0.1 * (<think></think> + \boxed{} format)
# No reward model, no critic, nothing to train.
#
# PHASE 2 HOOK: point CUSTOM_REWARD_PATH at infra/geo3k/reward.py
# (or your own) to override, without touching this launcher.
#
# The key is reward.custom_reward_function.* -- what verl's reward loader reads
# (verl/trainer/ppo/reward.py). NOT the legacy top-level custom_reward_function.*:
# only fully_async_main migrates that one (migrate_legacy_reward_impl), so under
# main_ppo it was silently ignored and a use case's reward replaced by the
# data_source default (which raises NotImplementedError for musique/hotpotqa).
# scripts/compose_check.py asserts the resolved path for every training job.
REWARD=()
if [ -n "${CUSTOM_REWARD_PATH:-}" ]; then
    REWARD+=(
        reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}"
        reward.custom_reward_function.name="${CUSTOM_REWARD_NAME:-compute_score}"
    )
    echo "[info] custom reward: ${CUSTOM_REWARD_PATH}::${CUSTOM_REWARD_NAME:-compute_score}"
fi
# Reward manager + its limits, exactly as the fully-async launcher emits them
# (reward.max_* are not in the reward schema -> added with '+').
if [ -n "${REWARD_MANAGER:-}" ]; then REWARD+=(reward.reward_manager.name="${REWARD_MANAGER}"); fi
if [ -n "${REWARD_MAX_CONCURRENT:-}" ]; then REWARD+=(+reward.max_concurrent="${REWARD_MAX_CONCURRENT}"); fi
if [ -n "${REWARD_MAX_RPM:-}" ]; then REWARD+=(+reward.max_rpm="${REWARD_MAX_RPM}"); fi
if [ -n "${REWARD_MAX_TPM:-}" ]; then REWARD+=(+reward.max_tpm="${REWARD_MAX_TPM}"); fi
if [ -n "${REWARD_TIMEOUT:-}" ]; then REWARD+=(+reward.timeout="${REWARD_TIMEOUT}"); fi

# --- multi-turn tool-calling overrides (appended LAST so they win) -----------
# Hydra is last-wins, so these must follow DATA/ROLLOUT/ACTOR/REF to replace the
# single-turn response-length budget set there. Identical in shape to the
# fully-async launcher's MULTITURN block; see the SUPPORT STATUS note at the top.
MULTITURN=()
if [ "${MULTI_TURN}" = "True" ]; then
    echo "[info] MULTI_TURN=True on the SYNC launcher: co-located ToolAgentLoop." \
         "The repo's measured agentic runs used the fully-async launcher" \
         "(docs/training-modes.md); this path is DRY_RUN-validated."
    MULTITURN=(
        # The agent loop needs vLLM in server mode. "async" here is the vLLM ENGINE
        # mode (AgentLoop), NOT the training mode -- the trainer stays synchronous.
        actor_rollout_ref.rollout.mode=async
        data.return_raw_chat=True                      # required for server/AgentLoop mode
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}"
        actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}"
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LEN}"
        actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}"
        actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
        # Whole-episode budget (verl sizes these for the full trajectory, not one turn).
        actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}"
        actor_rollout_ref.rollout.response_length="${RESP_BUDGET}"
        data.max_response_length="${RESP_BUDGET}"
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
        actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}"
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
    # Fail before the cluster spins up, not 10 minutes in.
    if [ -n "${FUNCTION_TOOL_PATH}" ] && [ ! -f "${FUNCTION_TOOL_PATH}" ]; then
        echo "FATAL: FUNCTION_TOOL_PATH does not exist: ${FUNCTION_TOOL_PATH}" >&2
        exit 1
    fi
fi
if [ -n "${CUSTOM_REWARD_PATH:-}" ] && [ ! -f "${CUSTOM_REWARD_PATH}" ]; then
    echo "FATAL: CUSTOM_REWARD_PATH does not exist: ${CUSTOM_REWARD_PATH}" >&2
    exit 1
fi

EXTRA=( model_engine=megatron )   # current route; ppo_megatron_trainer.yaml is deprecated

# =============================================================================
# DRY_RUN — print the fully-resolved invocation and exit.
# MUST come before the Ray bootstrap, or a dry run would start a Ray head.
#   DRY_RUN=1 MEGATRON_MODE=fsdp NUM_NODES=2 LOCAL_WORLD_SIZE=8 \
#     bash engine/train/run_grpo_megatron.sh
# =============================================================================
if [ "${DRY_RUN:-0}" = "1" ]; then
    set +x
    echo "---- resolved verl invocation (${MEGATRON_MODE}, offload=${OFFLOAD}) ----"
    printf 'python3 -m verl.trainer.main_ppo \\\n'
    for arg in "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" \
               "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
               ${REWARD[@]+"${REWARD[@]}"} ${MULTITURN[@]+"${MULTITURN[@]}"} "${EXTRA[@]}"; do
        printf '    %s \\\n' "${arg}"
    done
    n=$(( ${#ALGORITHM[@]} + ${#DATA[@]} + ${#MODEL[@]} + ${#ACTOR[@]} + ${#REF[@]} \
          + ${#ROLLOUT[@]} + ${#TRAINER[@]} + ${#REWARD[@]} + ${#MULTITURN[@]} + ${#EXTRA[@]} ))
    echo "    # ${n} overrides total"
    exit 0
fi

# =============================================================================
# Ray cluster
# =============================================================================
if [ "${NNODES}" -gt 1 ] && [ "${NODE_RANK}" != "0" ]; then
    ray_worker_wait_and_exit "${NNODES}" "${NGPUS_PER_NODE}" "${HEAD_ADDR}" "${NODE_RANK}"
    # never returns
fi

if [ "${NNODES}" -gt 1 ]; then
    ray_start_head "${NNODES}" "${NGPUS_PER_NODE}" "${HEAD_ADDR}"
    ray_install_cleanup_trap
fi

# =============================================================================
# Launch
# =============================================================================
# What is about to run, next to the checkpoints.
python3 "${HERE}/../lib/run_manifest.py" "${CKPT_DIR}/run_manifest.json" launcher=run_grpo_megatron.sh -- \
    "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" \
    "${TRAINER[@]}" ${REWARD[@]+"${REWARD[@]}"} ${MULTITURN[@]+"${MULTITURN[@]}"} "${EXTRA[@]}" "$@"
LOG="logs/${EXPERIMENT_NAME}-${RUN_TAG}.log"
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    ${REWARD[@]+"${REWARD[@]}"} \
    ${MULTITURN[@]+"${MULTITURN[@]}"} \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG}"
