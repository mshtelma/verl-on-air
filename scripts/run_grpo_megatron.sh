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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/hparams.sh
source "${HERE}/lib/hparams.sh"
# shellcheck source=lib/ray_cluster.sh
source "${HERE}/lib/ray_cluster.sh"

# --- FIPS ------------------------------------------------------------------
# air hosts run a FIPS kernel; non-FIPS crypto in the image aborts on SSL init.
# Set for the driver; Ray workers inherit the process environment.
# NEVER set RAY_RUNTIME_ENV_HOOK="" — Ray tries to import the empty string as a
# class path and dies with "expected a valid path like mymodule.provider_class".
export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0

hp_dump

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

# `auto`: offload only when the optimizer cannot be sharded thin enough to fit.
# classic replicates params/grads across DP, so it needs offload at <=16 GPUs;
# fsdp shards everything and only needs offload at <=8. docs/sizing.md has the
# per-GPU byte budget these thresholds come from.
if [ "${OFFLOAD}" = "auto" ]; then
  if [ "${MEGATRON_MODE}" = "fsdp" ]; then
    [ "${WORLD_GPUS}" -ge 16 ] && OFFLOAD=0 || OFFLOAD=1
  else
    [ "${WORLD_GPUS}" -ge 32 ] && OFFLOAD=0 || OFFLOAD=1
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

PROJECT_NAME="${PROJECT_NAME:-$(hp project_name verl-on-air)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(hp experiment_name grpo-megatron)}"

# verl requires (train_batch_size * rollout.n) % world_gpus == 0. Fail here with
# the actual arithmetic rather than let verl raise
# "real_train_batch_size must be divisible by minimal possible batch" after the
# cluster has already spun up.
REAL_BATCH=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
if [ $(( REAL_BATCH % WORLD_GPUS )) -ne 0 ]; then
  {
    echo "FATAL: train_batch_size(${TRAIN_BATCH_SIZE}) * rollout_n(${ROLLOUT_N})" \
         "= ${REAL_BATCH}, which is not divisible by world GPUs (${WORLD_GPUS})."
    echo "       ${REAL_BATCH} % ${WORLD_GPUS} = $(( REAL_BATCH % WORLD_GPUS ))"
    echo "       verl would reject this as 'real_train_batch_size must be" \
         "divisible by minimal possible batch'."
    echo "       Adjust train_batch_size or rollout_n in the YAML parameters."
  } >&2
  exit 1
fi

mkdir -p logs
RUN_TAG="$(date +%Y%m%d-%H%M%S)"

cat <<EOF
================ verl-on-air ================
mode              : ${MEGATRON_MODE}   (offload=${OFFLOAD})
model             : ${MODEL_PATH}
topology          : ${NNODES} node(s) x ${NGPUS_PER_NODE} GPU = ${WORLD_GPUS}
parallelism       : TP=${TP} PP=${PP} CP=${CP} EP=${EP} ETP=${ETP} GEN_TP=${GEN_TP}
batch             : train=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} n=${ROLLOUT_N}
seq               : prompt<=${MAX_PROMPT_LEN} response<=${MAX_RESPONSE_LEN}
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

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LEN}"
    data.max_response_length="${MAX_RESPONSE_LEN}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
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
    # Free the KV cache between rollout and training so the two peaks do not sum.
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
)
# Actor->vLLM weight sync moves tensors in fixed-size buckets. A tensor larger
# than the bucket aborts with "too large to fit in the bucket"; the embedding
# here is 248320x2048. Left unset because the config path for this moved between
# verl releases — see docs/troubleshooting.md before setting it.
[ -n "${WEIGHT_BUCKET_MB:-}" ] && ROLLOUT+=(
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB}"
)

# --- mode-specific --------------------------------------------------------
if [ "${MEGATRON_MODE}" = "fsdp" ]; then
    ACTOR+=(
        # Megatron-Bridge (NOT legacy mbridge): verl only passes
        # use_megatron_fsdp through the Bridge "provider" path.
        actor_rollout_ref.actor.megatron.vanilla_mbridge=False
        actor_rollout_ref.actor.megatron.use_megatron_fsdp=True
        # ZeRO-3: shard optimizer + grads + params.
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
# Precision-aware optimizer is on in BOTH cases: it drops Adam state from
# 12 to 8 bytes/param (~130 GB across the job), which is most of our margin.
ACTOR+=( +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True )

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

TRAINER=(
    trainer.critic_warmup=0                  # GRPO has no critic to warm up
    trainer.logger='["console","mlflow"]'    # air injects the MLflow context
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
    trainer.save_freq="${SAVE_FREQ:--1}"
    trainer.test_freq="${TEST_FREQ:--1}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
)
[ "${TOTAL_TRAIN_STEPS}" != "0" ] && TRAINER+=( trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" )

# --- reward ---------------------------------------------------------------
# Default: verl's built-in rule-based scorer, dispatched on the parquet's
# `data_source` column. For geo3k that is verl/utils/reward_score/geo3k.py:
#   0.9 * (boxed answer graded correct) + 0.1 * (<think></think> + \boxed{} format)
# No reward model, no critic, nothing to train.
#
# PHASE 2 HOOK: point CUSTOM_REWARD_PATH at scripts/reward/custom_reward.py
# (or your own) to override, without touching this launcher.
REWARD=()
if [ -n "${CUSTOM_REWARD_PATH:-}" ]; then
    REWARD+=(
        custom_reward_function.path="${CUSTOM_REWARD_PATH}"
        custom_reward_function.name="${CUSTOM_REWARD_NAME:-compute_score}"
    )
    echo "[info] custom reward: ${CUSTOM_REWARD_PATH}::${CUSTOM_REWARD_NAME:-compute_score}"
fi

EXTRA=( model_engine=megatron )   # current route; ppo_megatron_trainer.yaml is deprecated

# =============================================================================
# DRY_RUN — print the fully-resolved invocation and exit.
# MUST come before the Ray bootstrap, or a dry run would start a Ray head.
#   DRY_RUN=1 MEGATRON_MODE=fsdp NUM_NODES=2 LOCAL_WORLD_SIZE=8 \
#     bash scripts/run_grpo_megatron.sh
# =============================================================================
if [ "${DRY_RUN:-0}" = "1" ]; then
    set +x
    echo "---- resolved verl invocation (${MEGATRON_MODE}, offload=${OFFLOAD}) ----"
    printf 'python3 -m verl.trainer.main_ppo \\\n'
    for arg in "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" \
               "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" \
               ${REWARD[@]+"${REWARD[@]}"} "${EXTRA[@]}"; do
        printf '    %s \\\n' "${arg}"
    done
    n=$(( ${#ALGORITHM[@]} + ${#DATA[@]} + ${#MODEL[@]} + ${#ACTOR[@]} + ${#REF[@]} \
          + ${#ROLLOUT[@]} + ${#TRAINER[@]} + ${#REWARD[@]} + ${#EXTRA[@]} ))
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
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG}"
