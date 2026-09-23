# verl config & feature reference

← [verl-on-air](../README.md) · [configuration](configuration.md) · [tuning](tuning.md) · [training-modes](training-modes.md)

Exhaustive reference for **every** config parameter and verl feature the two launchers
set, why it has the value it has, and how a value flows from an air YAML into a
verl/Hydra override. Grounded line-by-line in the sources:

- `engine/train/run_grpo_megatron.sh` — the **sync** launcher (co-located rollout+train).
- `engine/train/run_grpo_fully_async.sh` — the **fully-async / disaggregated** launcher.
- `engine/lib/hparams.sh` — the air-`parameters:` → shell plumbing.

verl v0.9.0. For the *short* list of settings that matter, start at
[configuration.md](configuration.md) and [tuning.md](tuning.md); for the memory
arithmetic, [sizing.md](sizing.md). A few sections below reason about work not published
here (122B scaling, a dist-checkpoint bootstrap) — kept because the reasoning transfers,
but the scripts they name are not in the tree.

## Contents

1. [Big picture — what actually runs](#1-big-picture--what-actually-runs)
2. [How a parameter reaches verl (plumbing)](#2-how-a-parameter-reaches-verl-plumbing)
3. [`algorithm.*` — GRPO](#3-algorithm--grpo)
4. [`data.*` — dataset & batching](#4-data--dataset--batching)
5. [`actor_rollout_ref.model.*` — model & correctness flags](#5-actor_rollout_refmodel--model--correctness-flags)
6. [Megatron parallelism — TP/PP/CP/EP/ETP/GEN_TP](#6-megatron-parallelism--tpppcpepetpgen_tp)
7. [Backend modes — classic ZeRO-1 vs Megatron-FSDP ZeRO-3](#7-backend-modes--classic-zero-1-vs-megatron-fsdp-zero-3)
8. [`actor_rollout_ref.actor.*` — optimizer, PPO, KL, Megatron, offload](#8-actor_rollout_refactor--optimizer-ppo-kl-megatron-offload)
9. [`actor_rollout_ref.ref.*` — reference policy](#9-actor_rollout_refref--reference-policy)
10. [`actor_rollout_ref.rollout.*` — vLLM generation](#10-actor_rollout_refrollout--vllm-generation)
11. [`trainer.*` — orchestration, logging, checkpoint cadence](#11-trainer--orchestration-logging-checkpoint-cadence)
12. [Fully-async surface — top-level `rollout.*` + `async_training.*`](#12-fully-async-surface--top-level-rollout--async_training)
13. [The superseded v1 `separate_async` surface](#13-the-superseded-v1-separate_async-surface)
14. [Distributed checkpointing feature](#14-distributed-checkpointing-feature)
15. [`reward` — rule-based geo3k + custom hook](#15-reward--rule-based-geo3k--custom-hook)
16. [Process environment variables (non-Hydra)](#16-process-environment-variables-non-hydra)
17. [The fully-async exit-code guard](#17-the-fully-async-exit-code-guard)
18. [Quick reference — every env knob](#18-quick-reference--every-env-knob)

---

## 1. Big picture — what actually runs

Two launchers, two verl entrypoints, two Hydra config trees. Everything else is shared.

| | **sync** (`run_grpo_megatron.sh`) | **fully-async** (`run_grpo_fully_async.sh`) |
|---|---|---|
| Entry module | `verl.trainer.main_ppo` | `verl.experimental.fully_async_policy.fully_async_main` |
| Hydra config | default + `model_engine=megatron` | `--config-name=fully_async_ppo_megatron_trainer` (from the installed package's `config/` dir) |
| Rollout placement | **co-located** — one process trains and generates on the same GPUs; vLLM wakes to generate, sleeps to train | **disaggregated** — Rollouter and Trainer are separate processes on **disjoint** GPUs, MessageQueue between them |
| Weight sync | in-process actor→vLLM resync each step | cross-process NCCL broadcast (`checkpoint_engine.backend=nccl`) |
| Backends | `MEGATRON_MODE=fsdp` (default, ZeRO-3) **or** `classic` (ZeRO-1) | classic ZeRO-1 only |
| Step accounting | `trainer.total_training_steps` (dataloader-derived) | `rollout.total_rollout_steps` (streaming; sample budget) |

Both are invoked by air as `bash ${CODE_SOURCE_PATH}/engine/train/<launcher>.sh`, with the
launcher assembling a long list of Hydra dotted-path overrides into bash arrays and
`exec`ing python. `DRY_RUN=1` prints the fully-resolved invocation and exits before any
Ray bootstrap — the fastest way to see exactly what a given YAML will run.

The sync launcher passes `model_engine=megatron` (the current route; the older
`ppo_megatron_trainer.yaml` entry is deprecated in v0.9.0). The async launcher must
`cd` into verl's site-packages before launching because `fully_async_main`'s
`@hydra.main(config_path="config")` is **module-relative** and the megatron config
declares a CWD-relative `hydra.searchpath: file://verl/trainer/config`; verl is
pip-installed with no repo checkout, so neither path can be left to air's CWD.

---

## 2. How a parameter reaches verl (plumbing)

There are **two** input channels in every air YAML, and they are read differently:

**(a) `parameters:`** → materialized by air as a **YAML** file at `$HYPERPARAMETERS_PATH`,
read by the `hp` helper (`engine/lib/hparams.sh`):

```bash
MODEL_PATH="$(hp model_name "Qwen/Qwen3.5-35B-A3B")"   # hp <key> [default]
```

`hp` preserves the **missing-vs-empty** distinction via exit code 42: a key that is
*absent* returns the default, while `image_key: ""` returns the empty string (which is
how a YAML declares "text-only dataset, do not pass `data.image_key`" — collapsing empty
into the default would silently re-enable the multimodal path). Used for the per-run
knobs: `model_name`, `train_files`, `val_files`, `output_dir`, `total_epochs`,
`total_training_steps`, `train_batch_size`, `ppo_mini_batch_size`, `rollout_n`,
`max_prompt_length`, `max_response_length`, `actor_lr`, `image_key`.

**(b) `env_variables:`** → ordinary process environment, read with bash defaulting
(`${TP:-2}`). Used for topology/backend/feature switches: `TP`, `PP`, `CP`, `EP`, `ETP`,
`GEN_TP`, `MEGATRON_MODE`, `OFFLOAD`, `OFFLOAD_FRACTION`, `ROLLOUT_NNODES`,
`ROLLOUT_GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `SAVE_FREQ`, `USE_DIST_CKPT`, `DIST_CKPT_PATH`,
the async knobs, etc. (full list in §18).

**Hydra override syntax** the launchers emit — the prefix matters:

| form | meaning |
|---|---|
| `a.b.c=val` | override an **existing** config key |
| `+a.b.c=val` | **add** a key not in the schema (e.g. everything under `override_optimizer_config`) |
| `++a.b.c=val` | **force**-add/override even if present (used for `gradient_accumulation_fusion=False`) |

Three `override_*` groups are pass-throughs straight to the underlying libraries, which
is why they need `+`/`++` (they are not in verl's own schema):

- `actor.megatron.override_transformer_config.*` → Megatron `TransformerConfig` (recompute, MoE fusions, attention backend).
- `actor.optim.override_optimizer_config.*` → Megatron `OptimizerConfig` (CPU offload, precision-aware optimizer).
- `actor.megatron.override_ddp_config.*` → Megatron `DistributedDataParallelConfig` (FSDP sharding strategy).

---

## 3. `algorithm.*` — GRPO

| key | value | why |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | Group-Relative Policy Optimization: the advantage baseline is the **mean reward of the sampled group** (`rollout.n` completions per prompt), so there is **no value network → no critic worker at all**. That is a large part of why 35B/122B are tractable at this GPU budget. |
| `algorithm.use_kl_in_reward` | `False` | KL to the reference policy is applied as an explicit **loss** term (`actor.use_kl_loss=True`), not folded into the scalar reward. The two are mutually exclusive designs; we use the loss form. |

Both launchers set these identically.

---

## 4. `data.*` — dataset & batching

Shared:

| key | value | why |
|---|---|---|
| `data.train_files` / `data.val_files` | UC parquet paths | geo3k train/test (a **vision** diagram-reasoning task — see below). |
| `data.max_prompt_length` | `1024` | prompt cap; overlong prompts are filtered. |
| `data.max_response_length` | `2048` | generation cap; feeds the vLLM `max_model_len` sizing (§10). |
| `data.filter_overlong_prompts` | `True` | drop prompts over the cap rather than truncate mid-prompt. |
| `data.truncation` | `error` | if anything still exceeds the cap, **fail loudly** rather than silently truncate. |

Sync-only:

| key | value | why |
|---|---|---|
| `data.train_batch_size` | `hp train_batch_size` (32) | prompts per training step. verl requires `(train_batch_size × rollout.n) % trainer_gpus == 0`; the launcher checks this arithmetic up front (§8) and fails with the real numbers instead of letting verl raise "real_train_batch_size must be divisible…" after the cluster has spun up. |
| `data.shuffle` | `False` | deterministic ordering across the tiny geo3k subset. |
| `data.image_key` | `images` (only if `hp image_key` non-empty) | geo3k is **multimodal**; Qwen3.5 has a vision tower. Omitted for text-only datasets — the empty-string sentinel in §2 exists precisely to control this line. |

Async-only (streaming):

| key | value | why |
|---|---|---|
| `data.train_batch_size` | `0` | **streaming** mode — the sample budget is `rollout.total_rollout_steps`, not a per-step batch, so the training batch is "not effective". |
| `data.gen_batch_size` | `1` | streaming sample production granularity. |
| `data.return_raw_chat` | `True` | **required** for vLLM server / AgentLoop mode, which fully-async uses. |

> **geo3k is a vision task.** Qwen3.5-35B/122B are multimodal
> `Qwen3_5MoeForConditionalGeneration` (a `text_config` MoE + a `vision_config` tower).
> geo3k trains because the diagram is the input; the reward is on the boxed answer.

---

## 5. `actor_rollout_ref.model.*` — model & correctness flags

| key | value | why |
|---|---|---|
| `model.path` | UC dir (`…/models/Qwen3.5-122B-A10B`) | local UC copy, not a hub id — no egress at train time. |
| `model.trust_remote_code` | `True` | Qwen3.5 ships custom modeling code (Gated-DeltaNet, the VL-MoE arch). |
| `model.use_remove_padding` | `False` | **correctness, not tuning.** Qwen3.5's Gated-DeltaNet has **no THD (packed-sequence) support** in Megatron-LM, so the whole pipeline must run **BSHD**. This flag + the two `use_dynamic_bsz=False` flags below are the BSHD trio — differing from verl's recipe defaults. |
| `model.use_fused_kernels` | `False` (async only) | per verl's own Qwen3.5 async recipe. |
| `model.hybrid_engine` | `False` (async only) | disaggregated: the rollout is a **standalone** engine, not fused into the trainer process. (Sync co-located leaves this at the default hybrid engine.) |

---

## 6. Megatron parallelism — TP/PP/CP/EP/ETP/GEN_TP

Every parallel-degree knob is an env var read by both launchers and emitted to
`actor.megatron.*` (and mirrored on `ref.megatron.*`). GEN_TP is the **vLLM** rollout
degree and is independent of the trainer's TP.

| knob | override | meaning | notes |
|---|---|---|---|
| `TP` | `tensor_model_parallel_size` | trainer tensor parallel | must divide the model's attention heads (122B: 32 Q-heads → TP=2 ok). |
| `PP` | `pipeline_model_parallel_size` | pipeline stages | PP=2 splits the 48 layers into 2 stages → **halves per-GPU trainer footprint** (~76→~40 GiB); the lever that made 122B **co-located** sync fit. |
| `CP` | `context_parallel_size` | context parallel | 1 throughout (short sequences). |
| `EP` | `expert_model_parallel_size` | expert parallel | **the dominant MoE lever** — 92.5% of the 35B's weights are routed experts. 256 experts / EP = experts per rank. **EP=16 mandatory for 122B classic at ≤32 GPU**: classic replicates params+grads across DP, so EP=8 → ~70 GiB/GPU init OOM; EP=16 → ~41 GiB. |
| `ETP` | `expert_tensor_parallel_size` | tensor parallel **within** an expert | 1 throughout. |
| `GEN_TP` | `rollout.tensor_model_parallel_size` | **vLLM** tensor parallel | keep ≤ GPUs/node to stay intra-node (NVLink); cross-node GEN_TP≥16 uses NCCL and **dodges the custom-all-reduce kernel bug**. |

**Data-parallel degrees are derived, not set:**

- `train_DP = trainer_gpus / (TP × PP)` — `ppo_mini_batch_size` **must** be divisible by this (async launcher asserts it; §8).
- `expert_DP = train_DP / EP` (roughly) — at EP=16 on 16 trainer GPUs, expert-DP=1 (async 122B); on 32 GPUs, expert-DP=2 (sync 122B).

Worked examples from the two 122B finalists:

- **async:** 16 trainer GPU, TP2 PP1 EP16 → train-DP=8, expert-DP=1. (EP=16 spans both trainer nodes → cross-node expert all-to-all; a *conservative* async throughput — intra-node ETP=2/PP=2 would be fairer.)
- **sync:** 32 GPU, TP2 PP2 EP16 → train-DP=16, expert-DP=2.

---

## 7. Backend modes — classic ZeRO-1 vs Megatron-FSDP ZeRO-3

`MEGATRON_MODE` (sync launcher only; the async launcher is classic-only) selects the
sharding strategy. This is the single most consequential switch for memory.

| | **classic** (`MEGATRON_MODE=classic`) | **fsdp** (`MEGATRON_MODE=fsdp`, sync default) |
|---|---|---|
| ZeRO level | **ZeRO-1** — shards **optimizer** across DP; **replicates** params + grads | **ZeRO-3** — shards optimizer **+ grads + params** across DP |
| verl path | legacy mbridge (`vanilla_mbridge=True`) | Megatron-Bridge provider (`vanilla_mbridge=False` + `use_megatron_fsdp=True`) |
| when it fits | needs CPU offload at ≤16 GPU; the only 122B path at ≤32 GPU | offload-free 35B-A3B on 16×H100 (see `sizing.md`) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | **=1** (comm/compute overlap) | **unset** — with =1 the FSDP all-gather/reduce-scatter streams serialize behind compute and you silently lose the overlap |
| `use_precision_aware_optimizer` | **True** (halves Adam state 12→8 B/param) | **must NOT** be set — segfaults inside Transformer Engine's `multi_tensor_scale` grad-clip path on **all** ranks right after the first rollout |
| `gradient_accumulation_fusion` | (default on) | **False** (`++`) — Megatron-FSDP is incompatible with it |
| ddp sharding | n/a | `override_ddp_config.data_parallel_sharding_strategy=optim_grads_params` (explicit ZeRO-3; verl sets it by default when `use_megatron_fsdp=True` but we pin it so `sizing.md` can't be invalidated by a default change) |

`OFFLOAD` (`auto`|`0`|`1`) picks CPU offload. `auto`: fsdp offloads only at <16 GPU;
classic offloads at <32 GPU (classic replicates params/grads so it needs offload
sooner). The byte thresholds come from `sizing.md`.

> **Why classic for 122B.** ZeRO-3 (fsdp) 122B co-located would need ~128 GPU
> (`sizing.md`). Classic + **full CPU offload** (Adam in ~0.4–0.8 TB host RAM/node) is
> the only path that fits 122B at ≤32 GPU. Both 122B finalists are classic.

---

## 8. `actor_rollout_ref.actor.*` — optimizer, PPO, KL, Megatron, offload

**Optimizer & LR:**

| key | value | why |
|---|---|---|
| `actor.optim.lr` | `hp actor_lr` (1e-6) | GRPO actor LR. |
| `actor.optim.lr_decay_steps` | `LR_DECAY_STEPS` (=`total_rollout_steps`) | **async only, and mandatory.** Streaming (`train_batch_size=0`) gives verl no dataloader step count, so Megatron's `OptimizerParamScheduler` asserts `lr_decay_steps > 0` and the trainer dies at setup. The sync launcher derives this from the dataloader and doesn't set it. |

**PPO batching:**

| key | value | why |
|---|---|---|
| `actor.ppo_mini_batch_size` | `hp ppo_mini_batch_size` | optimizer-update granularity. Async: must be divisible by `train_DP` (launcher asserts). Sync 122B sets it == `train_batch_size` → exactly one optimizer update per step (like rung3). |
| `actor.ppo_micro_batch_size_per_gpu` | `1` | one sequence per micro-step — BSHD + memory-tight 122B. |
| `actor.ppo_max_token_len_per_gpu` | `4096` | token cap per micro-batch. |
| `actor.use_dynamic_bsz` | `False` | BSHD-required (see §5). |

**KL & entropy:**

| key | value | why |
|---|---|---|
| `actor.use_kl_loss` | `True` | KL-to-reference as a loss term. |
| `actor.kl_loss_coef` | `0.01` | KL weight. |
| `actor.kl_loss_type` | `low_var_kl` | the low-variance (k3) KL estimator. |
| `actor.entropy_coeff` | `0` | no entropy bonus. |

**Megatron block (`actor.megatron.*`):**

| key | value | why |
|---|---|---|
| `use_mbridge` | `True` | use mbridge for HF↔Megatron weight mapping. |
| `vanilla_mbridge` | `True` classic / `False` fsdp | legacy mbridge (the path verl's Qwen3.5 recipes test) vs the Bridge provider path (required to thread `use_megatron_fsdp`). |
| `use_megatron_fsdp` | `True` (fsdp only) | turns on ZeRO-3. |
| `use_remove_padding` | `False` | BSHD (§5), also set at the megatron level. |
| `tensor/pipeline/context/expert_model_parallel_size`, `expert_tensor_parallel_size` | §6 | the parallel degrees. |
| `dtype` | `bfloat16` | compute dtype. |
| `entropy_from_logits_with_chunking` | `True` | vocab is **248320**; un-chunked logits+entropy is ~3 GB per micro-batch, so chunk it. |

**`override_transformer_config.*` (Megatron `TransformerConfig`, via `+`):**

| key | value | why |
|---|---|---|
| `recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1` | recompute **everything** | this model is memory-bound, not compute-bound, here; activations are cheap to recompute relative to the HBM they free. **Both** launchers set this. |
| `moe_grouped_gemm=True`, `moe_permute_fusion=True` | MoE kernel fusions | with 256 experts these are the difference between usable and unusable throughput. **Sync launcher only.** |
| `moe_aux_loss_coeff=0.01`, `moe_z_loss_coeff=0.001` | MoE load-balancing / router-logit regularizers | **Sync launcher only.** |
| `attention_backend=auto` (`++`) | let Megatron pick the attention kernel | **classic sync only.** |

> **Documented asymmetry:** the **async** launcher sets **only** the three `recompute_*`
> flags in `override_transformer_config` — it does **not** set the MoE fusion / aux-loss /
> z-loss flags (it relies on verl/Megatron defaults for those). The async ACTOR comment
> mentions "MoE fusions are harmless for dense EP=1" but no such flags are actually
> emitted. If async MoE throughput ever needs tuning, this is the gap to close.

**`override_optimizer_config.*` (Megatron `OptimizerConfig`, via `+`) — offload group:**

| key | value | why |
|---|---|---|
| `optimizer_cpu_offload=True` | park optimizer state in host RAM | classic ZeRO-1 offload. |
| `optimizer_offload_fraction=${OFFLOAD_FRACTION}` (1) | fraction of Adam state offloaded | 1 = all of it (~0.4–0.8 TB host RAM/node). |
| `overlap_cpu_optimizer_d2h_h2d=True` | overlap the host↔device optimizer copies | hide the offload latency. |
| `use_precision_aware_optimizer=True` | 8-byte Adam state | **classic only** (segfaults under fsdp — §7). Async sets it inside its offload group; sync sets it separately whenever `MEGATRON_MODE!=fsdp`. |

The **async** launcher always emits `param_offload=True`, `optimizer_offload=True`,
`grad_offload=True` plus the four `override_optimizer_config` flags above (it is
classic-only and always offloads). The **sync** launcher emits `param_offload` +
`optimizer_offload` + the offload group **only under `OFFLOAD=1`** (and prints a host-RAM
warning), and sets `use_precision_aware_optimizer` independently of `OFFLOAD` whenever
mode is classic. Note the sync offload block does **not** set `grad_offload`.

**Up-front divisibility guards** (fail before the cluster spins up):

- **sync:** `(train_batch_size × rollout_n) % trainer_gpus == 0`.
- **async:** `ppo_mini_batch_size % train_DP == 0` (and `TP×PP ≤ trainer_gpus`).

---

## 9. `actor_rollout_ref.ref.*` — reference policy

The frozen reference model for the KL term. It mirrors the actor's parallelism and
Megatron/offload flags (so it builds and loads the same way), but has **no optimizer**:

| key | value | why |
|---|---|---|
| `ref.megatron.{tensor,pipeline,context,expert}_model_parallel_size`, `expert_tensor_parallel_size` | = actor | same sharding so weights map. |
| `ref.megatron.param_offload` | `True` | offload the frozen params too. |
| `ref.megatron.use_mbridge` / `vanilla_mbridge` / `use_megatron_fsdp` | = actor's mode | same load path. |
| `ref.log_prob_micro_batch_size_per_gpu` | `1` | BSHD log-prob pass. |
| `ref.log_prob_use_dynamic_bsz` | `False` | BSHD. |
| `ref.log_prob_max_token_len_per_gpu` | `4096` | token cap. |
| `ref.megatron.entropy_from_logits_with_chunking` | `True` | same 248320-vocab reason as the actor. |
| `ref.megatron.use_dist_checkpointing` + `dist_checkpointing_path` | when `USE_DIST_CKPT=True` | the ref **also loads its init weights**, so it needs the dist-ckpt source too (§14). |

---

## 10. `actor_rollout_ref.rollout.*` — vLLM generation

Shared vLLM config:

| key | value | why |
|---|---|---|
| `rollout.name` | `vllm` | the generation engine. |
| `rollout.tensor_model_parallel_size` | `GEN_TP` | vLLM TP (§6). |
| `rollout.gpu_memory_utilization` | `ROLLOUT_GPU_MEM_UTIL` (async default **0.8**, sync default **0.6**) | fraction of HBM vLLM may use. **Sizes only the KV cache**, not the weights — it is **not** the lever for a weight/KV-init OOM. Dedicated rollout GPUs run high (0.7); co-located runs low (0.3). |
| `rollout.n` | `hp rollout_n` (4) | completions per prompt — the GRPO group size. |
| `rollout.dtype` | `bfloat16` | generation dtype. |
| `rollout.calculate_log_probs` | `True` | needed for the importance ratio (and **required** by fully-async). |
| `rollout.log_prob_{micro_batch_size_per_gpu,max_token_len_per_gpu}` | `1`, `4096` | log-prob recompute batching. |
| `rollout.log_prob_use_dynamic_bsz` | `False` | BSHD. |
| `rollout.max_model_len` | `MAX_MODEL_LEN` (**8192**) | **caps the KV cache.** Unset, vLLM sizes KV for the model's config max (**262144**) → ~3 GiB KV/request → co-located KV OOM. We use prompt(≤1024)+response(2048); 8192 covers that plus VL image-token margin at ~0.1 GiB/request. |
| `rollout.max_num_batched_tokens` | `MAX_MODEL_LEN` (8192) | prefill batch cap, kept equal to `max_model_len`. |
| `rollout.free_cache_engine` | `True` | free the KV cache between generate and train so the two memory peaks **don't sum** (matters co-located). |
| `rollout.enable_chunked_prefill` | `True` | chunk long prefills. |
| `rollout.enable_prefix_caching` | `False` | off (short, mostly-unique prompts). |

Mode-specific:

| key | launcher | why |
|---|---|---|
| `rollout.mode=async` | async | vLLM **server / AgentLoop** mode — required by fully-async. |
| `rollout.checkpoint_engine.backend=nccl` | async (and sync separate_async) | trainer→rollout weight sync over NCCL (not the ZeRO-3 full-gather), so the trainer pays **no vLLM memory tax**. |
| `rollout.enforce_eager` | both, but **different sentinels** | disables vLLM CUDA-graph capture. **This is the workaround** and a per-launcher gotcha (below). |

> **`enforce_eager` — same feature, two different env conventions:**
> - **async:** `ROLLOUT_ENFORCE_EAGER` maps its value straight through — pass `'True'`/`'False'` (default `False`).
> - **sync:** the launcher tests `[ "$ROLLOUT_ENFORCE_EAGER" = "1" ]` — pass `'1'` (default off).
>
> Why it exists: at **intra-node GEN_TP≤8** on df1 H100, vLLM's CUDA-graph capture hits a
> broken custom-all-reduce kernel (`custom_all_reduce.cuh:455 'invalid argument'`) and
> kills every rollout worker at init. `enforce_eager=True` skips capture and dodges
> it — but eager generation is ~5–7× slower (122B gen ~49s → 220–350s/step). The **fair**
> fix is cross-node GEN_TP≥16 (NCCL all-reduce, keeps CUDA graphs), which is why the sync
> 122B run used GEN_TP=16 and needed no eager workaround.

**Weight-sync bucket (sync only, currently unset):** `WEIGHT_BUCKET_MB` →
`rollout.checkpoint_engine.update_weights_bucket_megabytes`. The actor→vLLM sync moves
tensors in fixed-size buckets; a tensor larger than the bucket aborts ("too large to fit
in the bucket"). The 248320×2048 embedding is ~970 MiB bf16, so any bucket must exceed
that. Left unset because the config path moved between verl releases (see the launcher
comment / `troubleshooting.md`).

---

## 11. `trainer.*` — orchestration, logging, checkpoint cadence

| key | value | why |
|---|---|---|
| `trainer.critic_warmup` | `0` | GRPO has no critic to warm up. |
| `trainer.logger` | `["console","mlflow"]` | air injects the MLflow run context; **per-step metrics land in MLflow**, not the driver console. |
| `trainer.project_name` / `experiment_name` | env / `hp` | MLflow grouping. |
| `trainer.nnodes` | `TRAINER_NNODES` | trainer node count (= NNODES − ROLLOUT_NNODES). |
| `trainer.n_gpus_per_node` | `NGPUS_PER_NODE` (async: `TRAINER_N_GPUS`) | per-node GPU count for the trainer pool. |
| `trainer.default_local_dir` | `hp output_dir` | checkpoint destination (a UC Volume path). **Use a fresh dir per run** — verl's default `trainer.resume_mode=auto` silently resumes from whatever checkpoint is already there. |
| `trainer.val_before_train` | `False` | skip the pre-train eval (perf runs). |
| `trainer.save_freq` | `SAVE_FREQ` (**-1** = never; perf runs **4**) | **async:** counts param-sync versions + a forced save at completion. **sync:** counts trainer global steps. Either way `=4` → ~one sharded dist-ckpt mid-run. |
| `trainer.test_freq` | `TEST_FREQ` (-1) | eval cadence, off. |
| `trainer.total_epochs` | `hp total_epochs` | data-headroom bound; the real bound is steps (below). |
| `trainer.total_training_steps` | `hp total_training_steps` (sync only, if ≠0) | hard step cap — how the sync perf run stops at 6 steps. |

---

## 12. Fully-async surface — top-level `rollout.*` + `async_training.*`

These exist **only** in the fully-async recipe and are what make it truly disaggregated
(vs the superseded v1 path in §13). Note `rollout.nnodes` / `rollout.n_gpus_per_node`
here are **top-level** — the Rollouter's own resource pool, disjoint from `trainer.*`.

| key | value | why |
|---|---|---|
| `rollout.nnodes` | `ROLLOUT_NNODES` | whole nodes given to the Rollouter. With `use_dynamic_resource_scheduling=False` (default) the trainer GPUs stay rollout-free → real disaggregation. |
| `rollout.n_gpus_per_node` | `ROLLOUT_N_GPUS` | GPUs per rollout node. |
| `rollout.total_rollout_steps` | `hp total_rollout_steps` | **total rollout SAMPLES** — the run's actual size bound in streaming mode. |
| `async_training.staleness_threshold` | `STALENESS` (0.1) | `0` = synchronous; `>0` = genuinely async (how stale a trajectory may be vs the current weights). |
| `async_training.trigger_parameter_sync_step` | `TRIGGER_SYNC_STEP` (2) | local optimizer updates between weight syncs. |
| `async_training.require_batches` | `REQUIRE_BATCHES` (1) | mini-batches fetched per update. |
| `async_training.partial_rollout` | `PARTIAL_ROLLOUT` (True) | allow partial (interruptible) rollouts for freshness. |

**The sample-budget arithmetic** (drives run duration):

```
samples per sync = trigger_parameter_sync_step × require_batches × ppo_mini_batch_size
number of syncs  = total_rollout_steps / samples-per-sync
```

e.g. `2 × 1 × 16 = 32` samples/sync; `total_rollout_steps=128` → **4 syncs**
(step 1 is compile, the rest give steady metrics). For a 35B run: `total_rollout_steps=512`
→ ~16 syncs → ~32 optimizer steps (a real reward trajectory, not a 2-step smoke).

**Topology split** (the launcher supports two modes):

- **whole-node** (`ROLLOUT_NNODES≥1`, multi-node): Rollouter gets that many whole nodes, Trainer gets the rest; each side uses all `NGPUS_PER_NODE`. This is what 35B/122B use.
- **within-node** (`ROLLOUT_NNODES=0`, single node): the one node is split `N_GPUS_ROLLOUT` rollout + rest trainer (the canonical 4+4 geo3k example).

---

## 13. The superseded v1 `separate_async` surface

`run_grpo_megatron.sh` still carries a `TRAINER_MODE=separate_async` block
(`trainer.v1.trainer_mode=separate_async` + `separate_async.parameter_sync_step` +
`num_warmup_batches` + `sampler.max_off_policy_threshold`, plus `hybrid_engine=False`
and the top-level `rollout.{nnodes,n_gpus_per_node,checkpoint_engine.backend}`).

**Do not use it for disaggregation.** It places the standalone rollout at
`start_rank=hybrid_num_replicas`, which with `hybrid_engine=False` is **0** — so the
rollout **collides onto the trainer GPUs** → vLLM OOM. The
fully-async recipe (§12) is the correct disaggregation path. The block is kept only
because its hard asserts (`train_batch_size == parameter_sync_step × ppo_mini_batch_size`,
`checkpoint_engine.backend != naive`, `rollout.nnodes > 0`) document the v1 contract.

---

## 14. Distributed checkpointing feature

The single most important 122B-enabling feature, and a subtle one because **one flag
controls two things**.

`actor_rollout_ref.actor.megatron.use_dist_checkpointing` (default `False`):

| | `False` (default) | `True` |
|---|---|---|
| **checkpoint save** | `model` content exported as a **full-gather HF** file via mbridge (`_save_model_as_hf_via_bridge`) → gathers all weights onto one GPU → **OOMs at 122B** | **sharded** Megatron dist checkpoint, no gather |
| **init weight-load** | load from HF `model.path` | load from `dist_checkpointing_path` |

So to get the sharded save you also flip init onto the dist path — which means you must
**pre-build a dist checkpoint from HF first**. Enabled via env in both launchers:

```
USE_DIST_CKPT=True  DIST_CKPT_PATH=/Volumes/.../Qwen3.5-122B-A10B-mcore-dist
```

which appends `use_dist_checkpointing=True` + `dist_checkpointing_path=…` to **both** the
actor and ref arrays (the ref loads init weights too — §9).

**The bootstrap (a converter script, not published on this branch -- see git history):** the stock
`scripts/converter_hf_to_mcore.py` **cannot** convert Qwen3.5 — it dispatches by
architecture, and multimodal `Qwen3_5MoeForConditionalGeneration` isn't registered, so it
falls into the text-only branch that can't build the vision tower. Our converter instead
reuses verl's **exact vanilla-mbridge init path**:

```python
bridge   = AutoBridge.from_config(hf_config, dtype)   # verl-patched mbridge
tf_config = bridge.config;  tf_config.bf16 = True
module, _ = make_megatron_module(wrap_config, tf_config, hf_config, bridge=bridge, provider=None)
bridge.load_weights(module, hf_path)                  # HF safetensors -> megatron module
dist_checkpointing.save(unwrap_model(chunks[0]).sharded_state_dict(), out,
                        sharded_strategy=None, async_sharded_save=False)
```

This is byte-for-byte what verl's `load_mcore_dist_weights()` reads back at init.
`wrap_config` uses `wrap_with_ddp=False` / `use_distributed_optimizer=False` (pure
conversion — no optimizer, grads, or activations), and `share_embeddings_and_output_weights`
follows the HF `tie_word_embeddings` flag.

**Reshard-aware:** convert at **TP=1/EP=8** (8×H100, one node), train at
**TP=2/EP=16** — dist_checkpointing reshards on load. EP shards the 256 experts so no rank
holds all of them (~40–55 GiB/GPU during convert, comfortable on 80).

**Two operational gotchas in that conversion:**

1. **Rendezvous:** `torchrun --standalone` binds the TCPStore to the container hostname
   (`node.host.local`), unroutable back to itself in the air network (errno 113).
   Force `--master_addr=127.0.0.1 --master_port=29500` (not `--standalone`).
2. **UC write:** the FUSE mount rejects **parallel** range writes (torch_dist writes many
   shards at once). Write to node-local `/local_disk0` (fast NVMe) first, then a
   **sequential** `cp -r` onto the UC Volume.

Result: `…/models/Qwen3.5-122B-A10B-mcore-dist`, ~245 GB / 8 shards.
Details: memory `verl-122b-dist-checkpoint`.

---

## 15. `reward` — rule-based geo3k + custom hook

Default (no config needed): verl's **built-in rule-based scorer**, dispatched on the
parquet's `data_source` column. For geo3k that is `verl/utils/reward_score/geo3k.py`:

```
reward = 0.9 × (boxed answer graded correct) + 0.1 × (<think></think> + \boxed{} format)
```

**No reward model, no critic, nothing extra to train** — which (with GRPO's missing value
net) is why this fits at all.

**Phase-2 hook (sync launcher):** set `CUSTOM_REWARD_PATH` (and optionally
`CUSTOM_REWARD_NAME`, default `compute_score`) to override with your own scorer via
`custom_reward_function.path` / `.name`, without touching the launcher.

---

## 16. Process environment variables (non-Hydra)

Set by the launchers (or air) into the process env — not Hydra overrides, but they change
behavior materially:

| var | value | why |
|---|---|---|
| `OPENSSL_FORCE_FIPS_MODE`, `OPENSSL_FIPS` | `0`, `0` | air hosts run a **FIPS kernel**; non-FIPS crypto in the image aborts on SSL init unless these are cleared. Set for the driver; Ray workers inherit. |
| `VLLM_USE_V1` | `1` | vLLM **v1** engine — required for the async server/AgentLoop mode; set in both launchers. |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `0` | disables the symmetric-memory all-reduce path. (Note: this does **not** prevent the custom-all-reduce crash — that's a different path.) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` (classic) / **unset** (fsdp) | classic wants =1 for comm/compute overlap; fsdp **requires it unset** or the FSDP collectives serialize behind compute (§7). The async launcher (classic-only) sets =1. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduces allocator fragmentation so the 122B dist-ckpt **save buffers fit** (async launcher; per the run-444 save-time OOM hint). |
| `PATH` | prepend `/opt/venv/bin` if `ray` missing | air's `command:` can run with a minimal PATH that omits the venv; a plain-scalar `command:` on a multi-node job caused `ray: command not found` → exit 127. Guarded in both launchers. |
| `NCCL_DEBUG` | `WARN` | EFA/RDMA already proven on df1 multi-node; errors still surface. Bump to `INFO` only when a cross-node sync misbehaves. |
| `VLLM_LOGGING_LEVEL` | `INFO` | see the AgentLoop server + KV/weights map at init. |
| `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `TOKENIZERS_PARALLELISM`, `RAY_TEMP_DIR` | from air YAML | HF cache to `/tmp`, fast HF downloads, tokenizer threading, Ray scratch. |

> **Never** set `RAY_RUNTIME_ENV_HOOK=""` — Ray tries to import the empty string as a
> class path and dies with "expected a valid path like mymodule.provider_class".

---

## 17. The fully-async exit-code guard

A launcher **feature**, not a verl config, but essential to reading run outcomes.

verl's experimental `fully_async_main` runs the Trainer and Rollouter as concurrent
components, and its exit status is wrong **in both directions** (pinned v0.9.0 source):

- **Success exits non-zero.** On normal completion the finished component cancels the
  other, raising `RuntimeError: cancelled` in the vLLM EngineCore → `RayTaskError` → exit
  1; air marks the job FAILED.
- **Failure can exit 0.** The Rollouter gathers its tasks with `return_exceptions=True`
  and then sends the ordinary stop signal, so a rollout or reward crash ends as a normal
  stop: the Trainer force-saves the version it reached and both components "complete
  successfully". `[ASYNC MAIN] Training completed or interrupted` is printed from a
  `finally:` block — after failures too — so no log marker means success.

So the launcher decides from the checkpoints, for **every** exit code
([`engine/lib/run_certificate.py`](../engine/lib/run_certificate.py)). A run is SUCCESS only if:

1. `<output_dir>/latest_checkpointed_iteration.txt` was (re)written **during this run**
   (a pre-launch snapshot excludes a tracker left by an earlier run in the same dir) —
   verl writes it only after the actor **and** rollouter saves have returned;
2. its value is the **planned final version** = `total_rollout_steps / (trigger_parameter_sync_step
   × require_batches × ppo_mini_batch_size)`; the launcher refuses budgets that do not divide
   exactly, and requires `SAVE_FREQ > 0` and `TEST_FREQ ≠ 0` (0 divides by zero at the end of
   `fit()` and skips the final save);
3. that `global_step_N` verifies — `actor/ckpt_contents.json` (written last, atomically)
   plus a complete HF export ([`engine/lib/verify_checkpoint.py`](../engine/lib/verify_checkpoint.py));
4. no component raised the abort channel (`<rendezvous>/ABORT.json`, see
   [`engine/lib/run_control.py`](../engine/lib/run_control.py)) — a watchdog stops the run
   as soon as one appears;
5. when overriding a **non-zero** exit, the log shows no hard-failure signature (OOM, NCCL,
   CUDA, assert, engine-init): defence in depth.

The verdict (reasons included) is written to `<output_dir>/run_result.json`.
`ALLOW_UNCERTIFIED=1` with `SAVE_FREQ=-1` is available for throwaway smokes and reports the
raw exit code, marked as uncertified.

> **Lesson: verify a SUCCESS against MLflow step metrics, not the `air` label.** A run
> that never logged a training step in MLflow did not train, whatever `air` says.

---

## 18. Quick reference — every env knob

**Async launcher (`run_grpo_fully_async.sh`)** — via `env_variables:`:

| env | default | effect |
|---|---|---|
| `ROLLOUT_NNODES` | `0` | whole rollout nodes (≥1 → whole-node split; 0 → within-node) |
| `N_GPUS_ROLLOUT` | `4` | rollout GPUs (within-node split only) |
| `TP` `PP` `CP` `EP` `ETP` | `2 1 1 1 1` | trainer parallel degrees |
| `GEN_TP` | `1` | vLLM rollout TP |
| `OFFLOAD_FRACTION` | `1` | Adam fraction offloaded to host RAM |
| `ROLLOUT_GPU_MEM_UTIL` | `0.8` | vLLM HBM fraction (KV sizing) |
| `ROLLOUT_ENFORCE_EAGER` | `False` | pass `'True'` to skip CUDA-graph capture |
| `MAX_MODEL_LEN` | `8192` | vLLM context / KV cap |
| `TRIGGER_SYNC_STEP` | `2` | local updates between weight syncs |
| `REQUIRE_BATCHES` | `1` | mini-batches per update |
| `STALENESS` | `0.1` | async freshness (0 = sync) |
| `PARTIAL_ROLLOUT` | `True` | interruptible rollouts |
| `LR_DECAY_STEPS` | `=total_rollout_steps` | LR horizon (mandatory) |
| `SAVE_FREQ` | `-1` | dist-ckpt save cadence (param-sync versions) |
| `USE_DIST_CKPT` / `DIST_CKPT_PATH` | `False` / — | sharded save + dist init-load |
| `NNODES` `NGPUS_PER_NODE` `NODE_RANK` `MASTER_ADDR` | derived | cluster topology (usually from air) |

**Sync launcher (`run_grpo_megatron.sh`)** — via `env_variables:`:

| env | default | effect |
|---|---|---|
| `MEGATRON_MODE` | `fsdp` | `fsdp` (ZeRO-3) or `classic` (ZeRO-1) — §7 |
| `OFFLOAD` | `auto` | `auto`\|`0`\|`1` CPU offload |
| `OFFLOAD_FRACTION` | `1` | Adam fraction offloaded (under OFFLOAD=1) |
| `TP` | `1` fsdp / `2` classic | trainer TP |
| `PP` `CP` `EP` `ETP` | `1 1 8 1` | trainer parallel degrees |
| `GEN_TP` | `8` | vLLM rollout TP (≥16 → cross-node, dodges the custom-all-reduce crash) |
| `ROLLOUT_GPU_MEM_UTIL` | `0.6` | vLLM HBM fraction |
| `ROLLOUT_ENFORCE_EAGER` | `0` | pass `'1'` to skip CUDA-graph capture |
| `MAX_MODEL_LEN` | `8192` | vLLM context / KV cap |
| `WEIGHT_BUCKET_MB` | unset | actor→vLLM weight-sync bucket size |
| `TRAINER_MODE` | `sync` | `sync` or (superseded) `separate_async` — §13 |
| `ROLLOUT_NNODES` | `0` | standalone rollout nodes (separate_async only) |
| `PARAM_SYNC_STEP` `ASYNC_WARMUP_BATCHES` `MAX_OFF_POLICY` | derived / `1` / unset | separate_async knobs |
| `SAVE_FREQ` `TEST_FREQ` | `-1` `-1` | checkpoint / eval cadence (global steps) |
| `USE_DIST_CKPT` / `DIST_CKPT_PATH` | `False` / — | sharded save + dist init-load |
| `VAL_BEFORE_TRAIN` | `False` | pre-train eval |
| `CUSTOM_REWARD_PATH` / `CUSTOM_REWARD_NAME` | unset / `compute_score` | custom reward hook (§15) |

Plus air `parameters:` (both launchers, via `hp`): `model_name`, `train_files`,
`val_files`, `output_dir`, `total_epochs`, `total_training_steps` (sync), `train_batch_size`
(sync), `ppo_mini_batch_size`, `rollout_n`, `max_prompt_length`, `max_response_length`,
`actor_lr`, `image_key` (sync).
