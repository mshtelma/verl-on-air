# verl flags the launchers set

What the two launchers pass to verl v0.9.0, and why each value is what it is. The settings you
are expected to change are in [configuration.md](configuration.md); the memory arithmetic behind
the parallel sizes is in [sizing.md](sizing.md). To see the exact overrides for one job, run it
with `DRY_RUN=1` (or `make config`, `make compose-check`); nothing starts, and it works on a laptop.

## The two launchers

| | sync (`engine/train/run_grpo_megatron.sh`) | fully-async (`engine/train/run_grpo_fully_async.sh`) |
|---|---|---|
| entry module | `verl.trainer.main_ppo` with `model_engine=megatron` | `verl.experimental.fully_async_policy.fully_async_main` |
| Hydra config | verl's default | `--config-name=fully_async_ppo_megatron_trainer` |
| rollout placement | co-located: vLLM wakes to generate and sleeps while the same GPUs train | Rollouter and Trainer are separate processes on disjoint GPUs, joined by a message queue |
| weight sync | in-process actor-to-vLLM resync every step | NCCL broadcast between the processes (`checkpoint_engine.backend=nccl`) |
| backends | `MEGATRON_MODE=fsdp` (ZeRO-3, default) or `classic` (ZeRO-1) | classic ZeRO-1 with CPU offload only |
| run length | `trainer.total_training_steps` | `rollout.total_rollout_steps` (a sample budget; streaming) |

Each launcher builds its Hydra overrides in bash arrays and `exec`s Python. The async launcher
`cd`s into verl's site-packages first, because that config's `hydra.searchpath` is relative to the
working directory and verl is installed without a repo checkout.

## How values reach verl

`parameters:` are read with `hp <key> <default>` (`engine/lib/hparams.sh`): `model_name`,
`train_files`, `val_files`, `output_dir`, `total_epochs`, `total_training_steps`,
`train_batch_size`, `ppo_mini_batch_size`, `rollout_n`, `total_rollout_steps`,
`max_prompt_length`, `max_response_length`, `actor_lr`, `image_key`. `hp` tells an absent key
from an empty one, so `image_key: ''` drops `data.image_key` instead of falling back to `images`.
`env_variables:` are read as `${VAR:-default}`.

The prefix on an override matters:

| form | meaning |
|---|---|
| `a.b.c=val` | override a key that exists in verl's schema |
| `+a.b.c=val` | add a key that is not in the schema |
| `++a.b.c=val` | add or override either way (used for `gradient_accumulation_fusion=False`) |

Three groups go straight to the underlying libraries, which is why they need `+` or `++`:
`actor.megatron.override_transformer_config.*` (Megatron `TransformerConfig`: recompute, MoE
fusions, attention backend), `actor.optim.override_optimizer_config.*` (Megatron
`OptimizerConfig`: CPU offload, precision-aware optimizer) and
`actor.megatron.override_ddp_config.*` (`DistributedDataParallelConfig`: the FSDP sharding
strategy).

## `algorithm.*`

| key | value | why |
|---|---|---|
| `adv_estimator` | `grpo` | the baseline is the mean reward of the prompt's group of `rollout.n` samples, so there is no value network and no critic worker |
| `use_kl_in_reward` | `False` | the KL to the reference policy is a loss term (`actor.use_kl_loss=True`) instead of part of the reward |

## `data.*`

| key | value | launcher | why |
|---|---|---|---|
| `max_prompt_length` / `max_response_length` | from `parameters` | both | multi-turn replaces `max_response_length` with the episode budget (see rollout below) |
| `filter_overlong_prompts` | `True` | both | drop prompts over the cap instead of cutting them |
| `truncation` | `error` | both | anything still over the cap fails loudly |
| `seed` | `SEED` | both | set only when `SEED` is; together with the Megatron and rollout seeds it fixes the data order and sampling |
| `train_batch_size` | `hp train_batch_size` | sync | prompts per step; `train_batch_size × rollout_n` must divide by the trainer GPUs, checked before the cluster starts |
| `shuffle` | `DATA_SHUFFLE` (`False`) | sync | the geo3k ladder keeps a fixed order; the sync search job sets `True` to match async |
| `image_key` | `images` if `hp image_key` is non-empty | sync | geo3k is a vision task (Qwen3.5 is a vision-language model); text-only jobs set `image_key: ''` |
| `train_batch_size` | `0` | async | streaming: the budget is `rollout.total_rollout_steps` |
| `gen_batch_size` | `1` | async | streaming granularity |
| `return_raw_chat` | `True` | async, and sync multi-turn | required by vLLM server mode and the agent loop |

## `actor_rollout_ref.model.*`

| key | value | why |
|---|---|---|
| `path` | the staged model on the Volume | no downloads at training time |
| `trust_remote_code` | `True` | Qwen3.5 ships custom modelling code |
| `use_remove_padding` | `False` | required: Qwen3.5's Gated-DeltaNet layers have no packed-sequence (THD) support in Megatron-LM, so everything runs padded (BSHD). The two `use_dynamic_bsz=False` settings below belong to the same requirement |
| `use_fused_kernels` | `False` (async) | as in verl's own Qwen3.5 async recipe |
| `hybrid_engine` | `False` (async) | the rollout is a standalone engine, not fused into the trainer |

## Parallelism

Every degree is an env var, emitted to `actor.megatron.*` and mirrored on `ref.megatron.*`.

| knob | override | notes |
|---|---|---|
| `TP` | `tensor_model_parallel_size` | must divide the attention heads |
| `PP` | `pipeline_model_parallel_size` | splits the layers into stages, which shrinks the trainer's per-GPU footprint; `1` in the shipped jobs |
| `CP` | `context_parallel_size` | `1` everywhere |
| `EP` | `expert_model_parallel_size` | the main lever for this MoE: 92.5% of the weights are routed experts, 256 experts divided by EP per rank |
| `ETP` | `expert_tensor_parallel_size` | `1` everywhere |
| `GEN_TP` | `rollout.tensor_model_parallel_size` | vLLM's tensor parallel, independent of the trainer's. Keep it within a node (NVLink) |

Data parallelism is derived: `DP = trainer GPUs / (TP × PP)`, and `ppo_mini_batch_size` must be
divisible by it. The expert data-parallel size is roughly `DP / EP`.

## Backends: classic Megatron and Megatron-FSDP

`MEGATRON_MODE` exists only in the sync launcher; the async launcher is always classic.

| | `classic` (ZeRO-1) | `fsdp` (ZeRO-3, sync default) |
|---|---|---|
| shards | the optimizer, over DP; params and grads are replicated | optimizer, grads and params |
| verl path | legacy mbridge (`vanilla_mbridge=True`) | the Megatron-Bridge provider (`vanilla_mbridge=False`, `use_megatron_fsdp=True`) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1`, for comm/compute overlap | unset; at `1` the FSDP collectives wait behind compute |
| `use_precision_aware_optimizer` | `True` (Adam state 12 to 8 bytes per param) | must not be set: it segfaults in TransformerEngine's `multi_tensor_scale` during gradient clipping |
| `gradient_accumulation_fusion` | default | `False` (`++`), incompatible with Megatron-FSDP |
| DDP sharding | n/a | `override_ddp_config.data_parallel_sharding_strategy=optim_grads_params`, which verl also sets by default; pinned so a default change cannot invalidate sizing.md |

`OFFLOAD=auto` offloads classic below 32 trainer GPUs and never offloads fsdp; fsdp below 16 GPUs
stops, because CPU offload crashes it.

## `actor_rollout_ref.actor.*`

| key | value | why |
|---|---|---|
| `optim.lr` | `hp actor_lr` | the policy learning rate |
| `optim.lr_decay_steps` | `LR_DECAY_STEPS` (= `total_rollout_steps`) | async only and required: streaming gives verl no step count, and Megatron's scheduler asserts `lr_decay_steps > 0` at setup |
| `ppo_mini_batch_size` | `hp ppo_mini_batch_size` | the optimizer update size; must divide by DP |
| `ppo_micro_batch_size_per_gpu` | `1` | one sequence per micro-step |
| `ppo_max_token_len_per_gpu` | `4096`, or the episode length for multi-turn | tokens per micro-batch |
| `use_dynamic_bsz` | `False` | required by the padded (BSHD) path |
| `use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | `True` / `0.01` / `low_var_kl` | KL to the reference as a loss, with the low-variance (k3) estimator |
| `entropy_coeff` | `0` | no entropy bonus |
| `megatron.use_mbridge` | `True` | HF-to-Megatron weight mapping |
| `megatron.dtype` | `bfloat16` | compute dtype |
| `megatron.entropy_from_logits_with_chunking` | `True` | the vocabulary has 248320 entries; unchunked logits and entropy take about 3 GB per micro-batch |
| `megatron.seed` | `SEED` | set only when `SEED` is |

`override_transformer_config` (via `+`) sets full recompute in both launchers
(`recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1`), because this
model is memory-bound here. The sync launcher also sets `moe_grouped_gemm=True`,
`moe_permute_fusion=True`, `moe_aux_loss_coeff=0.01` and `moe_z_loss_coeff=0.001`, and classic
sync adds `++attention_backend=auto`. The async launcher leaves the MoE settings at verl's and
Megatron's defaults.

Offload:

| key | value | why |
|---|---|---|
| `megatron.param_offload` / `optimizer_offload` | `True` | park parameters and optimizer state in host RAM |
| `megatron.grad_offload` | `True` (async only) | the sync offload block does not set it |
| `optim.override_optimizer_config.optimizer_cpu_offload` | `True` | Megatron's CPU optimizer |
| `optim.override_optimizer_config.optimizer_offload_fraction` | `OFFLOAD_FRACTION` (`1`) | share of the Adam state offloaded |
| `optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d` | `True` | hides the host-device copies |

The async launcher always offloads. The sync launcher offloads only with `OFFLOAD=1`, and sets
`use_precision_aware_optimizer` whenever the mode is classic.

## `actor_rollout_ref.ref.*`

The frozen reference for the KL term mirrors the actor's parallel sizes and load path
(`use_mbridge`, `vanilla_mbridge`, `use_megatron_fsdp`, and `++gradient_accumulation_fusion=False`
under fsdp), but has no optimizer. It sets `param_offload=True` whenever the actor offloads,
`log_prob_micro_batch_size_per_gpu=1`, `log_prob_use_dynamic_bsz=False`,
`log_prob_max_token_len_per_gpu` equal to the actor's, and chunked entropy. With
`USE_DIST_CKPT=True` it also loads its weights from `DIST_CKPT_PATH`.

## `actor_rollout_ref.rollout.*`

| key | value | why |
|---|---|---|
| `name` | `vllm` | the generation engine |
| `mode` | `async` (async, and sync multi-turn) | vLLM server mode, which the agent loop needs; here "async" names the vLLM engine mode, not the training mode |
| `tensor_model_parallel_size` | `GEN_TP` | |
| `gpu_memory_utilization` | `ROLLOUT_GPU_MEM_UTIL` (`0.8` async, `0.6` sync) | sizes the KV cache only, not the weights. Dedicated rollout GPUs run high; co-located runs lower |
| `n` | `hp rollout_n` | the GRPO group size |
| `temperature` | `ROLLOUT_TEMP` (`1.0`) | async only |
| `seed` | `SEED` | set only when `SEED` is |
| `dtype` | `bfloat16` | |
| `calculate_log_probs` | `True` | needed for the importance ratio, and required by fully-async |
| `log_prob_micro_batch_size_per_gpu` / `log_prob_use_dynamic_bsz` / `log_prob_max_token_len_per_gpu` | `1` / `False` / `4096` or the episode | padded log-prob recompute |
| `max_model_len` / `max_num_batched_tokens` | `MAX_MODEL_LEN` (`8192`, or the episode length) | caps the KV cache. Unset, vLLM sizes it for the model's 262144-token maximum, about 3 GiB per request |
| `free_cache_engine` | `True` | free the KV cache between generating and training so the two peaks do not add up |
| `enable_chunked_prefill` | `True` | |
| `enable_prefix_caching` | `ROLLOUT_PREFIX_CACHING` (async), `False` (sync) | verl flushes the cache on each weight sync |
| `enforce_eager` | `ROLLOUT_ENFORCE_EAGER` (`False`) | skips CUDA-graph capture: a few GiB saved, generation several times slower |
| `checkpoint_engine.backend` | `nccl` (async) | the trainer pushes weights over NCCL, so it holds no vLLM copy |
| `+engine_kwargs.vllm.disable_custom_all_reduce` | `True` when `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=True` (async) | with `GEN_TP` of 8 or less inside one node, vLLM's custom all-reduce crashes CUDA-graph capture on H100 and kills the rollout workers; NCCL is used instead and graphs are kept. It is scoped to the rollout, so the trainer keeps its `expandable_segments` allocator. Cross-node `GEN_TP` of 16 or more uses NCCL anyway |
| `checkpoint_engine.update_weights_bucket_megabytes` | `WEIGHT_BUCKET_MB` (sync, unset) | the actor-to-vLLM sync moves tensors in buckets, and the 970 MiB embedding must fit in one. Unset because the key moved between verl releases |

With `MULTI_TURN=True` both launchers add `multi_turn.enable=True`,
`multi_turn.max_assistant_turns` and `max_user_turns` (`MAX_TURNS`),
`multi_turn.max_tool_response_length`, `multi_turn.format` (`TOOL_FORMAT`),
`multi_turn.function_tool_path`, `agent.num_workers` and `agent.agent_loop_config_path`
(`engine/train/agent_loops.yaml`, which registers the role-span tool agent). They set
`prompt_length`, and `response_length` to the episode budget
`(max_prompt_length + max_response_length) × MAX_TURNS − max_prompt_length`, raise every token
limit above to the episode length, and pass `PYTHONPATH` to the Ray workers.

## `trainer.*`

| key | value | why |
|---|---|---|
| `critic_warmup` | `0` | GRPO has no critic |
| `logger` | `["console","mlflow"]` | AI Runtime supplies the MLflow context; per-step metrics are in MLflow, not the driver log |
| `project_name` / `experiment_name` | env or `parameters` | MLflow names |
| `nnodes` / `n_gpus_per_node` | the trainer's share of the nodes | async: the nodes left after `ROLLOUT_NNODES` |
| `default_local_dir` | `<output_dir>/<RUN_ID>` | one directory per run |
| `resume_mode` | from `RESUME` (`never` = `disable`) | verl's own default, `auto`, would silently resume from whatever is in the directory |
| `val_before_train` | `VAL_BEFORE_TRAIN` (`False`) | |
| `save_freq` | `SAVE_FREQ` (`-1`) | async counts weight syncs, sync counts steps; with a positive value the final version is also saved |
| `test_freq` | `TEST_FREQ` (`-1`) | in-loop validation is off; evals run as separate jobs |
| `total_epochs` | `hp total_epochs` | |
| `total_training_steps` | `hp total_training_steps` (sync, if not 0) | the step cap |

## Fully-async: `rollout.*` and `async_training.*`

These top-level keys exist only in the fully-async recipe. `rollout.nnodes` and
`rollout.n_gpus_per_node` describe the Rollouter's own pool, separate from `trainer.*`.

| key | value | why |
|---|---|---|
| `rollout.nnodes` / `rollout.n_gpus_per_node` | `ROLLOUT_NNODES` / derived | whole nodes for the Rollouter (or part of one node when `ROLLOUT_NNODES=0` and `N_GPUS_ROLLOUT` is set); the trainer's GPUs stay rollout-free |
| `rollout.total_rollout_steps` | `hp total_rollout_steps` | the run's size in samples |
| `async_training.staleness_threshold` | `STALENESS` (`0.1`) | `0` behaves synchronously; above 0 the Rollouter may use older weights |
| `async_training.trigger_parameter_sync_step` | `TRIGGER_SYNC_STEP` (`2`) | optimizer updates between weight syncs |
| `async_training.require_batches` | `REQUIRE_BATCHES` (`1`) | mini-batches per update |
| `async_training.partial_rollout` | `PARTIAL_ROLLOUT` (`True`) | interrupted rollouts continue after a sync |

Samples per sync are `trigger_parameter_sync_step × require_batches × ppo_mini_batch_size`, and the
number of syncs is `total_rollout_steps` divided by that: `2 × 1 × 16 = 32` samples per sync with
`total_rollout_steps=128` gives 4 syncs.

## `separate_async` (present, not used)

`run_grpo_megatron.sh` still accepts `TRAINER_MODE=separate_async`, verl's v1 disaggregated trainer
(`PARAM_SYNC_STEP`, `ASYNC_WARMUP_BATCHES`, `MAX_OFF_POLICY`, `CKPT_ENGINE_BACKEND`). Do not use it
to separate rollout from training: with `hybrid_engine=False` it places the standalone rollout at
rank 0, on the trainer's GPUs, and vLLM runs out of memory. The fully-async recipe above is the one
that keeps them apart.

## Distributed checkpoints

`USE_DIST_CKPT=True` with `DIST_CKPT_PATH` sets `use_dist_checkpointing` on the actor and the ref.
In verl v0.9.0 this one flag changes two things: checkpoints are saved as sharded Megatron
checkpoints instead of a gathered HF export, and the initial weights are loaded from
`DIST_CKPT_PATH` instead of `model.path`, so a dist checkpoint has to be built from the HF weights
first. None of the shipped jobs use it; [archive/122b-notes.md](archive/122b-notes.md) has
unvalidated notes on building one.

## Reward

Both launchers pass `reward.reward_manager.name` (`REWARD_MANAGER`),
`reward.custom_reward_function.path` and `.name` (`CUSTOM_REWARD_PATH`, `CUSTOM_REWARD_NAME`), and,
when set, `+reward.max_concurrent`, `+reward.max_rpm` and `+reward.timeout`. Without a custom
reward, verl's built-in scorer is chosen by the parquet's `data_source`; for geo3k that is
`verl/utils/reward_score/geo3k.py`:

```
reward = 0.9 × (the boxed answer is correct) + 0.1 × (<think></think> and \boxed{} present)
```

## Process environment

Not Hydra overrides, but they change behaviour:

| var | value | why |
|---|---|---|
| `VLLM_USE_V1` | `1` | vLLM's v1 engine, needed for server mode; both launchers |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `0` | turns off the symmetric-memory all-reduce path (a different path from the custom all-reduce above) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` (classic), unset (fsdp) | see the backend table |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (async) | less allocator fragmentation, so checkpoint save buffers fit |
| `PATH` | `/opt/venv/bin` prepended if `ray` is missing | a job's `command:` can run with a PATH that lacks the venv |
| `OPENSSL_FORCE_FIPS_MODE` / `OPENSSL_FIPS` | `0` / `0` | set in the image's `ENV`: AI Runtime hosts run a FIPS kernel, and non-FIPS crypto in the stack aborts at SSL start-up otherwise ([security.md](security.md)) |
| `NCCL_DEBUG`, `VLLM_LOGGING_LEVEL` | `WARN` | set in the job files; raise to `INFO` when debugging a cross-node sync |
| `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `TOKENIZERS_PARALLELISM`, `RAY_TEMP_DIR` | from the job file | HF cache in `/tmp`, fast downloads, tokenizer threads, Ray scratch |

Never set `RAY_RUNTIME_ENV_HOOK=""`: Ray tries to import the empty string as a class path and dies
with "expected a valid path like mymodule.provider_class".

## Exit status

verl's fully-async exit code is wrong in both directions: a finished run exits non-zero, and a run
whose rollout or reward crashed can exit 0 after saving the version it reached. The launchers
therefore decide from the checkpoints on every exit code, and write the verdict and its reasons to
`<output_dir>/<RUN_ID>/run_result.json`; [running-jobs.md](running-jobs.md) describes the verdict.

The full list of launcher knobs, with defaults, is in [configuration.md](configuration.md).
