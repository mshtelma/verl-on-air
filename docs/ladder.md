# The validation ladder — what we proved, and how

This project validates GRPO reinforcement learning on the Qwen3.5 family with
verl's Megatron/mcore backend on Databricks AI Runtime serverless GPU (workspace
`df1`, AWS/P5). The strategy is a **ladder**: each rung adds exactly one source
of risk, so a failure localises to that rung instead of the whole stack.

**Status: every rung is green.** The headline (rung 4 — offload-free 35B-A3B on
Megatron-FSDP across nodes) landed on 32×H100 / 4 nodes.

| rung | model | mode | GPUs | offload | key risk it retires | verdict |
|---|---|---|---|---|---|---|
| smoke | — | — | 1×A10 | — | image / CUDA / nvcc toolchain | ✅ |
| 1 | Qwen3.5-2B | fsdp | 8×H100 | 0 | end-to-end GRPO loop, geo3k data, vLLM rollout | ✅ 911s |
| 2 | Qwen3.5-9B | fsdp | 8×H100 | 0 | co-located rollout OOM (vLLM wake vs resident FSDP) | ✅ 1000s |
| 3 | Qwen3.5-35B-A3B | classic | 8×H100 | 1 | 35B MoE correctness + classic ZeRO-1 + CPU offload | ✅ 1477s |
| **4** | **Qwen3.5-35B-A3B** | **fsdp** | **32×H100 (4 nodes)** | **0** | **offload-free ZeRO-3, EFA multi-node, co-located weight sync** | **✅ 1443s** |

Reproduce any rung: `make rung1` … `make rung4` (targets in the `Makefile`, pointing at
`infra/geo3k/air/rung*.yaml`). All runs use the geo3k subset.

> **Why every run shows "2/3 steps" and still succeeds.** The staged geo3k
> subset is 64 train / 8 val examples. With `train_batch_size=32` that is 2
> batches = **2 GRPO steps per epoch**, and every rung runs `total_epochs=1`, so
> each run completes exactly 2 steps. `total_training_steps=3` is a ceiling the
> single epoch never reaches — the tqdm bar reads "2/3" but the job exits
> SUCCESS. **There is no df1 wall-clock cap** (rung3 ran 1477s, rung4 1443s).

---

## Per-rung passing configs

**rung 1 — Qwen3.5-2B, fsdp, 8×H100** (`infra/geo3k/air/rung1_2b_fsdp_8gpu.yaml`)
Proves the loop end to end. Small enough that co-location is trivial:
`OFFLOAD=0`, `ROLLOUT_GPU_MEM_UTIL=0.6`.

**rung 2 — Qwen3.5-9B, fsdp, 8×H100** (`infra/geo3k/air/rung2_9b_fsdp_8gpu.yaml`)
First real co-location fight. `OFFLOAD=0`, **`ROLLOUT_GPU_MEM_UTIL=0.35`**,
`GEN_TP=4`. util 0.6 and 0.5 both OOM'd on vLLM's `wake_up` KV re-map colliding
with resident FSDP state — 0.35 is the fit. pearson 0.9993/0.9994, kl ~4e-4.
Measured vLLM map (9B, GEN_TP=4, util 0.35): KV 20.69 GiB, awake footprint
~26 GiB, resident training ~47 GiB.

**rung 3 — Qwen3.5-35B-A3B, classic, 8×H100** (`infra/geo3k/air/rung3_35b_classic_8gpu.yaml`)
35B MoE on classic Megatron (ZeRO-1) + full CPU offload:
`OFFLOAD=1`, `OFFLOAD_FRACTION=1`, `TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`,
`ROLLOUT_GPU_MEM_UTIL=0.6`. This rung forced the **mbridge version-skew fix**
(see below) and was the first proof the 35B trains at all.

**rung 4 — Qwen3.5-35B-A3B, fsdp, 32×H100 / 4 nodes** (`infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml`)
The headline. Offload-free ZeRO-3, `TP=1 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`,
`ROLLOUT_GPU_MEM_UTIL=0.25`. Result: SUCCESS 1443s, 0 OOM, both GRPO steps
healthy — pearson **0.9978/0.9980**, kl **~0.001**, reward mean **0.174→0.225**
(improving), grad_norm 0.14/0.16. Peak train `max_reserved` **46.2 GiB** — within
1 GiB of the sizing model's "roomy 46.3". EFA confirmed: 2502 GDRDMA/Libfabric
lines, **0 TCP fallback**, 32 ranks / 4 nodes. vLLM map (util 0.25, eager): KV
7.66 GiB = 795k tokens, 3.03× concurrency, sleep frees 16.02 GiB.

> The file is still named `…16gpu.yaml`; the content is now 32 GPU. Rename
> deferred to avoid churning the `Makefile` rung4 target.

---

## rung 4 debugging saga (3 attempts to green)

rung 4 is the interesting one — it took three failed 16-GPU attempts to learn
that 16 GPUs **cannot** host this run co-located, and the fix was the user's
"scale GPUs, not offload" strategy.

| attempt | GPUs | change | result |
|---|---|---|---|
| 1 | 16 | util 0.35 | OOM at on_step_end weight sync — 61.7 train + 17.35 vLLM → 79.1/79.2 |
| 2 | 16 | util **0.25** | OOM at the same spot, within ~40 MiB — **util was the wrong lever** |
| 3 | 16 | **enforce_eager** | OOM still — eager cut vLLM 17.35→15.22 (~2 GiB), but the failing alloc grew to **1.89 GiB** and we were short ~1.9 GiB with no cheap lever left |
| **4** | **32** | **double the GPUs** | **✅ SUCCESS** — training resident halved (~40→~22 GiB); the 1.89 GiB gather fits with ~33 GiB to spare |

**The diagnosis.** Each PPO step, after the actor update, verl resyncs the fresh
weights into the co-located vLLM (`on_step_end` → `checkpoint_manager.update_weights`).
The OOM was always at that resync, in
`megatron_fsdp/uneven_dtensor.py:uneven_dtensor_to_full_tensor` → `torch.zeros(dtensor.shape)`
— the **full unsharded MoE expert tensor** (1.89 GiB) that ZeRO-3 must materialise
to export to HuggingFace/vLLM layout. It collided with vLLM's freshly re-woken
`['weights']` buffers (~15–17 GiB).

Two false leads, both instructive:
- **`ROLLOUT_GPU_MEM_UTIL` cannot fix it.** util sizes only vLLM's `['kv_cache']`
  tag, which is *asleep* during the weight sync. Attempts 1 and 2 (0.35, 0.25)
  OOM'd within 40 MiB of each other — proof the KV fraction was never the constraint.
- **`enforce_eager` helps but isn't enough.** It removes CUDA-graph capture,
  cutting vLLM's *awake* footprint by ~2 GiB. Real but far short of the ~1.9 GiB
  deficit; and `WEIGHT_BUCKET_MB=256` (the other "obvious" lever) would have
  aborted with `tensor too large to fit in the bucket` — the embedding is
  248320×2048 ≈ 970 MiB, larger than that bucket.

The only lever that closes it is cutting the **training resident** so the fixed
1.89 GiB gather + ~15 GiB awake vLLM fit under 80. Offload can't do that on FSDP
(it crashes on DTensors — `aten.is_pinned`), so the answer is **more GPUs**:
32-GPU ZeRO-3 halves every shard, dropping resident ~40→~22 GiB.

---

## Mechanics: how the pieces actually work

**verl has two training backends, and "FSDP" means a different thing in each.**

- **verl's native FSDP backend** (`strategy=fsdp`/`fsdp2`) — PyTorch FSDP/FSDP2.
  **This repo does not use it** (every rung is `model_engine=megatron`).
- **The Megatron backend's FSDP** — *Megatron-FSDP*, a ZeRO-2/3 sharded-DP built
  inside Megatron-core (`megatron/core/distributed/fsdp/…`), distinct from PyTorch
  FSDP. It **composes with** Megatron's TP/PP/EP by sharding along the DP axis.

**Classic vs fsdp mode** (`MEGATRON_MODE`, `engine/train/run_grpo_megatron.sh:258`):

| | shards optim | shards grads | shards params | offload |
|---|---|---|---|---|
| `classic` — distributed optimizer (**ZeRO-1**) | over DP | no (replicated) | no (replicated) | CPU offload works |
| `fsdp` — Megatron-FSDP `optim_grads_params` (**ZeRO-3**) | yes | yes | yes | **incompatible** (`aten.is_pinned` DTensor crash) |

The `data_parallel_sharding_strategy` enum is the ZeRO ladder: `no_shard` (0) /
`optim` (1, ≈ classic) / `optim_grads` (2) / `optim_grads_params` (3, rung4).
fsdp mode also requires `vanilla_mbridge=False` (Megatron-**Bridge** provider path),
`gradient_accumulation_fusion=False`, and forbids `use_precision_aware_optimizer`
(TE segfault). Full constraint list in `sizing.md §6`.

**How weights reach vLLM** (co-located, the path the OOM tracebacks walk):

1. `checkpoint_engine/base.py update_weights` wakes vLLM's `['weights']` tag.
2. `vllm_rollout.py update_weights` → `bucketed_weight_transfer.async_send_weights`
   streams named tensors in `WEIGHT_BUCKET_MB`-sized buckets.
3. **Megatron-Bridge** `model_bridge.stream_weights_megatron_to_hf` converts each
   param from Megatron layout (TP/EP splits, fused QKV, grouped-GEMM experts) to
   HuggingFace layout — the "Converting to HuggingFace … (N/3356)" bar.
4. Because fsdp params are DTensors, each is **all-gathered to full** via
   `uneven_dtensor_to_full_tensor` — the transient that OOM'd at 16 GPU.
5. Each bucket is copied into vLLM's weight buffers (device-to-device, same GPU).

Classic mode (`vanilla_mbridge=True`) exports from *replicated* params — no
DTensor full-gather — which is partly why rung3 never hit this OOM. A
**disaggregated** rollout (vLLM on separate GPUs) would replace step 5 with an
NCCL broadcast and sidestep the co-location headroom problem entirely.

---

## Measured vs the sizing model

The persistent-state model in `sizing.py` predicts 32-fsdp at **46.3 GiB** and we
measured **46.2 GiB** reserved — an excellent match. But the model's *persistent*
verdict called 16-fsdp "OK" (65.9 GiB), and 16-fsdp **OOMs** — because the
persistent budget omits the **co-located weight-sync peak**: the ~1.9 GiB
full-tensor gather on the training side *plus* vLLM awake at ~15–17 GiB (not the
8.7 GiB weight shard the table counts). Real 16-fsdp peak ≈ 80 GiB. The honest
**offload-free minimum for co-located GRPO on this model is 32 GPUs**, not 16.
`sizing.md`/`sizing.py` now carry this correction and the measured anchors.

---

## The mbridge version-skew fix (rung 3, baked into the image)

rung3 first crashed with
`Qwen3_5VLTransformerConfig.__init__() got an unexpected keyword argument 'async_tensor_model_parallel_allreduce'`.
Cause: legacy `mbridge` 0.15.1's `qwen3_5_vl_bridge.py` hard-codes that kwarg, but
megatron-core `core_v0.18.0`'s `TransformerConfig` dropped it. It hits the
classic/`vanilla_mbridge=True` path only (the Bridge provider path re-declares it,
so rung1/2 were immune). Fixed in `docker/Dockerfile` with a self-verifying `sed`
patch (step 5a1) plus an AST regression gate in the step-9 verification that fails
the build if any Qwen3.5 bridge forwards a kwarg absent from the config's fields.
Image tag bumped to **v5**.

---

## Pointers

- `docs/sizing.md` / `docs/sizing.py` — the memory arithmetic and topology table.
- `docs/troubleshooting.md` — every runtime failure and its fix (mbridge skew,
  TE segfault, weight-sync OOM, EFA vs TCP fallback, bucket-too-small).
- `docs/setup.md`, `docs/build-linux.md` — environment and image build.
