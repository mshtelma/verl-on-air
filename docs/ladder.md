# The validation ladder — what each rung proves

← [verl-on-air](../README.md) · [infra/](../infra) · [sizing](sizing.md) · [running-jobs](running-jobs.md)

Before spending on a real training job, prove the topology on cheap runs. Each rung adds
**exactly one** source of risk, so a failure localises to that rung instead of to "the
stack". All four are green on this image, on the small geo3k dataset.

| rung | model | mode | GPUs | offload | risk it retires | wall clock |
|---|---|---|---|---|---|---|
| smoke | — | — | 1×A10 | — | image / CUDA / driver / toolchain | ~2 min |
| 1 | Qwen3.5-2B | fsdp | 8×H100 | 0 | the whole GRPO loop, data, vLLM rollout | ~911 s |
| 2 | Qwen3.5-9B | fsdp | 8×H100 | 0 | co-located rollout memory (vLLM wake vs resident FSDP) | ~1000 s |
| 3 | Qwen3.5-35B-A3B | classic | 8×H100 | 1 | 35B MoE correctness + ZeRO-1 + CPU offload | ~1477 s |
| **4** | **Qwen3.5-35B-A3B** | **fsdp** | **32×H100** | **0** | **offload-free ZeRO-3, multi-node RDMA, co-located weight sync** | ~1443 s |

`make rung1` … `make rung4` (files in `infra/geo3k/air/`).

> **Why a rung prints "2/3 steps" and still succeeds.** The staged geo3k subset is 64 train
> / 8 val examples; at `train_batch_size=32` that is 2 batches = **2 GRPO steps per epoch**,
> and every rung runs `total_epochs=1`. `total_training_steps: 3` is an unreached ceiling —
> set it to `0` for a real run.

## The passing configs

- **rung 1** (2B dense) — `OFFLOAD=0`, `ROLLOUT_GPU_MEM_UTIL=0.6`. Small enough that
  co-location is trivial.
- **rung 2** (9B dense) — `OFFLOAD=0`, **`ROLLOUT_GPU_MEM_UTIL=0.35`**, `GEN_TP=4`. The
  first real co-location fight: util 0.6 and 0.5 both OOM'd on vLLM's `wake_up` KV re-map
  colliding with resident FSDP state.
- **rung 3** (35B MoE, classic) — `OFFLOAD=1`, `OFFLOAD_FRACTION=1`,
  `TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`, `ROLLOUT_GPU_MEM_UTIL=0.6`. Reproduces verl's own
  tested single-node config; first proof the 35B trains at all.
- **rung 4** (35B MoE, Megatron-FSDP) — `TP=1 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`,
  `ROLLOUT_GPU_MEM_UTIL=0.25`, no offload. Peak train `max_reserved` **46.2 GiB**, RDMA
  confirmed with zero TCP fallback, both steps healthy (log-prob correlation ~0.998,
  KL ~0.001, reward mean improving).

> The rung 4 file is named `…16gpu.yaml` but requests **32** — see below. The name is kept
> only because the `Makefile` target points at it.

## Why co-located 35B needs 32 GPUs, not 16

This is the ladder's most useful finding, and it is a *transient*, not steady state.

16-GPU Megatron-FSDP fits the **persistent** state comfortably (the model in
[sizing.md](sizing.md) says ~66 GiB/GPU) and still OOMs — always at the same place: the
`on_step_end` weight resync into the co-located vLLM. At that moment ZeRO-3 must
materialise the **full unsharded MoE expert tensor** (~1.9 GiB, via
`uneven_dtensor_to_full_tensor`) to convert Megatron layout → HuggingFace/vLLM layout, and
that lands on top of vLLM's freshly re-woken weight buffers (~15–17 GiB, *not* the 8.7 GiB
weight shard a persistent budget counts). Real peak ≈ 80 GiB.

Two levers that look obvious and do not work:

- **`ROLLOUT_GPU_MEM_UTIL` cannot fix it.** It sizes only vLLM's KV-cache tag, which is
  *asleep* during the weight sync. Attempts at 0.35 and 0.25 OOM'd within ~40 MiB of each
  other — proof the KV fraction was never the constraint.
- **`enforce_eager` helps but not enough.** Dropping CUDA-graph capture cut vLLM's awake
  footprint ~2 GiB, against a ~1.9 GiB deficit that grew with it. And `WEIGHT_BUCKET_MB`
  can't be shrunk past the embedding: 248320×2048 ≈ 970 MiB per tensor.

CPU offload is not available here either — it crashes on FSDP's DTensors
(`aten.is_pinned`). The only lever that closes the gap is **cutting the training resident**,
i.e. more GPUs: 32-GPU ZeRO-3 halves every shard (~40 → ~22 GiB) and the same transient fits
with room to spare.

So: **the offload-free minimum for *co-located* GRPO on this model is 32 GPUs.** The
fully-async use cases avoid the problem entirely — disjoint rollout/trainer pools mean the
weight sync is an NCCL broadcast between processes rather than a gather competing with a
woken vLLM on the same device ([training-modes.md](training-modes.md)).

## Mechanics worth knowing

**"FSDP" means two different things in verl.** verl's native FSDP backend
(`strategy=fsdp`/`fsdp2`, PyTorch FSDP) is **not used here** — every rung is
`model_engine=megatron`. `MEGATRON_MODE=fsdp` selects *Megatron-FSDP*, a sharded-DP
implementation inside megatron-core that composes with TP/PP/EP by sharding the DP axis.

| `MEGATRON_MODE` | shards optimizer | shards grads | shards params | CPU offload |
|---|---|---|---|---|
| `classic` — distributed optimizer (**ZeRO-1**) | yes | no (replicated) | no (replicated) | works |
| `fsdp` — `optim_grads_params` (**ZeRO-3**) | yes | yes | yes | **incompatible** |

fsdp mode also requires `vanilla_mbridge=False` (the Megatron-**Bridge** provider path),
`gradient_accumulation_fusion=False`, and forbids `use_precision_aware_optimizer` (it
segfaults in TransformerEngine). Full constraint list in [sizing.md](sizing.md) §6.

**How weights reach a co-located vLLM** — the path those OOM tracebacks walk: wake vLLM's
weight tag → stream named tensors in `WEIGHT_BUCKET_MB` buckets → Megatron-Bridge converts
each param from Megatron layout (TP/EP splits, fused QKV, grouped-GEMM experts) to
HuggingFace layout → **for fsdp, all-gather each DTensor to full** (the transient above) →
copy into vLLM's buffers. Classic mode exports from *replicated* params with no DTensor
gather, which is why rung 3 never hit this.

## Pointers

- [sizing.md](sizing.md) — the per-GPU byte budget and topology table (`python3 docs/sizing.py`).
- [troubleshooting.md](troubleshooting.md) — every runtime failure and its fix.
- [running-jobs.md](running-jobs.md) — how to run the rungs and everything else.
