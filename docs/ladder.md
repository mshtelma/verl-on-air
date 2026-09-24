# The validation ladder

Short GRPO runs on geo3k that prove the topology before a real training job, cheapest first.
Each rung changes several settings at once (last column), so a failure points to that set of
changes, and a pass says nothing about the configurations in between. All four pass on the
current image.

| rung | model | mode | GPUs | offload | what it proves | wall clock | changed from the rung before |
|---|---|---|---|---|---|---|---|
| smoke | - | - | 1×A10 | - | image, CUDA, driver, toolchain | ~2 min | - |
| 1 | Qwen3.5-2B | fsdp | 8×H100 | no | the GRPO loop, data, vLLM rollout | ~911 s | first GPU run |
| 2 | Qwen3.5-9B | fsdp | 8×H100 | no | co-located rollout memory (vLLM wake-up vs resident FSDP state) | ~1000 s | model 2B→9B, `GEN_TP` 2→4, util 0.6→0.35, response 1024→2048 |
| 3 | Qwen3.5-35B-A3B | classic | 8×H100 | yes | the 35B MoE with ZeRO-1 and CPU offload | ~1477 s | model →35B MoE, mode fsdp→classic, offload on, `TP` 1→2, `EP` 1→8, `GEN_TP` 4→8, util →0.6 |
| 4 | Qwen3.5-35B-A3B | fsdp | 32×H100 | no | offload-free ZeRO-3, multi-node RDMA, co-located weight sync | ~1443 s | mode classic→fsdp, 8→32 GPUs (4 nodes), offload off, `TP` 2→1, util 0.6→0.25, `enforce_eager` on |

Run them with `make rung1` to `make rung4` (files in `infra/geo3k/air/`). A rung that prints
"2/3 steps" and succeeds is correct: 64 training examples at `train_batch_size=32` is 2 steps per
epoch, every rung runs one epoch, and `total_training_steps: 3` is a cap it never reaches. Set it
to `0` for a real run.

## Passing configurations

- Rung 1 (2B dense): no offload, `ROLLOUT_GPU_MEM_UTIL=0.6`.
- Rung 2 (9B dense): no offload, `ROLLOUT_GPU_MEM_UTIL=0.35`, `GEN_TP=4`. At 0.6 and 0.5, vLLM's
  wake-up collided with the resident FSDP state.
- Rung 3 (35B MoE, classic): `OFFLOAD=1`, `OFFLOAD_FRACTION=1`, `TP=2 EP=8 GEN_TP=8`,
  `ROLLOUT_GPU_MEM_UTIL=0.6`. This is verl's own tested single-node configuration.
- Rung 4 (35B MoE, Megatron-FSDP): `TP=1 EP=8 GEN_TP=8`, `ROLLOUT_GPU_MEM_UTIL=0.25`, no offload.
  Peak reserved memory 46.2 GiB, RDMA with no TCP fallback, both steps healthy (log-prob
  correlation about 0.998, KL about 0.001, reward rising).

## Why co-located 35B needs 32 GPUs

On 16 GPUs, Megatron-FSDP fits the persistent state (about 63 GiB per GPU, see
[sizing.md](sizing.md)) but runs out of memory every time in the end-of-step weight sync into
the co-located vLLM: to convert Megatron's layout to vLLM's, ZeRO-3 gathers each expert tensor in
full (about 1.9 GiB) on top of vLLM's freshly woken weight buffers (15 to 17 GiB, not the 8 GiB
shard a persistent budget counts), a peak of about 80 GiB.

`ROLLOUT_GPU_MEM_UTIL` can't help, since it sizes only the KV cache, which is asleep during the
sync (runs at 0.35 and 0.25 failed within 40 MiB of each other). `enforce_eager` saves about
2 GiB, `WEIGHT_BUCKET_MB` can't go below the 970 MiB embedding, and CPU offload crashes on FSDP's
DTensors. 32 GPUs halve each training shard (about 37 to 18 GiB) and the transient fits. That is
the smallest validated offload-free configuration for co-located GRPO on this model and workload
(geo3k, response 2048, `rollout_n=5`); nothing between 16 and 32 was tried. The fully-async use
cases avoid the problem, since their weight sync is an NCCL broadcast between separate GPU pools
([training-modes.md](training-modes.md)).

## FSDP here means Megatron-FSDP

verl's PyTorch FSDP backend (`strategy=fsdp`/`fsdp2`) is not used. Every rung runs
`model_engine=megatron`, and `MEGATRON_MODE=fsdp` selects Megatron-FSDP, megatron-core's sharded
data parallelism, which composes with TP, PP and EP. Classic (ZeRO-1) shards only the optimizer
and works with CPU offload; fsdp (ZeRO-3) shards optimizer, grads and params and doesn't support
offload. fsdp mode also needs `vanilla_mbridge=False` and `gradient_accumulation_fusion=False`,
and rules out `use_precision_aware_optimizer` (full list in [sizing.md](sizing.md)).

During the weight sync, Megatron-Bridge converts each tensor to Hugging Face layout, and in fsdp
mode each DTensor is gathered in full first. Classic exports from replicated params with no
gather, which is why rung 3 never hit the transient.
