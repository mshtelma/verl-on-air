# Sizing Qwen3.5-35B-A3B GRPO

`python3 docs/sizing.py` reproduces every number here, and `tests/test_sizing.py` keeps the two
in step. Memory is in GiB, as torch and verl report it; one H100 has 81,559 MiB = 79.65 GiB
usable. Numbers are analytic (arithmetic on `config.json` and the stated assumptions), measured
(logged by a real job), or hypotheses (predicted for a configuration nobody ran).

## Where the parameters are

`Qwen/Qwen3.5-35B-A3B`: 40 layers (10 full attention, 30 Gated-DeltaNet), hidden size 2048, 256
experts with top-8 routing, a 248,320-token vocabulary with untied embeddings, a vision tower.

| component | params | share |
|---|---|---|
| routed experts | 32.21 B | 92.5% |
| embeddings + lm_head | 1.02 B | 2.9% |
| Gated-DeltaNet layers | 0.76 B | 2.2% |
| vision tower | 0.41 B | 1.2% |
| full attention + shared expert + router | 0.42 B | 1.2% |
| total | 34.82 B | |

Only the routed experts shard by EP. The shared expert (0.13 B) is a dense TP-sharded MLP in
Megatron's layout, replicated across EP ranks like attention. So expert parallelism is the main
memory lever, and the size of the expert data-parallel group decides whether FSDP helps.

## Unsharded state

GRPO has no critic, so the unsharded state is:

```
params bf16                  64.8 GiB
grads  bf16                  64.8 GiB
Adam fp32 (m, v, master)    389.1 GiB
ref policy bf16              64.8 GiB
vLLM rollout weights bf16    64.8 GiB
────────────────────────────────────
                            648.5 GiB   vs 8 × H100 ≈ 637 GiB usable
```

That is over budget before any activation, so the question is where the optimizer state goes.

## ZeRO-1 or ZeRO-3

`MEGATRON_MODE=classic` (the distributed optimizer, ZeRO-1) shards only the optimizer state over
DP. Params and grads are replicated and stay at `(experts/EP + non_expert/TP) × 2 bytes` however
many nodes you add. `MEGATRON_MODE=fsdp` (Megatron-FSDP `optim_grads_params`, ZeRO-3) shards all
three. Analytic per-GPU persistent memory in GiB, with no offload, Adam at 12 B/param and
`GEN_TP=8` (vLLM weights shard by `GEN_TP`, not by the GPU count):

| GPUs | mode | params | grads | Adam | ref | vLLM | sum | + overhead | analytic verdict |
|---|---|---|---|---|---|---|---|---|---|
| 8 | classic | 9.9 | 9.9 | 48.6 | 9.9 | 8.1 | 86.5 | 104.5 | OOM |
| 8 | fsdp | 8.1 | 8.1 | 48.6 | 8.1 | 8.1 | 81.1 | 99.1 | OOM |
| 16 | classic | 9.9 | 9.9 | 24.3 | 9.9 | 8.1 | 62.2 | 80.2 | OOM (by 0.5 GiB) |
| 16 | fsdp | 4.1 | 4.1 | 24.3 | 4.1 | 8.1 | 44.6 | 62.6 | fits, but OOMed at the weight sync (below) |
| 32 | classic | 9.9 | 9.9 | 12.2 | 9.9 | 8.1 | 50.0 | 68.0 | tight |
| 32 | fsdp | 2.0 | 2.0 | 12.2 | 2.0 | 8.1 | 26.3 | 44.3 | fits, validated |

Gradients are counted as bf16 (`--grad-bytes 4` shows the fp32 bound). The overhead is a fixed
18 GiB guess (CUDA context 2, chunked logits and activations 6, FSDP transient 4, weight-sync
bucket 6) that doesn't scale with sequence length, micro-batch or `rollout_n`, so a verdict near
the budget is only as good as this constant.

## The weight-sync peak (measured)

The table is steady state. Co-located GRPO peaks higher during the actor-to-vLLM weight sync at
the end of each step: vLLM is awake with 15 to 17 GiB (weights plus buffers, not the 8 GiB shard)
while ZeRO-3 gathers each parameter into a full tensor (up to 1.9 GiB for the largest expert).
Logged on the rung 4 workload (Megatron-FSDP, `TP=1 EP=8 GEN_TP=8`, `ROLLOUT_GPU_MEM_UTIL=0.25`,
no offload, 64 geo3k examples, `train_batch_size=32`, `rollout_n=5`, prompt 1024, response 2048),
with the train peak taken from verl's `max_memory_reserved_gb`:

| GPUs | train peak | vLLM awake | total | vs 79.65 GiB | outcome |
|---|---|---|---|---|---|
| 16 | 63.4 | 17.0 | 80.4 | over | OOM at the weight sync |
| 32 | 46.2 | 15.2 | 61.4 | under | success |

The 32-GPU analytic estimate (44.3 GiB) is about 2 GiB below the measured 46.2 GiB: a
plausibility check, not a validation. Lowering `ROLLOUT_GPU_MEM_UTIL` doesn't fix the 16-GPU OOM
(it sizes only the KV cache, which is asleep during the sync), `enforce_eager` saves only about
2 GiB, `WEIGHT_BUCKET_MB` can't go below the 970 MiB embedding, and CPU offload doesn't work with
FSDP. More GPUs did.

So the smallest validated offload-free co-located configuration is 32 GPUs with Megatron-FSDP,
on this workload. Nothing between 16 and 32 GPUs, and no other sequence budget or micro-batch,
was tested. Classic without offload is a hypothesis only: 16 GPUs come out just over budget at
80.2 GiB (72.1 with the 8 B/param precision-aware optimizer) and 32 GPUs tight at 68.0. None of
these ran; the validated classic configuration is 8 GPUs with full CPU offload (rung 3).

## 8 GPUs with full offload

verl's own `run_qwen3_5_35b_megatron.sh` runs this model on one node (`TP=2 PP=1 CP=1 EP=8 ETP=1
GEN_TP=8`, everything offloaded), and rung 3 reproduces it. verl's perf table for the similar
Qwen3-30B-A3B at EP=8 with full offload shows MFU falling from 0.40 on 8 GPUs to 0.37 on 16 and
0.31 on 32: extra nodes add cross-node traffic without shrinking the per-rank optimizer state.
Scaling out only pays if offload goes off too, which is what rung 4 does.

The cost of 8 GPUs lands in host memory (analytic):

```
expert Adam state, EP=8, expert-DP=1   360 GiB
non-expert Adam state                   29 GiB
offloaded actor params                  79 GiB
offloaded ref params                    79 GiB
                                      ~548 GiB per node
```

The smoke test prints the node's `MemTotal`; below about 550 GiB, 8-GPU classic won't fit.

## Expert data parallelism

`expert-DP = GPUs / (EP × ETP × PP)`. At 8 GPUs with EP=8 it is 1, and FSDP has nothing to shard
the experts over (EP=4 gives 2 but doubles each rank's expert params); at 16 GPUs it is 2, at 32
it is 4. A measurement reported to Megatron-LM found EP8 with FSDP at expert-DP 1 both slower and
hungrier than EP8 alone, so check that expert-DP is above 1 before enabling Megatron-FSDP on a
MoE model.

## Settings required for correctness

| setting | why |
|---|---|
| `use_remove_padding=False` on `model.` and `actor.megatron.` | Gated-DeltaNet has no packed-sequence (THD) support in Megatron-LM, so everything runs BSHD |
| `use_dynamic_bsz=False` (actor, ref log-prob, rollout log-prob) | required by BSHD |
| `CUDA_DEVICE_MAX_CONNECTIONS` unset in fsdp mode | `1`, right for classic, serialises the FSDP collectives behind compute |
| `gradient_accumulation_fusion=False` in fsdp mode | not supported by Megatron-FSDP |
| `vanilla_mbridge=False` in fsdp mode | verl passes `use_megatron_fsdp` only through the Megatron-Bridge path; legacy mbridge ignores it |
| `use_precision_aware_optimizer` in classic mode only | under Megatron-FSDP it segfaults in TransformerEngine's `multi_tensor_scale` (the grad-clip path) |
| `entropy_from_logits_with_chunking=True` | unchunked logits and entropy over 248,320 tokens take about 3 GB per micro-batch |

## Larger models

Rung 4 shards only along EP and FSDP's data-parallel dimension (`TP=PP=CP=1`); a larger model
needs the other axes too, which the launchers already take as env vars. A rough per-GPU budget
for `P` billion parameters (hypothesis; the transient is one data point, 46.2 minus 18.2 GiB):

```
persistent training state (ZeRO-3) = P × 18 bytes / GPUs     ≈ 16.8·P / GPUs GiB
vLLM rollout weights               = P ×  2 bytes / GEN_TP   ≈ 1.86·P / GEN_TP GiB
co-located sync peak               ≈ persistent + ~28 GiB transient + vLLM awake
```

vLLM weights shard by `GEN_TP` only, so they don't shrink as you add nodes:

| model size | vLLM weights at `GEN_TP=8` | at 16 | at 32 |
|---|---|---|---|
| 35 B | 8.1 GiB | 4.1 | 2.0 |
| 72 B dense | 16.7 GiB | 8.4 | 4.2 |
| 235 B MoE | 54.6 GiB | 27.3 | 13.7 |

At 235 B, co-location needs cross-node rollout TP (`GEN_TP≥16`) or a disaggregated rollout,
where vLLM has its own GPUs and the weight sync is an NCCL broadcast rather than the ZeRO-3
gather. Disaggregation is the safer bet.
