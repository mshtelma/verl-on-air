# Sizing: how many H100s does Qwen3.5-35B-A3B GRPO actually need?

← [verl-on-air](../README.md) · [training-modes](training-modes.md) · [ladder](ladder.md) · [tuning](tuning.md)

Reproduce the arithmetic with `python3 docs/sizing.py` (checked by `tests/test_sizing.py`).
It computes in **bytes** and shows **GiB** (2³⁰) throughout — the unit torch and verl report
memory in; one H100 has **81,559 MiB = 79.65 GiB** usable. Every number below is one of:

- **analytic** — arithmetic on the model's `config.json` and the assumptions stated next to it;
- **measured** — logged by a real job (named, with its workload and image); the script only
  restates it;
- **hypothesis** — what the analytic model predicts for a configuration nobody ran.

## 1. Where the parameters actually are

`Qwen/Qwen3.5-35B-A3B` → `Qwen3_5MoeForConditionalGeneration`, `qwen3_5_moe`:

| field | value |
|---|---|
| layers | 40 |
| hidden | 2048 |
| experts | 256, top-8 |
| `moe_intermediate_size` | 512 |
| attention | 16 Q / 2 KV heads, `head_dim` 256 |
| layer mix | `full_attention_interval: 4` → 10 full-attn + **30 Gated-DeltaNet** |
| vocab | 248320, `tie_word_embeddings: false` |
| vision | 27 layers, hidden 1152 |
| max ctx | 262144 |

Recomputed parameter budget:

| component | params | share |
|---|---|---|
| routed experts | 32.21 B | **92.5 %** |
| GDN linear-attn ×30 | 0.76 B | 2.2 % |
| embed + lm_head | 1.02 B | 2.9 % |
| vision tower | 0.41 B | 1.2 % |
| full-attn ×10 + shared expert + router | 0.42 B | 1.2 % |
| **total** | **34.82 B** | |

(analytic) The shared expert (0.13 B) is **not** expert-parallel: in the Megatron layout verl
converts, it is a dense TP-sharded MLP (`mlp.shared_experts.linear_fc1/fc2`, not under
`mlp.experts`), so it is replicated across EP ranks like attention. Only the routed experts are
sharded by EP.

**Everything follows from that first row.** 92.5 % of the weights are routed
experts, so expert parallelism is the dominant memory lever and the geometry of
the expert-DP group decides whether FSDP helps or hurts.

## 2. Naive requirement

Unsharded GRPO state (no critic — GRPO uses a group-relative baseline):

```
params bf16                  64.8 GiB
grads  bf16                  64.8 GiB
Adam fp32 (m, v, master)    389.1 GiB
ref policy bf16              64.8 GiB
vLLM rollout weights bf16    64.8 GiB
────────────────────────────────────
                            648.5 GiB   vs 8xH100 ≈ 637 GiB usable
```

Over budget before a single activation. So the question is never "does it fit",
it is "where does the optimizer state go".

## 3. The ZeRO-1 vs ZeRO-3 distinction (the crux)

| | shards optimizer | shards grads | shards params |
|---|---|---|---|
| `MEGATRON_MODE=classic` — distributed optimizer (**ZeRO-1**) | yes, over DP | **no** | **no** |
| `MEGATRON_MODE=fsdp` — Megatron-FSDP `optim_grads_params` (**ZeRO-3**) | yes | yes | yes |

Classic replicates params+grads across DP, so those two terms are stuck at
`(experts/EP + non_expert/TP) × 2 bytes` no matter how many nodes you add.

**Analytic.** Per-GPU persistent HBM in GiB, no offload, Adam 12 B/param, `GEN_TP=8`
(vLLM weights shard by `GEN_TP`, **not** by N — they do not shrink with nodes):

| N | mode | params | grads | Adam | ref | vLLM | sum | +overhead | analytic verdict |
|---|---|---|---|---|---|---|---|---|---|
| 8 | classic | 9.9 | 9.9 | 48.6 | 9.9 | 8.1 | 86.5 | 104.5 | OOM |
| 8 | fsdp | 8.1 | 8.1 | 48.6 | 8.1 | 8.1 | 81.1 | 99.1 | OOM |
| 16 | classic | 9.9 | 9.9 | 24.3 | 9.9 | 8.1 | 62.2 | 80.2 | OOM (by 0.5 GiB) |
| 16 | fsdp | 4.1 | 4.1 | 24.3 | 4.1 | 8.1 | 44.6 | 62.6 | fits — but **measured OOM at the sync**, below |
| 32 | classic | 9.9 | 9.9 | 12.2 | 9.9 | 8.1 | 50.0 | 68.0 | tight |
| **32** | **fsdp** | **2.0** | **2.0** | **12.2** | **2.0** | **8.1** | **26.3** | **44.3** | **fits — validated** |

The assumptions, all adjustable in the script:

- **Precision.** params, ref and vLLM weights bf16 (2 B); Adam = fp32 m, v and master (12 B);
  **gradients bf16 (2 B)**. Megatron keeps fp32 main grads when gradients are accumulated or
  reduced in fp32; `python3 docs/sizing.py --grad-bytes 4` shows that bound (+2 B/param).
- **Overhead = 18 GiB, a fixed guess**: CUDA context 2 + chunked logits/activations 6 + FSDP
  transient 4 + weight-sync bucket 6. It does not scale with sequence length, micro-batch or
  `rollout_n`, so every verdict near the budget is only as good as this constant.
- **Placement.** Routed experts shard by EP×ETP (and, in classic, their Adam state by
  expert-DP); everything else, the shared expert included, by TP (and DP for classic Adam).

**Measured.** This table is per-GPU steady state. Co-located GRPO peaks higher during the
on_step_end actor→vLLM weight sync, which the persistent budget misses: vLLM is *awake*
holding ~15–17 GiB (weights + buffers, not the ~8 GiB shard) while ZeRO-3 all-gathers each
param to a full unsharded tensor (~1.9 GiB for the largest MoE expert). Logged on the rung-4
workload — Megatron-FSDP `TP=1 EP=8 GEN_TP=8`, `ROLLOUT_GPU_MEM_UTIL=0.25`, no offload, geo3k
64-example subset, `train_batch_size=32`, `rollout_n=5`, prompt 1024 / response 2048, image
v5, df1, 2026-09-09 (train peak = verl's `max_memory_reserved_gb`, which is GiB):

| N | train peak (reserved) | vLLM awake | total | vs 79.65 GiB | outcome |
|---|---|---|---|---|---|
| 16 | 63.4 GiB | 17.0 | 80.4 | over | **OOM** at the weight sync |
| 32 | **46.2 GiB** | 15.2 | 61.4 | under | **SUCCESS** |

The 32-fsdp analytic estimate (44.3 GiB) is ~2 GiB below the measured reserved peak
(46.2 GiB) — the same unit now, but with a fixed overhead guess that is a plausibility
check, not a validation of the model. `util` cannot fix the 16-GPU OOM (it sizes only the KV
tag, asleep during the sync); neither can `enforce_eager` alone (~2 GiB) nor a small
`WEIGHT_BUCKET_MB` (the embedding is 970 MiB > the bucket). Only thinning the resident via
more GPUs did — offload is unavailable on FSDP.

### Conclusions

- **The smallest validated offload-free co-located configuration is 32 GPUs** (Megatron-FSDP,
  the workload above). 16 GPUs was measured to OOM at the weight sync although its persistent
  state fits (62.6 GiB analytic). Nothing between 16 and 32 GPUs, no other sequence budget
  and no other micro-batch was tested, so this is not a universal minimum.
- **Classic (ZeRO-1), hypothesis only**: at Adam 12 B/param the analytic model puts 16 GPUs
  just over the budget (80.2 GiB — within the uncertainty of the overhead guess) and 32 GPUs
  tight (68.0); with the precision-aware optimizer (8 B/param) 16 GPUs fits analytically
  (72.1, tight). None of these was run. The validated classic configuration is 8 GPUs **with**
  full CPU offload (rung 3, §4).
- `use_precision_aware_optimizer=True` is **classic-only** — it segfaults under Megatron-FSDP
  inside TransformerEngine's `multi_tensor_scale` (grad-clip path).

## 4. Why 8 GPUs still "works" — and what it costs

verl's own `run_qwen3_5_35b_megatron.sh` header claims 8 GPUs / 1 node,
`TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`, `ALL_OFFLOAD=True`. That is real, and it
is what `infra/geo3k/air/rung3_35b_classic_8gpu.yaml` reproduces. Corroborated by verl's perf
table for the near-identical Qwen3-30B-A3B (30 B total / 3 B active):

| GPUs | nodes | TP | PP | EP | offload_fraction | offload_optim | MFU |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 1 | 8 | 1.0 | True | **0.40** |
| 16 | 2 | 1 | 1 | 8 | 1.0 | True | 0.37 |
| 32 | 4 | 1 | 1 | 8 | 1.0 | True | 0.31 |

Two things to read off that table:

1. 8 GPUs is genuinely sufficient **with full CPU offload**.
2. MFU *decreases* with node count. With EP pinned at 8 and offload left on,
   extra nodes add cross-node traffic without shrinking per-rank optimizer
   state. Scaling out only pays off if you scale out *and* turn offload off —
   which is exactly rung 4.

The bill for 8 GPUs lands on the host, not the GPU:

```
expert Adam state, EP=8, expert-DP=1  ->   360 GiB  in CPU RAM
non-expert Adam state                 ->    29 GiB
offloaded actor params                ->    79 GiB
offloaded ref params                  ->    79 GiB
                                          ~548 GiB per node   (analytic)
```

`infra/diagnostics/air/smoke_test.yaml` prints the node's actual `MemTotal` for this reason. If
it is under ~550 GiB, 8-GPU classic is not viable and you must go to rung 4.
(On AWS, `GPU_8xH100` is P5-class — `p5.48xlarge` carries ~2 TiB — so this is
expected to pass. But Databricks does not document the node shape behind
`GPU_8xH100`, so it is an assumption rather than a guarantee. That is precisely
why the smoke test measures `MemTotal` instead of trusting the SKU.)

## 5. Why EP=8 specifically at 16 GPUs

`expert-DP = world / (EP × ETP × PP)`.

| N | EP | ETP | expert-DP | consequence |
|---|---|---|---|---|
| 8 | 8 | 1 | **1** | FSDP has no DP dim for experts → pure overhead |
| 8 | 4 | 1 | 2 | shards, but per-rank expert params double |
| **16** | **8** | **1** | **2** | FSDP shards experts; the config we ship |
| 32 | 8 | 1 | 4 | comfortable; offload off, longer sequences fine |

The expert-DP=1 case is not theoretical: NVIDIA/Megatron-LM issue #2772 measured
Qwen3-30B-A3B with EP8+FSDP at **55.6 GB reserved / 111 TFLOPS** versus EP8
alone at **45.5 GB / 138 TFLOPS** (as that issue reports them) — FSDP was both slower *and* hungrier. Never
enable Megatron-FSDP for a MoE model without checking that expert-DP > 1.

## 6. Non-negotiable correctness constraints

Not tuning — the run is wrong or dead without these:

| constraint | why |
|---|---|
| `use_remove_padding=False` (on `model.` **and** `actor.megatron.`) | Qwen3.5's Gated-DeltaNet has no THD/packed-sequence support in Megatron-LM; everything must run BSHD |
| `use_dynamic_bsz=False` (actor, ref log-prob, rollout log-prob) | required by the BSHD path |
| `CUDA_DEVICE_MAX_CONNECTIONS` **unset** in fsdp mode | `=1` (correct for classic) serialises FSDP collectives behind compute and destroys overlap |
| `gradient_accumulation_fusion=False` in fsdp mode | incompatible with Megatron-FSDP |
| `vanilla_mbridge=False` in fsdp mode | verl only threads `use_megatron_fsdp` through the Megatron-Bridge provider path; legacy mbridge silently ignores it |
| `entropy_from_logits_with_chunking=True` | vocab 248320 → un-chunked logits+entropy is ~3 GB per micro-batch |

## 7. Scaling to a bigger model (fully distributed)

rung 4 exercised **one** parallelism axis for sharding (EP + Megatron-FSDP on
the DP dimension) with `TP=PP=CP=1`. A materially bigger model needs the axes we
left at 1 — that is what "fully distributed" means here. The launcher already
threads `TP`, `PP`, `CP`, `EP`, `ETP`, `GEN_TP` as env vars, so this is
configuration, not new code.

**Parametric budget** (total params `P` in billions, bf16). Per GPU:

```
persistent train (ZeRO-3)  = P × 18 bytes / N          ≈ 16.8·P / N   GiB
vLLM rollout weights       = P ×  2 bytes / GEN_TP      ≈  1.86·P / GEN_TP GiB   ← shards by GEN_TP ONLY, not N
co-located sync peak       ≈ train_peak + vLLM_awake    (train_peak ≈ persistent + ~28 GiB transient)
```

For the 35B (P = 34.8) that gives persistent/32 ≈ 18.2 GiB and vLLM/GEN_TP 8 ≈ 8.1 GiB
(analytic); the ~28 GiB transient is simply measured peak minus persistent at 32 GPUs
(46.2 − 18.2), one data point — so the formula for a bigger model is a **hypothesis** to
check with a smoke run, not a prediction.

**The dominant scaling wall is the vLLM term.** Because rollout weights shard by
`GEN_TP` (intra-node = 8) and **not** by the world size, a big model's vLLM
footprint does not shrink as you add nodes:

| P (total) | vLLM wt @ GEN_TP=8 | @ GEN_TP=16 | @ GEN_TP=32 |
|---|---|---|---|
| 35 B | 8.1 GiB | 4.1 | 2.0 |
| 72 B (dense) | 16.7 GiB | 8.4 | 4.2 |
| 235 B (MoE) | 54.6 GiB | 27.3 | 13.7 |

At 235 B, `GEN_TP=8` alone eats ~55 GiB — co-location is essentially impossible
without **cross-node rollout TP** (`GEN_TP≥16`) or a **disaggregated rollout**
(vLLM on its own GPUs, so training pays no vLLM tax at all — and the weight sync
becomes an NCCL broadcast instead of the ZeRO-3 full-gather that OOM'd rung 4).

**Two realistic targets** (hypotheses — neither was run):

| target | topology (example) | GPUs | co-located? | notes |
|---|---|---|---|---|
| **dense ~72 B** (e.g. Qwen2.5-72B) | `TP=8 PP=1` + FSDP, `GEN_TP=8` | ~64 (8 nodes) | yes | clean TP+DP demo; **different family** → needs staging + a Bridge check |
| **bigger MoE ~235 B-A22B** | `TP=4 EP=8 PP=2` + FSDP, `GEN_TP=16` | ~128 (16 nodes) co-located, or ~64 train + rollout GPUs disaggregated | marginal | same family/tooling; the true "fully distributed" run; heavy quota |

**Recommendation.** For the biggest robust demonstration, go **disaggregated** at
the MoE — it removes the vLLM tax that is the actual scaling wall and matches the
multi-node best-practice (disaggregation = the robustness ceiling). If the goal
is simply "a bigger model, fully distributed, minimal new machinery," the **dense
72B at ~64 GPUs with `TP=8`** is the smaller lift — but it changes model family.

**Open decisions before launching:** (1) which model (and is it staged?);
(2) co-located vs disaggregated rollout; (3) node/quota reality — 4 nodes was
granted, but 8–16 nodes is a larger ask and `capreq-intake` self-service is
denied to this principal, so capacity needs a `team.engineering` path.
