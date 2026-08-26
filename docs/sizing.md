# Sizing: how many H100s does Qwen3.5-35B-A3B GRPO actually need?

Every number here is derived from the model's real `config.json` and
cross-checked against verl's own published configs. Reproduce the arithmetic
with `python3 docs/sizing.py`.

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

Per-GPU persistent HBM, no offload, Adam 12 B/param, `GEN_TP=8`
(vLLM weights shard by `GEN_TP`, **not** by N — they do not shrink with nodes):

| N | mode | params | grads | Adam | ref | vLLM | sum | +overhead | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 8 | classic | 10.6 | 10.6 | 52.2 | 10.6 | 8.7 | 92.6 | 110.6 | **OOM** |
| 8 | fsdp | 8.7 | 8.7 | 52.2 | 8.7 | 8.7 | 87.0 | 105.0 | **OOM** |
| 16 | classic | 10.6 | 10.6 | 26.1 | 10.6 | 8.7 | 66.5 | 84.5 | **OOM** |
| **16** | **fsdp** | **4.4** | **4.4** | **26.1** | **4.4** | **8.7** | **47.9** | **65.9** | **OK** |
| 32 | classic | 10.6 | 10.6 | 13.1 | 10.6 | 8.7 | 53.4 | 71.4 | tight |
| 32 | fsdp | 2.2 | 2.2 | 13.1 | 2.2 | 8.7 | 28.3 | 46.3 | roomy |

`overhead = 18 GB`: CUDA ctx 2 + chunked logits/activations 6 + FSDP transient
4 + weight-sync bucket 6. Budget 79.6 GB.

### Conclusions

- **Offload-free minimum is 16 GPUs, and only Megatron-FSDP reaches it.** Classic
  at 16 GPUs OOMs at 84.5 GB purely because it replicates params and grads.
- Classic needs **32+** GPUs to run offload-free.
- `use_precision_aware_optimizer=True` (Adam 8 B/param instead of 12) is enabled
  in both modes and buys real margin: 16-GPU fsdp drops 65.9 → 57.2 GB.

## 4. Why 8 GPUs still "works" — and what it costs

verl's own `run_qwen3_5_35b_megatron.sh` header claims 8 GPUs / 1 node,
`TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8`, `ALL_OFFLOAD=True`. That is real, and it
is what `air/20_...classic_8gpu.yaml` reproduces. Corroborated by verl's perf
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
expert Adam state, EP=8, expert-DP=1  ->   361 GiB  in CPU RAM
non-expert Adam state                 ->    28 GiB
offloaded actor params                ->    79 GiB
offloaded ref params                  ->    79 GiB
                                          ~546 GiB per node
```

`air/00_smoke_test.yaml` prints the node's actual `MemTotal` for this reason. If
it is under ~550 GiB, 8-GPU classic is not viable and you must go to rung 4.
(Azure's 8×H100 SKU is `ND96isr H100 v5`, which carries ~1.9 TiB, so this is
expected to pass — but Databricks does not document the node shape behind
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
alone at **45.5 GB / 138 TFLOPS** — FSDP was both slower *and* hungrier. Never
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
