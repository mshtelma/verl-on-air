# infra — platform validation ("will my expensive run work?")

← [verl-on-air](../README.md) · [running-jobs](../docs/running-jobs.md) · [sizing](../docs/sizing.md) · [ladder](../docs/ladder.md)

This is **not** a use case. It is what you run *before* spending money on a real training
job, to prove the hard parts of the platform work on your cluster and your image. Two
tiers, cheapest first.

The rule that justifies all of it: a failure found on 1 A10 costs a couple of GPU-minutes;
the same failure found 40 minutes into a 32-GPU job costs GPU-hours.

## Tier 1 — `diagnostics/` (seconds to minutes, mostly 1×A10)

Two kinds, and only the first is ever read as a pass:

- **Gates** exit non-zero unless their claim holds, and end with one machine-readable line,
  `PROBE_VERDICT {"probe": ..., "ok": ..., "status": PASS|FAIL|INCONCLUSIVE|ERROR, "reasons": [...]}`
  (`diagnostics/probe_verdict.py`; `PROBE_VERDICT_OUT=<path>` also writes it to a file).
  INCONCLUSIVE is a failure: a gate that could not decide has not passed.
- **Diagnostics** measure and print; their exit status is not a verdict, and "it ran" says
  nothing about your job.

| job | kind | what it checks | when |
|---|---|---|---|
| `air/smoke_test.yaml` | **gate** | image imports, arch, driver floor, **a real on-device bf16 matmul**, CPU RAM, C compiler, **`AutoBridge` resolves the model (required: the `MEGATRON_MODE=fsdp` gate; `SMOKE_REQUIRE_BRIDGE=0` demotes it, recorded)** | after every image build (`make smoke`) |
| `air/probe_tool_format.yaml` | **gate** | **verl's parser for `PROBE_TOOL_FORMAT` (default `qwen3_coder`) reads exactly the tool call your model's own chat template writes** — the template-rendered call only; a hand-written sample is reported, never a pass | **before any agentic training job** |
| `diagnostics/probe_cross_node_http.py` | **gate** | every one of the N×N node pairs answers HTTP (the judge topology); a node with an empty or partial peer map, or its own `all_ok` false, fails | before a multi-node judge on a new cluster |
| `usecases/agentic-search/probe_vs_access.py` | **gate** | an ANN and a HYBRID query on `QA_VS_INDEX` return rows with id/title/text; read-only, installs nothing | before the first search job |
| `air/diag_cuda.yaml` | diagnostic | CUDA visible, bf16 on device, driver ≥ CUDA 13 floor | a node pool looks wrong |
| `air/diag_te.yaml` | diagnostic | TransformerEngine multi-tensor path | TE-related crash |
| `air/probe_image_engines.yaml` | diagnostic | which model architectures this image's vLLM can serve | before choosing a judge model |
| `air/probe_vllm_multinode.yaml` | diagnostic | how to serve one model across nodes with this vLLM | before a multi-node judge |
| `air/test_rollout_allreduce.yaml` (8×H100) | diagnostic | the vLLM custom-all-reduce crash + the two graph-preserving fixes | rollout dies at init |
| `air/env_probe.yaml` | diagnostic | what the runtime actually injects (env, PATH, venv) | "it works locally" |

Two of these earn their keep repeatedly:

- **`probe_tool_format`** catches the most expensive *silent* failure in agentic RL. With
  the wrong parser the model's tool calls fail to decode, the job never errors, and you
  pay full price to train an agent that never uses its tools.
- **`smoke_test`**'s `cpu ram` line decides whether CPU offload is viable at all
  (`OFFLOAD=1` wants ~400–500 GB per node), and its `AutoBridge` check gates
  `MEGATRON_MODE=fsdp` entirely.

## Tier 2 — `geo3k/` (the scaling ladder)

Real GRPO runs on the small **geo3k** dataset that prove the topology holds. Each rung
changes exactly **one** variable from the previous, so a failure localises to that rung
instead of to "the stack".

| rung | model | backend | GPUs | offload | purpose | measured |
|---|---|---|---|---|---|---|
| `air/rung1_2b_fsdp_8gpu.yaml` | Qwen3.5-2B (dense) | Megatron-FSDP | 8 | no | the whole code path, cheapest | ~911 s |
| `air/rung2_9b_fsdp_8gpu.yaml` | Qwen3.5-9B (dense) | Megatron-FSDP | 8 | no | co-located rollout memory starts to matter | ~1000 s |
| `air/rung3_35b_classic_8gpu.yaml` | **35B-A3B** (MoE) | classic Megatron | 8 | **yes** | reproduces verl's own tested config | ~1477 s |
| `air/rung4_35b_fsdp_32gpu.yaml` | **35B-A3B** (MoE) | **Megatron-FSDP** | **32** | **no** | **the headline topology** | ~1443 s |

> **Rung 4 runs on 32 GPUs, not the 16 the sizing model gives — read this before quoting a
> number.** 16 GPUs is the smallest topology whose *persistent* state fits offload-free,
> and that is the number the sizing model predicts. But the co-located rollout adds a
> **weight-sync transient** the persistent budget misses (the ZeRO-3→HF full-tensor gather
> landing on top of vLLM's re-woken weights), which OOMs at 16. 32 halves the shards and
> the same transient fits with room to spare. The file name is kept only because the
> `Makefile` rung4 target points at it. Full arithmetic: [`../docs/sizing.md`](../docs/sizing.md).

Rungs 3 and 4 are the same model, data and reward with two different sharding strategies —
that contrast *is* the point. Classic Megatron's distributed optimizer is ZeRO-1 and
**replicates** params and grads across DP, so adding nodes never fixes its OOM below 32
GPUs; Megatron-FSDP is ZeRO-3 and shards all three.

Note that the *use cases* do not use this topology at all: they run **fully-async**, where
the trainer gets its own node with classic+offload and generation happens on a disjoint
node — 16 GPUs total. Two different answers to the same memory problem; see
[`../docs/training-modes.md`](../docs/training-modes.md).

### The reward-variance gate

`geo3k/air/2_baseline.yaml` is not a training job — it samples the base model and reports
**the fraction of sample-groups with non-zero reward variance**. GRPO normalises reward
within each group of `rollout_n` samples, so a group where every sample scores identically
yields advantage 0 and contributes **no gradient**. That fraction is your effective batch
size, and `pass@1` will not tell you what it is.

Run it before committing a dataset to a real training job. It is the cheapest insurance in
the repo, and both shipped use cases changed dataset because of this exact effect
(HotpotQA → MuSiQue; GSM8K → MATH).

`geo3k/reward.py` is a tested, dict-returning drop-in reward you can point
`CUSTOM_REWARD_PATH` at — useful as a minimal reference for the reward contract. One
measured subtlety it demonstrates: geo3k's accuracy term is gated on `\boxed{}`
extraction, so a *correct but unboxed* answer scores **0.00, not 0.90**. Run
`python3 infra/geo3k/reward.py` to see the whole surface.

## Shared setup

`infra/air/stage_model.yaml` stages the base `Qwen3.5-35B-A3B` (~70 GB) from Hugging Face
to the Unity Catalog Volume **once**; every use case and rung reads it from there. Pulling
it per run would cost 10–20 minutes of paid H100 time each time.

## Run

```bash
make smoke                                        # tier-1 image pre-flight
make prep && make stage                           # geo3k data + the base model
make baseline                                     # the reward-variance gate
make rung1 && make rung2 && make rung3 && make rung4

air run --file infra/diagnostics/air/probe_tool_format.yaml -p <profile> --watch   # before agentic jobs
```

Each rung ships `total_training_steps: 3` as a smoke cap — set it to `0` for a real run.
The 64-example geo3k subset gives 2 steps per epoch, so a rung printing "2/3" and SUCCESS
is correct, not truncated. Details in [`../docs/ladder.md`](../docs/ladder.md); operational
guide in [`../docs/running-jobs.md`](../docs/running-jobs.md).
