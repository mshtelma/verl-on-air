# infra — platform validation ("will my expensive run work?")

This is **not** a use case. It's what you run *before* spending money on a real training
job, to prove the hard parts of the platform work on your cluster + image. Two tiers,
cheapest first.

## Tier 1 — `diagnostics/` (seconds to minutes)

Fast health probes. Run these first; a failure here costs a couple of GPU-minutes
instead of surfacing 16 GPU-hours into a training run.

| job | checks |
|---|---|
| `diagnostics/air/smoke_test.yaml` | image imports, arch, CPU RAM, compiler (1×A10, ~2 min) |
| `diagnostics/air/diag_cuda.yaml` | CUDA visible, bf16 matmul on-device, driver floor |
| `diagnostics/air/diag_te.yaml` | TransformerEngine multi-tensor path |
| `diagnostics/air/probe_tool_format.yaml` | the tokenizer emits/parses `qwen3_coder` tool calls |
| `diagnostics/air/probe_vllm_multinode.yaml`, `test_rollout_allreduce.yaml`, `env_probe.yaml`, `probe_image_engines.yaml` | multi-node vLLM, the custom-all-reduce workaround, env, engines |

## Tier 2 — `geo3k/` (the scaling ladder)

Actual GRPO training runs on the **geo3k** dataset that prove the topology holds — in
particular that a 35B MoE trains with **Megatron-FSDP and no CPU offload on 16×H100**,
the config classic Megatron can't reach below 32 GPUs. Each rung changes **one** variable
from the previous, so a failure is cheap to localise.

| rung | model | backend | GPUs | offload | purpose |
|---|---|---|---|---|---|
| `geo3k/air/rung1_2b_fsdp_8gpu.yaml` | Qwen3.5-2B (dense) | Megatron-FSDP | 8 | no | full code path, cheapest |
| `geo3k/air/rung2_9b_fsdp_8gpu.yaml` | Qwen3.5-9B (dense) | Megatron-FSDP | 8 | no | FSDP sharding starts to matter |
| `geo3k/air/rung3_35b_classic_8gpu.yaml` | **35B-A3B** (MoE) | classic Megatron | 8 | **yes** | reproduces verl's own tested config |
| `geo3k/air/rung4_35b_fsdp_16gpu.yaml` | **35B-A3B** (MoE) | **Megatron-FSDP** | **16** | **no** | **the headline topology** |

Rungs 3 and 4 are the same model, data and reward with two different sharding
strategies — that contrast *is* the point. The memory arithmetic (why 16 GPUs, why FSDP)
is in [`../docs/sizing.md`](../docs/sizing.md).

`geo3k/air/2_baseline.yaml` also runs the **reward-variance gate**: GRPO learns only from
within-group reward variance, so it reports the fraction of groups with non-zero variance
— run it before committing a dataset to a real training job.

## Shared setup

`infra/air/stage_model.yaml` stages the base `Qwen3.5-35B-A3B` (~67 GiB) from Hugging Face
to the Unity Catalog Volume once; every use case and rung reads it from there.

## Run

```bash
make smoke                                        # tier-1 image preflight
air run --file infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml -p df1 --watch   # tier-2 headline
```
