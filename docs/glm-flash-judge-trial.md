# GLM-5.3-Flash judge trial

On `trial/verl-0.10`, the math configuration targets **24 H100 GPUs**: 8 for the
trainer, 8 for rollout, and 8 for the judge. The previous configuration used 16
judge GPUs and 32 GPUs overall. This saves one allocated node; total GPU-hours
also depend on judge throughput and training duration.

Current image: `michaelshtelma587/verl-megatron-air:v12-verl010-glmflash-fix1`.
It includes the weight-serializer repair discovered during qualification. It is published
at **16,138,980,912 bytes (16.14 GB)** and passes its build gates, four numerical
regressions, and the repository training/tool-loop imports. It is registered
**AVAILABLE** on **df1**, the profile selected by the user. The A10 smoke passes;
H100 qualification is currently blocked by the workspace GPU quota (details below).
The verified registry digest recorded in `docker/IMAGE.lock` is:

```text
sha256:7bb2fbf97763e4aae1c79ed7f6c5515a25519c868bbbce002476c722d19f388a
```

The earlier v11 control is registered on df1 and passed its A10 smoke. Its digest is:

```text
sha256:b05fc38e1714107e68ac906ee0daf7b8709acc332f00b4822e8f23084fc50332
```

## Compatibility and memory

The standard vLLM 0.29 wheel lacks `Glm5NextForConditionalGeneration`. The released
**0.30.0** wheel and its tagged source include it, and still require **torch 2.13.0**.
The trial retains the original torch/CUDA ABI and native wheel hashes, upgrades
Transformers to **5.16.1**, and follows vLLM's FlashInfer/CUTLASS/QuACK/MCP dependencies
under the regenerated lock. Build gates require the actual GLM config and registry.

Model: `zai-org/GLM-5.3-Flash`, pinned revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
The Hub metadata reports **321,323,031,390 total parameters**, roughly 18B active,
and **328,337,455,672 bytes** across 62 safetensors files (about 306 GiB).
This is the native FP8 checkpoint, with about 6.9B parameters retained in BF16.
One 8×80 GB H100 node has enough aggregate memory for the weights plus serving
overhead at the trial's 16k context. Actual startup and concurrent serving must pass
the GPU probe; weight size alone is insufficient.

Use TP=8, the `glm47` reasoning/tool parsers, text-only serving, BF16/auto KV on Hopper,
and the recipe's `--no-enable-flashinfer-autotune`. The initial trial disables custom
all-reduce and uses eager execution, as do the bounded qualification probes.

Flash always reasons. Its template ignores `enable_thinking=False` and defaults to
maximum reasoning effort. The trial sets `JUDGE_DISABLE_THINKING=0` and passes
`JUDGE_REASONING_EFFORT=low` through `chat_template_kwargs`. The client still requires a
complete, schema-constrained JSON verdict in final `content`; reasoning text cannot
supply a grade or a silent fallback pass.

## Qualification

Local gates passed on 2026-10-05: v11 is **16,138,942,720 bytes (16.14 GB)**,
below the 19.5 GB gate; native hashes, torch ABI, CUDA/CCCL compilation, the actual
Flash model registry, and all **304 exact index pins** verify. The fully-async entry
point and repository's tool-agent loop import from the built image. All 28 jobs
pass AIR schema validation on df1 and all eight training jobs compose. The CPU
suite had 546 passes and three known expected failures; two assertions for the old
judge topology were corrected, and all 72 affected tests then passed.

V11 A10 smoke **814922857310924** passed with no required failures. The first
H100 training run **188317559647139** failed before generation: NVIDIA Bridge
exports strided weights, but verl's `split_weight_chunks` calls `view(-1)`.
The v12 patch accepts only the SHA256 of that pinned upstream file, makes data
buffers contiguous, and avoids copying weights for metadata-only relays. Four CPU
regressions exercise exact BF16/FP32 reconstruction, both receiver paths, storage
reuse, and the patch's identity guard. The original implementation fails the two
numerical/metadata cases; the patched implementation passes all four.

The first Flash probe **790755514808491** hit its 600-second health deadline
while compiling DeepGEMM kernels. Model loading had succeeded at **38.8 GiB per
GPU**, with **29.38 GiB per GPU** available for KV cache, confirming memory headroom
on one TP8 H100 node. Serving and grading remain unqualified. The probe now allows
1800 seconds for cold loading/compilation and 300 seconds for the measured 91-second
local copy, still within a 40-minute job with no retries. Engine readiness uses the
same extended timeout. Startup progress is recorded every 30 seconds.

### V12 results on df1

All submissions below used source commit `5d107d4` and the verified v12 digest above.
Local lint passes, all eight training jobs compose, all 28 AIR specs validate on df1,
and all 43 affected dispatch/lifecycle/image-lock tests pass. The image also passes
the four numerical serializer regressions and the actual repository imports.

| Qualification | AIR run | Result |
|---|---|---|
| A10 image smoke | [603122941791377](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/603122941791377) | **PASS**, v12 tag confirmed, no required failures |
| 2B training/checkpoint | [1051249691764395](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/1051249691764395) | Rejected before code ran: H100 workspace quota |
| TP8 Flash serving/grading | [1014041783782713](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/1014041783782713) | Rejected before code ran: H100 workspace quota |
| Checkpoint resume | — | Pending a certified `global_step_2` from the repaired image |

The platform reports: `Workspace has exceeded its GPU quota for GPU_8xH100. The
quota limit for this workspace is 24 node(s).` No active H100 AIR runs were visible
in the subsequent all-user query; that query does not establish available quota.
Neither rejected run produced a training checkpoint or judge verdict. The v12
weight-transfer fix and longer startup allowance still need H100 qualification.
The full 24-GPU math job has not been launched.

Staging is already complete on df1: run **587738139601693** verified all 69 files,
including all 62 model shards, and wrote the pinned `STAGED.json`. Reuse that verified
snapshot for the remaining probes.

### Remaining qualifications

1. Run the 15-minute Qwen3.5-2B training probe with a fresh run identity. Require a
   certified `global_step_2`, a completed checkpoint manifest, and a verified HF export.
   It uses one 8×H100 node, zero retries, and at most two H100 GPU-hours per attempt.
2. Resume that exact RUN_ID with `RESUME=auto` and `total_rollout_steps=16`. Keep the
   learning-rate horizon at 16 in both runs. Require trainer and dataloader restoration
   from step 2 and a certified `global_step_4`; the same 15-minute bound applies.
3. Run `infra/diagnostics/air/glm_flash_judge.yaml`: 1×8 H100, TP=8, 40-minute timeout,
   zero retries. Maximum 5.34 H100 GPU-hours per attempt. It starts the real judge,
   grades all eight existing calibration cases, then grades 32 requests at concurrency
   16. Every verdict must parse and match the expected correctness, without fallback.
   It records request latency and throughput under a unique run directory.

```bash
make trial-train AIR_PROFILE=df1 BUDGET_OK=1
make trial-judge AIR_PROFILE=df1 BUDGET_OK=1
```

The full math run is configured in `usecases/math/air/4_train.yaml` with
`TRAINING_NODES=2`, `JUDGE_NODES=1`, and `JUDGE_TP=8`. Its output root is isolated at
`/Volumes/main/mshtelma/verl/ckpt/qwen3_5-35b-math-rl-glmflash/<RUN_ID>`.
Qualify the judge first. Passing the fixed cases establishes the serving/grading
contract; it does not establish equal judging quality to GLM-5.3 on competition math,
nor does concurrency 16 establish capacity for all 512 potential reward calls.
Compare `judge_valid`, `judge_agree`, latency, and independent MATH-500 accuracy in a
bounded end-to-end trial before relying on the smaller judge for a long run.

Sources: [model snapshot](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a),
[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash),
[vLLM 0.30 registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py),
[Transformers 5.16.1 config](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/glm5_next/configuration_glm5_next.py).
