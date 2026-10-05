# GLM-5.3-Flash judge trial

On `trial/verl-0.10`, the math configuration targets **24 H100 GPUs**: 8 for the
trainer, 8 for rollout, and 8 for the judge. The previous configuration used 16
judge GPUs and 32 GPUs overall. This saves one allocated node; total GPU-hours
also depend on judge throughput and training duration.

**The bounded df1 qualifications passed on 2026-10-05:** A10 image smoke, Qwen3.5-2B
training and checkpoint resume, and GLM-5.3-Flash serving/grading on one TP8 H100 node.
The full 24-GPU math job has not been launched. The image uses a pinned verl
**0.10.0.dev** snapshot with a guarded local repair; these results do not establish
production stability or equal judging quality to the previous model.

Current image: `michaelshtelma587/verl-megatron-air:v12-verl010-glmflash-fix1`.
It includes the weight-serializer repair discovered during qualification. It is
published at **16,138,980,912 bytes (16.14 GB)** and registered **AVAILABLE** on
**df1**, the profile selected by the user. The verified registry digest recorded
in `docker/IMAGE.lock` is:

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
The TP8 probe loaded and served it on one 8×80 GB H100 node with a configured 16k
context limit. GPU memory after grading was **77,025–77,505 MiB used / 81,559 MiB total**
per GPU, approximately 75.2–75.7 GiB used. The earlier loading measurement reported
38.8 GiB of weights and 29.38 GiB available for KV cache per GPU; vLLM reserves
cache memory, so final device usage includes more than the model weights.

Use TP=8, the `glm47` reasoning/tool parsers, text-only serving, BF16/auto KV on Hopper,
and the recipe's `--no-enable-flashinfer-autotune`. The initial trial disables custom
all-reduce and uses eager execution, as do the bounded qualification probes.

Flash always reasons. Its template ignores `enable_thinking=False` and defaults to
maximum reasoning effort. The trial sets `JUDGE_DISABLE_THINKING=0` and passes
`JUDGE_REASONING_EFFORT=low` through `chat_template_kwargs`. The client still requires a
complete, schema-constrained JSON verdict in final `content`; reasoning text cannot
supply a grade or a silent fallback pass.

## V12 qualification on df1

The A10 smoke used source commit `5d107d4`; the successful H100 submissions used
`22878fe8af6e85c67453b189ff3447a754391c54`. All used the verified v12 image above.

| Qualification | AIR run | Result |
|---|---|---|
| A10 image smoke | [603122941791377](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/603122941791377) | **PASS**, v12 tag confirmed, no required failures |
| 2B training/checkpoint | [480787754760751](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/480787754760751) | **SUCCESS**, certified `global_step_2`, raw exit 0; 702 seconds |
| TP8 Flash serving/grading | [97695668337323](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/97695668337323) | **SUCCESS / PASS**, all 8 calibration cases and 32 concurrent test requests correct |
| Checkpoint resume | [276390230259878](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/276390230259878) | **SUCCESS**, restored step 2 and dataloader; certified `global_step_4`, raw exit 0; 737 seconds |

Local lint, all eight training compositions, all 28 AIR schema validations on df1,
and all 43 affected dispatch/lifecycle/image-lock tests passed. The image passed
native hashes, torch ABI, CUDA/CCCL compilation, the actual Flash model registry,
all **304 exact index pins**, and the repository training/tool-loop imports. Four
numerical serializer regressions passed. An earlier full CPU suite had 546 passes
and three known expected failures; after correcting two old judge-topology
assertions, all 72 affected tests passed.

Staging run [587738139601693](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/587738139601693)
verified all 69 files, including all 62 model shards, at
`/Volumes/main/mshtelma/verl/models/GLM-5.3-Flash` and wrote the pinned `STAGED.json`.

### Judge measurements

Logical RUN_ID: `glmflash-fix1-capacity-retry-20261005T231140Z-22878fe`.
The probe uses the real reward client and requires schema-constrained JSON in
final content, with no fallback. All 8 sequential calibration cases and all 32
test requests at concurrency 16 parsed and matched their expected grades.

| Measurement | Observed |
|---|---|
| Cold server startup | 1,112.19 seconds, about **18.5 minutes** |
| Local weight copy | 90 seconds, about 3,479 MiB/s |
| Weight shard loading | 28.69 seconds |
| Concurrent test wall time | 20.93 seconds |
| Concurrent throughput | **1.53 requests/second** |
| Median request latency | 9.73 seconds |
| p95 request latency | **14.35 seconds** |
| Reasoning effort | `low` |

The job has a 40-minute timeout and no retries, bounding each attempt to 5.34 H100
GPU-hours. Copy, health, and engine-readiness deadlines are 300, 1800, and 1800
seconds respectively. The cold startup result supports these extended deadlines;
the full math configuration allows 2400 seconds for judge readiness.

### Training and resume evidence

Both 15-minute, zero-retry probes used the same logical RUN_ID:

```text
verl010-glmflash-fix1-retry-20261005T231405Z-22878fe
```

Their isolated output is:

```text
/Volumes/main/mshtelma/verl/ckpt/verl010-smoke/verl010-glmflash-fix1-retry-20261005T231405Z-22878fe
```

One 8×H100 node used 4 trainer and 4 rollout GPUs with Qwen3.5-2B and geo3k. The
initial run used `total_rollout_steps=8`; the second used `RESUME=auto` and
`total_rollout_steps=16`. The actor's `lr_decay_steps=16` stayed unchanged. Each
run is bounded to two H100 GPU-hours.

The certificates verified complete checkpoint manifests and HF exports of
**4,426,558,832 bytes**. Step 2 has identity `766faa2fa02925bb`; step 4 has identity
`61e5cda90ad71f68`. Both report `certified=true`, the expected final version, raw
and final exit 0, and no problems or hard-error matches. The resume log explicitly
records trainer restoration from `global_step_2`, rollout dataloader restoration
from `global_step_2/data.pt`, and a new dataloader checkpoint at
`global_step_4/data.pt`. Terminal AIR status is SUCCESS for both runs. Checkpoint
retention is two, so the original step 2 can be pruned after resume; its initial
certificate is preserved locally.

Verdict and certificate extracts, terminal statuses, and original streamed logs
are saved under `.cache/glm-flash/` in the trial worktree. The AIR links above
provide the corresponding remote run records.

### Repairs and resolved capacity errors

V11 A10 smoke **814922857310924** passed. Its first H100 training run
**188317559647139** failed before generation: NVIDIA Bridge exports strided
weights, but verl's `split_weight_chunks` calls `view(-1)`. The v12 patch accepts
only the SHA256 of the pinned upstream file, makes data buffers contiguous, and
avoids copying weights for metadata-only relays. Four CPU regressions exercise
exact BF16/FP32 reconstruction, both receiver paths, storage reuse, and the
patch's identity guard. The original fails two cases; the patched implementation
passes all four. The successful v12 training and resume also exercise actual
weight transfer on H100s.

The first v11 Flash probe **790755514808491** hit its 600-second health deadline
while compiling DeepGEMM kernels after loading the model. V12 extends the bounded
startup allowance and records progress every 30 seconds; the successful probe
took 1112 seconds to start serving.

The first v12 H100 submissions **1051249691764395** and **1014041783782713** were
rejected before code ran by df1's workspace GPU quota. Capacity became available
on the next judge attempt, and the successful runs above resolved that blocker.
No other workloads were canceled.

## Reproduce and assess the full workload

```bash
make trial-train AIR_PROFILE=df1 BUDGET_OK=1
make trial-judge AIR_PROFILE=df1 BUDGET_OK=1
```

The full math run is configured in `usecases/math/air/4_train.yaml` with
`TRAINING_NODES=2`, `JUDGE_NODES=1`, and `JUDGE_TP=8`. Its output root is isolated at
`/Volumes/main/mshtelma/verl/ckpt/qwen3_5-35b-math-rl-glmflash/<RUN_ID>`.
The successful probes establish basic single-node 2B training/save/resume and
the Flash serving/grading contract on eight H100s. Full 35B MoE training, the
combined math workload, and multi-node EFA transport remain untested on this image.
Passing the fixed judge cases does not establish equal judging quality to GLM-5.3
on competition math, and concurrency 16 does not establish capacity for all 512
potential reward calls or a reduction in total GPU-hours.
Compare `judge_valid`, `judge_agree`, latency, and independent MATH-500 accuracy in a
bounded end-to-end trial before relying on the smaller judge for a long run.

Sources: [model snapshot](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a),
[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash),
[vLLM 0.30 registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py),
[Transformers 5.16.1 config](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/glm5_next/configuration_glm5_next.py).
