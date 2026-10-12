# Using GLM-5.3-Flash as a judge

GLM-5.3-Flash served and graded on one eight-H100 node in the qualification
probe. The math recipe therefore allocates **24 H100s**: eight for training,
eight for rollout and eight for the judge. The previous judge needed two
nodes, for 32 GPUs overall. This reduces the allocation by one node; total
GPU-hours still depend on startup, throughput and training duration.

OfficeQA's measured experiment instead calls a hosted Flash service and
allocates no judge GPUs. Its [guide](../usecases/officeqa/README.md) describes
that setup.

## Model and serving configuration

The recipe uses the native FP8 [GLM-5.3-Flash
checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash). It has about 321B total
parameters, roughly 18B active, and 328 GB of weights. The staged revision is
recorded by the model-staging job. The vLLM 0.30.0 and Transformers 5.16.1 pins
include its architecture; vLLM 0.29 does not.

The probe served with TP8 and a 16k context on eight 80 GB H100s. Measured device
use after grading was about 75–76 GiB per GPU, including the reserved KV cache.
Keep the context and memory settings within the qualified budget when starting
with this configuration.

Flash always reasons: its chat template ignores `enable_thinking=False` and
defaults to maximum reasoning effort. The reward client sets
`JUDGE_DISABLE_THINKING=0` and `JUDGE_REASONING_EFFORT=low`. A grade must appear as
schema-constrained JSON in final content; reasoning text is not a verdict.

The server uses the `glm47` parsers, text-only serving and the recipe's
`--no-enable-flashinfer-autotune`. The qualification probe uses eager execution
with custom all-reduce disabled. Dependencies and image digests are in the
[stack guide](verl-0.10-trial.md) and [image lock](../docker/IMAGE.lock).

## Qualification results

The real reward client passed all eight calibration cases and all 32 test
requests at concurrency 16. Each response parsed and matched its expected
grade without fallback.

| Measurement | Observed |
|---|---|
| Cold server startup | About 18.5 minutes |
| Local weight copy | 90 seconds |
| Weight loading | 28.7 seconds |
| Concurrent throughput | 1.53 requests/second |
| Median latency | 9.73 seconds |
| p95 latency | 14.35 seconds |
| Reasoning effort | `low` |

Compilation on first startup dominates the wait. The probe allows 40 minutes
with no retries, bounding an attempt to 5.33 H100-hours. The math configuration
allows 2,400 seconds for judge readiness. Copying the weights to local NVMe
avoids repeatedly reading them from the Volume.

The accompanying 2B probe completed two training updates and then resumed from
step 2 to step 4, restoring both trainer and dataloader state. That checks the
repaired weight-transfer path used by this image. See the
[stack qualification](verl-0.10-trial.md) for its scope.

## Try it in your workspace

Follow [setup](setup.md), stage the judge with the
[math staging job](../usecases/math/air/2_stage_judge.yaml), and run:

```bash
make trial-judge BUDGET_OK=1
```

The full math job is in [4_train.yaml](../usecases/math/air/4_train.yaml), with
`TRAINING_NODES=2`, `JUDGE_NODES=1` and `JUDGE_TP=8`. Its reward workers can issue
up to 512 concurrent calls, whereas the qualification measured concurrency 16.
Before a long run, check throughput with the intended call volume and compare
judge agreement with independent MATH-500 correctness. The full combined math
workload remains unqualified, and passing these fixed cases does not establish
equal judging quality to the previous model.

Source references: [vLLM serving recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash),
[vLLM 0.30 model registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py),
[Transformers model configuration](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/glm5_next/configuration_glm5_next.py).
