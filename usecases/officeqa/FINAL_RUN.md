# OfficeQA experiment protocol

The measured experiment trains Qwen3.5-35B-A3B on OfficeQA Full for 100 updates
and compares the original actor with the final checkpoint. It uses PyTorch
2.13.0, verl 0.10.0.dev0 and vLLM 0.30.0. Start with the
[run instructions](README.md#run-a-new-experiment); the
[result report](../../results/officeqa/2026-10-11-final-report.md) contains the
measurements and examples.

## Questions and corpus

OfficeQA Full contains 113 easy and 133 hard questions. Our experiment uses
all easy questions and 90 hard questions for
training, with the remaining 43 hard questions held out. Sorting hard-question
UIDs by their SHA256 hash fixes the split without using model outcomes.
This split follows the [Miles OfficeQA example](https://github.com/mshtelma/miles/tree/main/examples/glm5_3_flash/officeqa);
the creators do not define a training split. Their Pro evaluation uses all 133
hard questions.

Preparation checks that training and held-out IDs are disjoint and rejects
identical question text across splits. Gold answers and source-document hints
stay out of the policy prompt. A manifest records the source files, corpus,
chunks and parquet hashes. [DATA.lock.json](DATA.lock.json) pins the dataset
revision and content hashes; preparation rejects a different snapshot. The
restored renderer was checked against every measured document and the complete
chunk index. Both splits use the same 697-bulletin corpus, so the
evaluation measures generalization to new questions within that archive.

BM25 uses the creators' published transformed text. Grep and document reading
use the lossless archive rebuilt from parsed JSON. The two table renderings
can differ, including numeric formatting; the builder retains both to reproduce
the measured inputs.

## Agent and reward

The agent uses the Miles search, grep, read, list and Python tools, then finishes
with `submit_report(answer, path)`. [SOURCE.json](SOURCE.json) records the port's
provenance. The runtime assigns observation IDs to tool responses and records
which responses reach the actor. A submitted path can cite only delivered
observations. The prompt asks calculations to identify their source observations;
the support judge checks the evidence. Late dependencies are recorded as a
diagnostic and do not by themselves disqualify a report.

The binary reward requires a correct final answer and a supported path. Numeric
answers use our checker at zero tolerance, including checks on
scale, sign and list length. The support judge is called only for correct
answers. Invalid verdicts, service failures and unresolved evidence assessments
stop the run. A report whose cited evidence exceeds the judge's context budget
receives zero reward and a `report_over_budget` metric; evidence is never clipped.

The creators' benchmark scores final-answer correctness. Our answer checker
and evidence-path requirement differ from that evaluation, so we report
answer accuracy and supported pass@1 separately. Answer accuracy counts correct
final answers in well-formed reports, regardless of citation validity or the
support verdict; malformed reports count as incorrect. The current policy has no
web-search or image tool, although some OfficeQA questions need external values
or visual information.

Training and both evaluations share the prompt, controller, tools, scoring and
sampling policy:

| Budget | Value |
|---|---:|
| Assistant turns | 128 |
| Tool calls per turn | 1 |
| Generated tokens per turn | 4,096 |
| Tool-result characters | 8,000 |
| Context tokens | 98,304 |
| Response budget, including tool context | 96,256 tokens |

## Training and evaluation

Training starts from the original actor and uses all 203 training questions,
shuffled with seed 42. The fixed horizon is 100 updates: 400 prompt groups and
1,600 trajectories, with four samples per group and a learning rate of `1e-6`.
Checkpoints are saved every 25 updates. Baseline outcomes do not select the
training questions, and held-out outcomes do not select the checkpoint or run
length.

Each evaluation runs four samples for every held-out question and for 16 fixed
training probes, for 236 episodes in total. The trained evaluation requires the
step-100 checkpoint and checks that the served model matches it. Before/after
comparisons require the same questions, source, policy and data.

Supported pass@1 estimates the chance that one sample has both a correct answer
and accepted evidence. Pass@3 estimates the chance of at least one supported
answer in three samples; it uses the four recorded samples per question.
The paired confidence intervals
resample questions, keeping each question's four samples together. They measure
question uncertainty for this run, not variation across training seeds.

## Compute and judge

Training uses 16 H100s: eight for the Megatron trainer/reference and eight for
vLLM rollout. The trainer uses TP2/CP4/EP8/DP1, packed sequences, a fused LM head
and CPU offload. Each evaluation uses eight H100s.

The judge is a hosted GLM-5.3-Flash service, so no additional judge GPUs are
allocated. Hosted inference tokens are billable. The
[setup instructions](README.md#run-a-new-experiment) cover the model service and
permissions. Five calibration cases (two supported, three unsupported) are each
graded twice before evaluation and training. Each judge request has a 180-second timeout, bounded retries and
a 600-second overall deadline, below the 900-second reward timeout.

The full pipeline's configured timeout bound is about 641 GPU-hours, including
preparation and export. Actual training took about 9.5 hours, and the two
evaluations took about an hour each. See your prepared plan for the applicable
cost bound.

## Run records and completion checks

Preparation requires a clean checkout and saves the code, image, configuration
and submission keys. The controller runs each stage in order and saves its
state locally. Repeating the run command resumes monitoring the existing jobs.
Use the status command to read the saved state.

The local experiment directory contains the plan, state, logs and comparison.
The remote data and checkpoint paths use the experiment name. Image pins are
in [IMAGE.lock](../../docker/IMAGE.lock); copied-code provenance is in
[SOURCE.json](SOURCE.json). Exact source and model identities for the measured
run are retained in its [audit](../../results/officeqa/2026-10-11-final-audit.json).

A failed stage stops the pipeline. In the published experiment, AIR
reported training **FAILED** with `CLIENT_ERROR / RUN_EXECUTION_ERROR`, although
both node commands exited successfully. Before export and evaluation continued,
a manual review verified the clean final certificate, all 1,600 training
trajectories, the final checkpoint, both node exits and the absence of another
error trace. The review pins its evidence files by SHA256. The repository
validates those files but does not automate assembling this audit, and the
exception applies only to that specific AIR error. The failed AIR receipt
remains part of the record; its cause is unresolved.

Checkpoint verification checks shard headers, complete tensors and sampled
contents; it does not hash all 70 GB of weights. A clean process exit alone is
insufficient to approve continuation.

## Recorded outcome

| Held-out metric | Original | Step 100 | Change | Paired 95% interval |
|---|---:|---:|---:|---:|
| Supported pass@1 | 8.72% | 20.93% | +12.21 pp | [+4.65, +20.35] pp |
| Supported pass@3 | 19.77% | 30.23% | +10.47 pp | [0.00, +21.51] pp |
| Answer accuracy | 19.19% | 26.74% | +7.56 pp | [+1.74, +13.95] pp |

Supported successes increased from 15/172 to 36/172: 11 questions improved and
two regressed. The pass@1 interval excludes zero; the pass@3 interval includes
zero. Both evaluations scored every planned episode without infrastructure
errors. The result is from one seeded training run on 43 held-out questions.

The [report](../../results/officeqa/2026-10-11-final-report.md) links the
per-question results, behavior audit and selected improvements and regressions.
