# OfficeQA integration pilot

The pilot checked that retrieval, Python calculations, evidence scoring,
training and checkpoint evaluation worked together. It trained on two selected
questions for two updates. For the full 203-question training experiment, use
the [OfficeQA guide](README.md) and [experiment protocol](FINAL_RUN.md).

## Scope

The implementation adapts the [Miles OfficeQA agent](https://github.com/mshtelma/miles/tree/39093bb88df82a17bc6ebdcb04257f2bd03f6b88/examples/glm5_3_flash/officeqa)
for verl. The agent searches,
greps and reads Treasury Bulletins, calculates with isolated Python, and
finishes with `submit_report(answer, path)`. Its reward requires a correct value
and a supported evidence path. [SOURCE.json](SOURCE.json) records the source
files and reward semantics.

The underlying experiment split has 113 easy + 90 hard training questions and
43 held-out hard questions. The pilot baseline evaluates 16 training probes
and eight held-out questions, with four samples per question. Training selects
only probes whose samples have mixed binary rewards. Held-out outcomes are
excluded from that selection.

| Setting | Pilot value |
|---|---|
| Actor | Qwen3.5-35B-A3B |
| Judge | GLM-5.3-Flash, served on eight H100s |
| Agent limit | 16 turns, one tool call per turn |
| Generation per turn | 4,096 tokens |
| Tool response | Up to 8,000 characters |
| Context | 98,304 tokens |
| Training | Two updates; eight groups × four trajectories |
| Trainer | Eight H100s, TP2/CP4/EP8/DP1, packed sequences and fused LM head |
| Total training allocation | 24 H100s: trainer, rollout and judge nodes |
| Each evaluation | 16 H100s: actor and judge nodes |
| Checkpoints | Both updates saved; evaluate `global_step_2` |
| Job retries | Zero |

Training and evaluation share the agent loop, prompt and reward. Observation
IDs are assigned by the runtime, and only delivered observations count as
evidence. Five support-calibration cases (two supported, three unsupported)
are each graded twice before training and evaluation;
a scorer or service failure stops the job.

The retrieval index contains 697 bulletins and 131,113 BM25 chunks. The image
adds `bm25s` to the tested development stack; versions and digests are recorded
in the [image lock](../../docker/IMAGE.lock).

## What the pilot established

Preparation verified the corpus, questions and rendered prompts. On the actual
image, Python arithmetic, NumPy/pandas and the isolation checks passed. The
agent loop ran with the real tools and recorded every delivered observation
and final report.
The baseline and checkpoint evaluations each scored all 96 episodes
without infrastructure errors. Both training updates saved complete model and
dataloader checkpoints.

The two-update pilot's held-out supported pass@1 was **6.25% → 9.38%**, with
pass@3 unchanged at **12.50%**. Only one additional sample succeeded, on a
question the baseline already answered. These results establish that the
integration works; the eight-question evaluation is too small to establish a
general learning gain.

| Split | Questions × samples | Supported successes, baseline → trained | Supported pass@1 | Supported pass@3 |
|---|---|---|---|---|
| Training probes | 16 × 4 | 3/64 → 7/64 | 4.69% → 10.94% | 10.94% → 17.19% |
| Hard held out | 8 × 4 | 2/32 → 3/32 | 6.25% → 9.38% | 12.50% → 12.50% |

Seven of eight held-out questions had no supported success in either model's
four samples. Most failed episodes exhausted the 16-turn budget. Across the
32 training trajectories, ten earned supported rewards and five of eight
groups had mixed rewards. The
[pilot report](../../results/officeqa/2026-10-06-pilot.json) contains the
per-question results and the checks on data, policy and model identity.

## Run the pilot

First configure your workspace, image and Volume using
[setup](../../docs/setup.md). Build and upload the corpus using the
[OfficeQA instructions](README.md#run-a-new-experiment). The pilot serves its
own judge, so stage the full Flash weights as well as the actor:

```bash
make stage
make math-judge
make officeqa-prep
make officeqa-toolcheck
make officeqa-baseline BUDGET_OK=1

# Use the eval.json path printed by the baseline.
make officeqa-train BUDGET_OK=1 BASELINE=<baseline-eval.json>
make officeqa-eval BUDGET_OK=1 CKPT=<checkpoint-root>/global_step_2
```

Training checks the baseline for mixed-reward groups before selecting questions.
Each submission writes its own artifacts under the run directory.

The optional `officeqa-train-longer` preset continues on the same two selected
pilot questions. Its timeout allows up to 864 GPU-hours and it has no published
result. Use `officeqa-final-prepare` for the measured full-data configuration.
