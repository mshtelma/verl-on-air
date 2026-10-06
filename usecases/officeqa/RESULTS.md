# OfficeQA pilot results

Data, isolated compute and the complete controller have passed the qualifications
linked in [README.md](README.md). The first baseline run
[5796847728497](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/5796847728497)
failed during judge calibration from commit `4fa7f71` with the qualified v12 trial
image. Nine of ten calibration calls passed; one concurrent cold tokenizer load
failed in Transformers' lazy import. No policy question was evaluated and no
training was started. The reward raised the abort channel; this is an infrastructure
failure, not a negative answer or evidence judgment.

The corrected calibration passed 10/10 on run
[867023020948845](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/867023020948845).
The compact baseline was then canceled after 44 episodes finished without a
terminal submission: 41 reached the 12-action limit and three stopped without a
tool call. All 44 had resolved zero rewards and no infrastructure errors. This
was an incomplete diagnostic run, not a benchmark result, and training was not
launched. Traces showed keyword-overlap retrieval returning irrelevant periods,
and a calculation cut off at the 1,024-token generation limit.

The revised pilot adds pinned BM25 retrieval, 16 actions, 4,096 tokens per
generation and 8,000-character tool observations. It uses a new immutable
16-action prompt snapshot and preserves the official question split and pilot
IDs. Its Qwen server uses CUDA graphs with custom all-reduce disabled; baseline,
training and checkpoint evaluation retain identical controller/reward budgets.

The v13 image passed its exact dependency-lock check and a real BM25 ranking
probe, then was pushed and registered on `df1`; its digest is in
[IMAGE.lock](../../docker/IMAGE.lock). Revised preparation
[285706995986882](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/285706995986882)
passed isolated arithmetic and produced a maximum prompt length of 1,907 tokens.
Qualification
[720095605769482](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/720095605769482)
verified unchanged Miles source hashes, corpus/chunk hashes, official split and
pilot question IDs. It indexed 697 bulletins / 131,113 chunks with BM25, passed
compute staging, and completed the real continuous-token controller with 16
requests, 15 delivered observations, a final-turn submission and 31 role spans.

The revised 96-episode baseline
[747485359876719](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/747485359876719),
submitted from `92dce54` with logical ID
`officeqa-baseline3-20261006T022752Z-92dce54`, completed successfully. All 96
episodes were scored with zero infrastructure errors; the artifact is valid.
Flash passed all ten support calibrations. The actor served the 98,304-token
context with CUDA graphs; capture took 24 seconds and 0.44 GiB per GPU.

| Baseline split | Questions × samples | Supported correct episodes | Strict pass@1 | Strict pass@3 | Terminal submissions |
|---|---|---|---|---|---|
| Training probes | 16 × 4 | 3 / 64 | 4.69% | 10.94% | 6 / 64 |
| Hard held out | 8 × 4 | 2 / 32 | 6.25% | 12.50% | 2 / 32 |

Four training-side answers matched gold, but only three had a valid supported
path. Two training groups had mixed binary rewards: `UID0186` (one success in
four) and `UID0236` (two successes in four). The other 14 training groups had
zero successes. Most episodes exhausted the action budget, so this remains a
weak starting policy despite the successful runtime integration.

The first bounded training attempt
[75550371685956](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/75550371685956)
used these two training questions and passed all ten Flash calibrations. Its
live rollouts included supported correct answers and zero rewards, but the first
actor update failed when Megatron converted long-trajectory vocabulary logits to
float32. TP2 / EP8 / DP4 required individual allocations of 18–23 GiB with only
11–16 GiB free. This attempt does not count as completed training.

The retry uses TP2 / CP4 / EP8 / DP1 on the same eight trainer GPUs, with packed
sequences and verl's fused LM head. The pinned Megatron Core 0.19.2 supports packed
Gated DeltaNet; the fused head avoids the full vocabulary-logits tensor and CP4
shards the sequence. Megatron TP8 is not valid for this actor's two KV heads. The image,
dataset, seed, controller, reward and episode budgets remain identical. The pilot
still runs eight prompt groups × four trajectories and two saved updates;
held-out outcomes are excluded from selection.

The retry
[635676617334740](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/635676617334740),
submitted from `777d581` with logical ID
`officeqa-train4-20261006T035332Z-777d581`, completed successfully. The artifact
audit verified all 32 trajectories, five mixed-reward groups, ten supported
correct answers and zero infrastructure errors. Parameter versions 0 and 1 each
produced 16 trajectories; `UID0186` and `UID0236` each contributed 16. These are
training rewards on two selected questions, not an accuracy comparison with the
24-question evaluation set. Flash passed all ten support calibrations again.

Both `global_step_1` and `global_step_2` completed their model and dataloader
saves. The final certificate reports `certified=true`, expected/observed version
2, raw/final exit 0, no abort and no hard-error matches. The final HF export holds
70,214,492,304 bytes of weights; its verified identity is `b21d08a7b51558e6` and
its checkpoint-contents manifest SHA256 is
`33024922adc9b07f55817ccedff6987c138f42f6b7218dd6e47ec9ed32b55583`.
The checkpoint is:

```text
/Volumes/main/mshtelma/verl/ckpt/qwen3_5-35b-officeqa-miles-pilot/officeqa-train4-20261006T035332Z-777d581/global_step_2
```

Paired checkpoint evaluation
[580525832146364](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/580525832146364)
was submitted from the same commit with logical ID
`officeqa-checkpoint3-20261006T045242Z-777d581`, completed successfully. All
96 episodes were scored with zero infrastructure errors, and the artifact is
valid. The paired audit verified the exact checkpoint identity, identical
dataset/manifest and question-ID hashes, controller, strict reward, sampling
policy and image. Recomputed pass@k summaries match the recorded metrics.

| Split | Questions × samples | Supported correct, base → checkpoint | Strict pass@1, base → checkpoint | Strict pass@3, base → checkpoint |
|---|---|---|---|---|
| Training probes | 16 × 4 | 3 / 64 → 7 / 64 | 4.69% → 10.94% | 10.94% → 17.19% |
| Hard held out | 8 × 4 | 2 / 32 → 3 / 32 | 6.25% → 9.38% | 12.50% → 12.50% |

The extra held-out success is on the same question, `UID0190`, that had both
baseline successes. Seven of eight held-out groups still have all-zero rewards;
held-out pass@3 is unchanged. The checkpoint's training-probe successes are on
`UID0236` (four), `UID0169` (two) and `UID0199` (one); the selected training
question `UID0186` has no successful evaluation sample. Most failures still
exhaust the action budget. These point estimates are from a small integration
pilot with two selected training questions and two updates; they do not establish
a general learning gain.

The paired artifact is:

```text
/Volumes/main/mshtelma/verl/eval/officeqa/officeqa-checkpoint3-20261006T045242Z-777d581/eval.json
```

Flash passed 10/10 support calibrations before this evaluation. The actor's local
weights matched the certified checkpoint. vLLM emitted `EngineDeadError` during
the launcher's deliberate post-evaluation shutdown, after all scores and the valid
artifact had been written; the job terminated `SUCCESS`. All 15 OfficeQA
qualification, diagnostic, pilot and export jobs are terminal.

[The machine-readable pilot report](../../results/officeqa/2026-10-06-pilot.json)
records image/dependency pins, data and artifact hashes, the training certificate,
all 32 training rewards, paired per-question sample rewards and job states.
The full Miles curriculum and the complete 43-question held-out split have not
been run.

A continuation to `global_step_100` was launched on `df1` as
[747475740428360](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/747475740428360)
from commit `9139556`, resuming the certified step-2 checkpoint. It requests
98 further updates and 1,568 trajectories on the same two training questions,
with 24 H100s, checkpoints every ten updates, two retained actor checkpoints,
no retries and a 36-hour limit. Training was in startup when
[the launch record](../../results/officeqa/2026-10-06-longer-launch.json) was written.
A background monitor will launch the same 96-episode paired evaluation after
the final checkpoint and training artifacts pass their audits. Results are pending.
