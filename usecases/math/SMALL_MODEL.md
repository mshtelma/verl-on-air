# One-node math experiment

Branch `experiment/verl010-small-math` is based on the OfficeQA/verl 0.10 trial
at `33ff660`. It reuses the v13 image and the existing calculator, math judge
reward, and independent MATH-500 grader. Repository code travels in the AIR
snapshot; this change requires no image rebuild.

The actor is **Qwen3.5-2B**, pinned to
`15852e8c16360a2fea060d615a32b45270f8a8fc` (2.27B total parameters, including
the vision component; these episodes are text only). Its 4.43 GB checkpoint,
training, weight synchronization and exact resume already passed on the 0.10
stack. That qualification used short, single-turn Geo3K episodes; this math
tool-agent preset still needs its own GPU qualification.

One **8xH100 node** is split into four classic Megatron trainer/reference GPUs
and four independent TP1 vLLM rollout replicas. Training TP/CP/EP are all one,
with DP4, CPU offload and micro-batch one. The total training context is 8,192
tokens, with at most four assistant/tool turns. A small dense model needs no
multi-node or expert-parallel setup.

The judge is the existing **`databricks-glm-5-3-flash`** endpoint in **df1**.
It was READY with CAN_QUERY permission on 2026-10-06, and passed all eight
existing math calibration cases using the reward client's JSON-schema request
and strict verdict parser. This is a hosted API: it consumes no GPUs from our
training job, but endpoint tokens are billable and rate limits still apply.
Other discovered choices include Claude Sonnet 5.5, Gemini 3.8 Flash and GPT-6
Luna; their calibration and throughput have not been tested here.

`JUDGE_PROVIDER=databricks` uses refreshable ambient SDK credentials inside AIR.
For local calibration only, set `JUDGE_DATABRICKS_PROFILE=df1`; no token is
stored in the job spec. The hosted request sends `reasoning_effort=low` as an
API parameter, while the existing self-hosted client retains its template
parameter. Every request must produce a valid verdict. Any judge failure after
the bounded retry raises the run's abort channel, and fallback samples are
flagged, scored zero and never treated as valid judge responses.

Math is the first choice because a 2B model can receive useful rewards on the
easier levels, the tool is small, and the existing independent answer scorer
lets us distinguish improved correctness from higher judge scores. Agentic
search is the next choice for retrieval training and already has a free
exact-match reward. OfficeQA requires much longer contexts, evidence accounting
and more tool skills, making it a harder first small-model experiment.

Training uses only the MATH **train** split, levels **1-3**, in its own directory.
The experiment evaluates all **500 MATH-500** problems before training and after
a fixed final checkpoint at temperature zero. Compare the paired independent
answer grades, with confidence intervals, rather than using judge-score increases
as evidence of learning. Record reward spread and correctness during the pilot
before proceeding to a longer run. The MATH
mirror is accepted only through the existing pinned content-digest verification.

## Run

All model, dataset, checkpoint and evaluation paths are separate from OfficeQA.
Preparation uses a single A10 for downloads and CPU work; baseline, training
and checkpoint evaluation each use one 8xH100 node, sequentially.

```bash
make math-small-prep AIR_PROFILE=df1
make math-small-baseline AIR_PROFILE=df1 BUDGET_OK=1 RUN_ID=<baseline-id>
# Inspect the completed baseline before training.
make math-small-train AIR_PROFILE=df1 BUDGET_OK=1 RUN_ID=<train-id>
make math-small-eval AIR_PROFILE=df1 BUDGET_OK=1 RUN_ID=<eval-id> \
  CKPT=/Volumes/main/mshtelma/verl/ckpt/qwen3_5-2b-math-hosted-glmflash/<train-id>/global_step_4
```

The initial training pilot is **four optimizer updates**: 32 prompt groups,
four trajectories each, mini-batch eight and synchronization every update.
It saves complete checkpoints at steps two and four. It has a 45-minute
timeout, zero retries and a hard bound of **6 GPU-hours**. Each full evaluation
is capped at 60 minutes (**8 GPU-hours**), and preparation at 30
minutes (**0.5 A10 GPU-hours**). API judge usage is additional. Four updates
qualify the experiment; they are not a meaningful long training run.

Artifacts are named per RUN_ID. Every training trajectory records its runtime
rollout group, parameter version, judge validity, reward and independent answer
grade. Complete evaluation artifacts and training audits are also exported in
checksummed log envelopes, so results can be recovered if Volume downloads fail.

If the pilot has valid judge responses and useful within-group reward variation,
continue from step four to a fixed **100 total optimizer updates** (800 prompt
groups and 3,200 training trajectories in total). Both stages use the same learning
rate horizon of 800. Set the continuation timeout from observed pilot throughput;
save a bounded number of checkpoints and evaluate only the predetermined final
step. Evaluation never selects the training examples or checkpoint.

Preparation pins `fsspec==2026.6.0` alongside datasets 5 and Hub 1.33; the stock
environment's older fsspec fails dataset resolution. It also runs all ten judge
calibration cases using ambient AIR authentication before any H100 job starts.

The first pilot stopped before its first optimizer update when GLM returned
`correct=false, score=0.7` for partially correct working. The strict parser and
zero-failure budget stopped that run. The revised schema constrains each
correctness flag to its valid score band, and the rubric emphasizes final-answer
correctness. Calibration now includes two additional partial-working cases
(ten total). Replay the recorded training answers with `recheck_judge.py` on
the A10 preparation environment before restarting H100 training. Preserve the
failed attempt and its source revision alongside the replacement run.
