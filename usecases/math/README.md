# math — MATH-500 agent with a calculator + LLM-judge reward

Train `Qwen3.5-35B-A3B` with GRPO to solve competition math as an **agent**: it reasons
step by step, calls a `calculator` tool for arithmetic, and boxes a final answer. The
reward is an **LLM judge** (GLM-5.3) grading the solution — the second reward pattern in
this repo, complementing agentic-search's rule-based EM.

> A **template**, not a benchmark claim. Its job is to show the **LLM-judge reward**
> machinery end to end: stage a judge, co-locate and serve it inside the training job,
> reach it from the reward function, and optimise its graded score. The repo's measured
> demo result is agentic-search ([`../../RESULTS.md`](../../RESULTS.md)); if your task
> needs a judge rather than a rule, copy *this* one and bring your dataset.

## Anatomy

| file | what it is | engine hook |
|---|---|---|
| `reward.py` | the **LLM-judge** reward — calls the served judge, returns a graded 0..1 score (rule check as validator/fallback) | `CUSTOM_REWARD_PATH` |
| `tool.py` | the `calculator` tool (safe AST arithmetic, no `eval`) | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | competition MATH (L3–5) → train/test parquet | `train_files`/`val_files` |
| `eval.py` | MATH-500 held-out benchmark; same tool, `\boxed{}` scored by mathematical equivalence | `EVAL_SCRIPT` |

## How the judge is served — the part worth copying

df1 has **no cross-job connectivity** and one image per job, so the trainer and the judge
have to live in **one job**. `engine/train/dispatch_agentic.sh` splits the nodes by rank:

```
compute.num_accelerators: 32   ->  4 nodes
├── TRAINING_NODES=2   ranks 0-1  GRPO  (ROLLOUT_NNODES=1 -> rank 0 generates, rank 1 trains)
└──                    ranks 2-3  serve the judge at TP=16
```

The judge head publishes its OpenAI endpoint to a **Unity Catalog rendezvous file**;
training waits for it (`JUDGE_WAIT_TIMEOUT`), and rank 0 writes a `training_done` sentinel
from an `EXIT` trap so the judge shuts down on success, failure *or* signal. The two halves
are separate Ray clusters (different ports, the judge with its own Ray pin) and talk only
over HTTP.

One non-obvious detail in `reward.py`: it resolves the judge URL **at call time** from the
rendezvous file rather than trusting an env var, because a Ray actor does not reliably
inherit the driver's exports. That bug once cost a full run in which the judge was never
called and every sample silently fell back to the rule.

The judge also **rides the training image** — no second image, no re-base — because this
image's vLLM already registers the judge's architecture. Verify that for any judge you
substitute with `infra/diagnostics/air/probe_image_engines.yaml`.

## Run it — prep → stage judge → baseline → train → eval

```bash
# 1. MATH L3-5 -> parquet. Stock environment: no custom image, no GPU work.
air run --file usecases/math/air/1_prep_data.yaml     -p df1 --watch

# 2. stage the judge once (~744 GB, resumable: a retry skips complete shards)
air run --file usecases/math/air/2_stage_judge.yaml   -p df1 --watch

# 3. EVAL base model on MATH-500. Ships as a 32-question harness smoke:
air run --file usecases/math/air/3_baseline_eval.yaml -p df1 --watch
#    ...then the real baseline:
air run --file usecases/math/air/3_baseline_eval.yaml -p df1 --watch \
  --override env_variables.EVAL_LIMIT=0

# 4. TRAIN: GRPO + co-located judge, 4 nodes (2 train + 2 judge)
air run --file usecases/math/air/4_train.yaml         -p df1 --watch

# 5. EVAL a checkpoint at the SAME eval settings as step 3
air run --file usecases/math/air/5_eval.yaml          -p df1 --watch \
  --override env_variables.MODEL_PATH=<ckpt>/actor/model/huggingface \
             env_variables.EVAL_MODEL_PATH=<ckpt>/actor/model/huggingface
```

Prerequisite for 3–5: the base model staged
(`air run --file infra/air/stage_model.yaml -p df1 --watch`).

Checkpoints land at `ckpt/qwen3_5-35b-math-rl/global_step_N/actor/model/huggingface/`.
`SAVE_FREQ: '12'` with 24 weight syncs (`768 / (2×1×16)`) means saves at 12 and 24 — pick a
**divisor** of the sync count or the run can finish with no checkpoint at all.

## The settings that matter here

**The judge reward** — the whole point of this use case

| setting | value | why |
|---|---|---|
| `CUSTOM_REWARD_PATH` | `…/reward.py` | the judge-calling scorer |
| `REWARD_MANAGER` | `rate_limited` | verl's **async** reward loop — required for a network-bound reward |
| **`REWARD_MAX_CONCURRENT`** | `64` | verl's internal default is **1 = serial**. Unset, the judge throttles the entire run |
| `REWARD_TIMEOUT` | `120` | per-call ceiling |
| `REWARD_SOURCE` | `judge` | optimise the judge score; the rule is validation/fallback |
| **`NORM_ADV_BY_STD_IN_GRPO`** | **`False`** | the judge score is *graded*. With std-normalisation on, 0.05 and 1.0 get the same advantage and your graded reward collapses to binary ([`../../docs/tuning.md`](../../docs/tuning.md)) |

**Judge server** (ranks 2–3): `JUDGE_ENGINE=vllm` · `JUDGE_MODEL_PATH` ·
`JUDGE_TP=16` · `JUDGE_MAX_MODEL_LEN=16384` · `JUDGE_GPU_MEM_UTIL=0.90` ·
`JUDGE_LOCAL_CACHE=/local_disk0/judge_cache` (bulk-copy off UC FUSE first — much faster
than random-reading it) · `JUDGE_RAY_VERSION=2.48.0` (multi-node serving pin) ·
`JUDGE_HEALTH_TIMEOUT=2400` (a 744 GB first load is slow) · `JUDGE_EXTRA_ARGS` for
engine-specific parsers.

**Judge client** (inside the reward actors): `JUDGE_MAX_TOKENS=4096` ·
`JUDGE_TIMEOUT=90` · `JUDGE_TEMPERATURE=0` (deterministic grading) ·
`JUDGE_TRAJECTORY_CHARS` (how much trajectory the judge sees — truncate too hard and it
grades blind) · `JUDGE_DISABLE_THINKING=1` (some reasoning models think unconditionally at
high effort and wreck the parse rate) · `JUDGE_DEBUG=1` to log prompts.

**Agent loop**: `MULTI_TURN=True` · `MAX_TURNS=4` (arithmetic needs few turns, unlike
retrieval) · `TOOL_FORMAT=qwen3_coder` · `MAX_TOOL_RESPONSE_LEN=512`.

**Async**: `TRAIN_MODE=async` · `ROLLOUT_NNODES=1` · `STALENESS=0.5` — higher than
agentic-search's `0.1` because there are *three* things to overlap here: generate ‖ train ‖
judge.

Full reference: [`../../docs/configuration.md`](../../docs/configuration.md) §10.

## A note on headroom — read this before picking a dataset

Reward must have **variance** for GRPO to learn anything. A GSM8K version of this use case
ran end to end flawlessly and taught the model nothing: a 35B-A3B with a calculator solves
grade-school arithmetic ~95–100%, so the reward pinned at 0.94–1.0 and flatlined. Every
sample in a group scored alike ⇒ zero advantage ⇒ zero gradient.

That is why `prep_data.py` uses MATH with `MATH_LEVELS=3,4,5`: pick a difficulty band where
the base model is neither always right nor always wrong. Verify it with the baseline eval
and the variance gate (`infra/geo3k/air/2_baseline.yaml` shows the pattern) **before**
spending a 4-node training job.

MATH also justifies the judge: its answers are LaTeX (`\frac{1}{2}`, `2\sqrt2`, matrices,
expressions) that do not exact-match cleanly — exactly where a reference-guided judge beats
a string rule.

## Swapping the judge

Change `MODEL_ID`/`MODEL_DIR` in `2_stage_judge.yaml` and
`JUDGE_MODEL_PATH`/`JUDGE_TP` in `4_train.yaml`. A judge that fits **one** node makes this
a 3-node job (`num_accelerators: 24`, `TRAINING_NODES: 2`); a hosted judge API needs no
judge nodes at all — set `TRAINING_NODES` to the node count and point `JUDGE_BASE_URL` at
the endpoint, using `REWARD_MAX_RPM`/`REWARD_MAX_TPM` to respect its rate limits.
