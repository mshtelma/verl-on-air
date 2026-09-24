# math

Trains `Qwen3.5-35B-A3B` with GRPO to solve competition math as an agent: it reasons step by
step, calls a `calculator` tool for arithmetic, and boxes a final answer. The reward is an LLM
judge (GLM-5.3) grading the solution. This is the repo's second reward pattern, next to
agentic-search's exact match.

It is a template, not a benchmark result. It shows the judge machinery end to end: stage a judge,
serve it inside the training job, reach it from the reward function, and optimise its graded
score. If your task needs a judge rather than a rule, start from this one.

## Files

| file | purpose | engine hook |
|---|---|---|
| `reward.py` | calls the served judge, accepts only a strictly valid verdict, returns its 0..1 score; a flagged, budgeted fallback when there is no valid verdict | `CUSTOM_REWARD_PATH` |
| `judge_selfcheck.py` | calibration cases (correct, wrong, prompt injection) the judge must grade before training | `PRE_TRAIN_CHECK` |
| `tool.py` | the calculator: allowlisted AST arithmetic (no `eval`), with sizes, powers and factorials capped; replies never echo the input | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | MATH levels 3–5 to train/test parquet | `train_files` / `val_files` |
| `grading.py` | the final-answer extractor (last `\boxed{}`, else an explicit `####`, never a bare number) and the grader (verl's `prime_math`), shared by the reward's rule score and the eval | |
| `eval.py` | the MATH-500 benchmark with the same tool, graded by `grading.py` | `EVAL_SCRIPT` |

Training optimises the judge's score, which is only a stand-in for correctness: a judge can be
wrong, and the policy is pushed to please it. The independent target is exact correctness on
MATH-500, graded by `grading.py`. The same grader scores every training sample as `acc` (used as
the reward only on a judge fallback), and `judge_agree` shows where the two disagree. If the judge
score rises while `acc` stays flat, the policy is learning the judge, not the math.
`tests/test_grading.py` has the labelled edge cases, including the ones `prime_math` gets wrong.

## Serving the judge

AI Runtime jobs cannot reach each other and each job runs one image, so the trainer and the judge
share a job. `engine/train/dispatch_agentic.sh` splits the nodes by rank:

```
compute.num_accelerators: 32  (4 nodes)
  ranks 0-1  GRPO (TRAINING_NODES=2; ROLLOUT_NNODES=1: rank 0 generates, rank 1 trains)
  ranks 2-3  the judge at TP=16 (JUDGE_NODES=2)
```

The judge head publishes its OpenAI-compatible endpoint to a rendezvous file on the Volume.
Training waits for it (`JUDGE_WAIT_TIMEOUT`), and rank 0 writes a `training_done` sentinel from an
`EXIT` trap so the judge shuts down however training ends. The two halves are separate Ray
clusters and talk only over HTTP. `reward.py` reads the judge URL from the rendezvous file at call
time, because Ray actors don't reliably inherit the driver's environment.

The judge runs on the training image, whose vLLM supports GLM-5.3's architecture. If you swap the
judge, check its architecture with `infra/diagnostics/air/probe_image_engines.yaml`.

## Run

```bash
# 1. MATH levels 3-5 to parquet (stock environment, no GPU work)
air run --file usecases/math/air/1_prep_data.yaml -p <profile> --watch

# 2. stage the judge once (~744 GB; a retry skips complete shards)
air run --file usecases/math/air/2_stage_judge.yaml -p <profile> --watch

# 3. base model on MATH-500. A 32-problem smoke first is cheap; give it its own EVAL_OUT,
#    since artifacts are never overwritten
air run --file usecases/math/air/3_baseline_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_LIMIT=32 env_variables.EVAL_EXPECT_N=32 \
             env_variables.EVAL_OUT=<volume>/eval/math500_base_smoke.json
air run --file usecases/math/air/3_baseline_eval.yaml -p <profile> --watch

# 4. train: GRPO with the judge in the same job (2 train + 2 judge nodes)
air run --file usecases/math/air/4_train.yaml -p <profile> --watch

# 5. eval a checkpoint with the same settings as step 3
air run --file usecases/math/air/5_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_MODEL_PATH=<run>/global_step_24
```

Steps 3 to 5 need the base model staged (`infra/air/stage_model.yaml`). Checkpoints go to
`ckpt/qwen3_5-35b-math-rl/<RUN_ID>/global_step_N/actor/model/huggingface/`. The run has 24 weight
syncs (`768 / (2 × 1 × 16)`) and `SAVE_FREQ: '12'`, so it saves at 12 and 24. The final version is
always saved.

## Settings

The judge reward:

| setting | value | notes |
|---|---|---|
| `REWARD_MANAGER` | `rate_limited` | verl's async reward loop, needed for a network-bound reward |
| `REWARD_MAX_CONCURRENT` | `64` | per reward worker (8 workers, so up to 512 calls in flight); at verl's default of 1 the judge throttles the whole run |
| `REWARD_TIMEOUT` | `120` | per-sample ceiling; the judge client's `JUDGE_DEADLINE_S` (110) stays below it |
| `REWARD_SOURCE` | `judge` | optimise the judge score; `grading.py`'s result is logged as `acc` |
| `JUDGE_FALLBACK` / `JUDGE_MAX_FAIL_RATE` | `rule` / `0.05` | a sample with no valid verdict is scored by exact match and flagged; if more than 5% of a worker's recent calls fail, the run aborts |
| `PRE_TRAIN_CHECK` | `judge_selfcheck.py` | grades the calibration cases through the served judge before training; a miss stops the job |
| `NORM_ADV_BY_STD_IN_GRPO` | `True` | verl's default, set explicitly. It keeps the order and relative gaps of a group's scores; `False` would weight low-spread groups less. Not yet ablated ([docs/tuning.md](../../docs/tuning.md)) |

Judge server (ranks 2–3): `JUDGE_ENGINE=vllm`, `JUDGE_MODEL_PATH`, `JUDGE_NODES=2`,
`JUDGE_TP=16`, `JUDGE_MAX_MODEL_LEN=16384`, `JUDGE_GPU_MEM_UTIL=0.90`,
`JUDGE_LOCAL_CACHE=/local_disk0/judge_cache` (copies the weights to local NVMe first, which is much
faster than reading from the Volume), `JUDGE_RAY_VERSION=2.48.0` (the image's prebuilt Ray for
multi-node serving), `JUDGE_HEALTH_TIMEOUT=2400` (the first 744 GB load is slow), and
`JUDGE_EXTRA_ARGS` for engine-specific parsers.

Judge client (inside the reward workers): `JUDGE_MAX_TOKENS=4096`, `JUDGE_TIMEOUT=50` per attempt,
`JUDGE_RETRIES=1` (transient failures only), `JUDGE_DEADLINE_S=110` in total, `JUDGE_TEMPERATURE=0`,
`JUDGE_TRAJECTORY_CHARS` (how much of the trajectory the judge sees; a cut is logged as
`judge_input_truncated`), `JUDGE_DISABLE_THINKING=1` (some reasoning models otherwise think at
length and break parsing), and `JUDGE_DEBUG=1` to log verdicts. Read `judge_agree` and
`judge_score` relative to `judge_valid`; the metrics are described at the top of
[`reward.py`](reward.py).

Agent loop: `MULTI_TURN=True`, `MAX_TURNS=4` (arithmetic needs few turns), `TOOL_FORMAT=qwen3_coder`,
`MAX_TOOL_RESPONSE_LEN=512`. Async: `TRAIN_MODE=async`, `ROLLOUT_NNODES=1`, and `STALENESS=0.5`,
higher than agentic-search's 0.1 because generation, training and judging all overlap here.

Full reference: [docs/configuration.md](../../docs/configuration.md).

## Picking a dataset

GRPO needs reward variance within groups. A GSM8K version of this use case ran cleanly and taught
the model nothing: with a calculator it solves grade-school problems 95–100% of the time, the
reward sat at 0.94–1.0, and most groups were all correct. That is why `prep_data.py` uses MATH
levels 3–5, where the base model is neither always right nor always wrong. Check this with the
baseline eval and the variance gate (`infra/geo3k/air/2_baseline.yaml` shows the pattern) before
paying for a 4-node training job.

MATH also motivates the judge: its answers are LaTeX (`\frac{1}{2}`, `2\sqrt2`, matrices,
expressions) that don't exact-match cleanly.

## Swapping the judge

Change `MODEL_ID` / `MODEL_DIR` in `2_stage_judge.yaml` and `JUDGE_MODEL_PATH` / `JUDGE_TP` in
`4_train.yaml`. A judge that fits on one node makes this a 3-node job: `num_accelerators: 24`,
`TRAINING_NODES: '2'`, `JUDGE_NODES: '1'`, `JUDGE_TP: '8'`. A hosted judge API needs no judge
nodes: set `TRAINING_NODES` to the node count, drop `JUDGE_NODES`, point `JUDGE_BASE_URL` at the
endpoint, and use `REWARD_MAX_RPM` / `REWARD_MAX_TPM` for its rate limits.
