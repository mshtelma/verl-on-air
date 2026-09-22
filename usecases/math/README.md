# math — MATH-500 agent with a calculator + LLM-judge reward

Train `Qwen3.5-35B-A3B` with GRPO to solve competition math as an **agent**: it reasons
step by step, calls a `calculator` tool for arithmetic, and boxes a final answer. The
reward is an **LLM judge** (GLM-5.3) scoring the solution — the second reward pattern in
this repo, complementing agentic-search's rule-based EM.

> A **template**, not a benchmark claim. Its job is to show the **LLM-judge reward**
> machinery end-to-end (co-locate a judge, serve it, optimise its score). The repo's
> measured demo result is agentic-search ([`../../RESULTS.md`](../../RESULTS.md)); if your
> task needs a judge instead of a rule, copy *this* one and bring your dataset.

## Anatomy

| file | what it is | engine hook |
|---|---|---|
| `reward.py` | the **LLM-judge** reward — calls a served GLM endpoint, returns a graded 0..1 score (rule check as fallback) | `CUSTOM_REWARD_PATH` |
| `tool.py` | the `calculator` tool (safe AST arithmetic) | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | competition MATH (L3–5) → train/test parquet | `train_files`/`val_files` |
| `eval.py` | MATH-500 held-out benchmark; same tool + `\boxed{}` equivalence scoring | `EVAL_SCRIPT` |

The judge is **decoupled**: it runs in its own engine (a newer vLLM/SGLang than the
training image) and the reward reaches it over HTTP. During training the dispatcher
co-locates it — some nodes train, the rest serve the judge (`TRAINING_NODES` <
total nodes) — see [`../../engine/train/dispatch_agentic.sh`](../../engine/train/dispatch_agentic.sh)
and [`../../engine/serve/serve_judge.sh`](../../engine/serve/serve_judge.sh).

## Jobs — prep → (stage judge) → eval → train → eval

```bash
air run --file usecases/math/air/1_prep_data.yaml     -p df1 --watch  # MATH L3-5 -> parquet
air run --file usecases/math/air/2_stage_judge.yaml   -p df1 --watch  # download the GLM judge model
air run --file usecases/math/air/3_baseline_eval.yaml -p df1 --watch  # EVAL: base model on MATH-500
air run --file usecases/math/air/4_train.yaml         -p df1 --watch  # TRAIN: GRPO + co-located judge (4 nodes)
air run --file usecases/math/air/5_eval.yaml          -p df1 \        # EVAL: trained checkpoint
  --override env_variables.MODEL_PATH=<…/global_step_36/actor/model/huggingface> \
            env_variables.EVAL_MODEL_PATH=<same>
```

## The knobs that matter here

- **The judge reward** (`reward.py`) — the whole point. It optimises the judge's graded
  score; the rule check is a validation/fallback.
- **`NORM_ADV_BY_STD_IN_GRPO=False`** — the judge score is *graded*, not 0/1. With GRPO's
  std-normalisation on, 0.05 and 1.0 collapse to the same advantage; turn it off so the
  gradient respects the grade. (See [`../../docs/tuning.md`](../../docs/tuning.md).)
- **judge throughput** — `REWARD_MAX_CONCURRENT` gates concurrent gradings; the judge is
  the reward bottleneck, so size it for your rollout rate.

## A note on headroom

Reward has to have *variance* for GRPO to learn (see the baseline gate in
[`../../docs/tuning.md`](../../docs/tuning.md)). Easy math is correctness-saturated for a
strong base model — pick a difficulty band where the base model is neither always right
nor always wrong, or GRPO has no signal to work with.
