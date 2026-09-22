# agentic-search — multi-hop RAG agent (the flagship demo)

Train `Qwen3.5-35B-A3B` with GRPO to be a **multi-hop search agent**: given a question,
it runs a multi-turn tool loop — `vector_search` / `keyword_search` / `read_article`
over a **Databricks Vector Search** index — and commits a final answer. The reward is a
**rule-based exact match**: no LLM judge, no reward model. This is the cheapest possible
agentic-RL setup, and it's the main showcase for *how little* a use case has to bring.

> This is a **template**, not a benchmark result. The example numbers live in
> [`../../RESULTS.md`](../../RESULTS.md): base **54%** → trained **~57% (peak 58.5%)** EM
> at a matched 12-turn eval on 200 held-out MuSiQue questions. Bring your own
> corpus/questions and the same jobs run unchanged.

## Anatomy — a use case is this thin

| file | what it is | engine hook |
|---|---|---|
| `reward.py` | the rule-based EM scorer (also the eval scorer — same code) | `CUSTOM_REWARD_PATH` |
| `tool.py` | the agent's tools over Vector Search (`vector_search`/`keyword_search`/`read_article`) | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | MuSiQue questions → train/test parquet (defines the shared `SYSTEM_PROMPT`) | `train_files`/`val_files` |
| `eval.py` | held-out benchmark; reuses `reward.py` + `tool.py` so eval == training | `EVAL_SCRIPT` |
| `build_corpus.py`, `create_vs_index.py` | build the passage corpus + the Vector Search index | prep jobs |
| `analyze_traces.py` | the **recall × conversion** diagnostic (how we find the bottleneck) | — |
| `tests/test_reward.py` | CPU unit tests for the reward (20, pure stdlib) | — |

Everything hard (the 35B MoE topology, the multi-turn agent loop, fully-async rollout,
weight sync) lives once in [`../../engine/`](../../engine); this folder just plugs into it.

## Jobs — prep → train → eval → deploy

```bash
air run --file usecases/agentic-search/air/1_prep_data.yaml    -p df1 --watch  # questions + corpus
air run --file usecases/agentic-search/air/2_build_index.yaml  -p df1 --watch  # Vector Search index (wait until ONLINE)
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch  # EVAL: base model
air run --file usecases/agentic-search/air/4_train.yaml         -p df1 --watch  # TRAIN: GRPO, fully-async, 16xH100
air run --file usecases/agentic-search/air/5_eval.yaml          -p df1 \        # EVAL: trained checkpoint
  --override env_variables.MODEL_PATH=<…/global_step_20/actor/model/huggingface> \
            env_variables.EVAL_MODEL_PATH=<same>
air run --file usecases/agentic-search/air/6_deploy.yaml        -p df1 --watch  # DEPLOY: spec/recipe (SERVE=1 to serve)
```

Base and trained eval use the **same** 12-turn budget so the delta is apples-to-apples.

## The one knob to tune first

`MAX_TURNS` — the agent's retrieval hop budget. It's the headline lever here (8→12 lifted
the base model +2 EM and protected recall). Match it to how many hops your questions
need. The full knob guide is [`../../docs/tuning.md`](../../docs/tuning.md); the training
mode is fully-async — [`../../docs/training-modes.md`](../../docs/training-modes.md).

## Why rule-based reward (no judge)?

The answer is a short span; exact match against the gold answer is a faithful, free,
deterministic reward. That makes this the cheapest agentic-RL loop in the repo — a good
first template to copy. (The `math` use case shows the other pattern: an **LLM-judge**
reward for open-ended answers.)
