# agentic-search — multi-hop RAG agent (the flagship demo)

← [verl-on-air](../../README.md) · [the case study](../../RESULTS.md) · [running-jobs](../../docs/running-jobs.md) · [configuration](../../docs/configuration.md)

Train `Qwen3.5-35B-A3B` with GRPO to be a **multi-hop search agent**: given a question it
runs a multi-turn tool loop — `vector_search` / `keyword_search` / `read_article` over a
**Databricks Vector Search** index — and commits a final answer in `<answer>…</answer>`.
The reward is a **rule-based exact match**: no LLM judge, no reward model. This is the
cheapest possible agentic-RL setup, and the main showcase for *how little* a use case has
to bring.

> This is a **template**, not a benchmark result. The example numbers live in
> [`../../RESULTS.md`](../../RESULTS.md): base **54%** → trained **~57% (peak 58.5%)** EM
> at a matched 12-turn eval on 200 held-out MuSiQue questions. Bring your own
> corpus/questions and the same jobs run unchanged.

## Anatomy — a use case is this thin

| file | what it is | engine hook |
|---|---|---|
| `reward.py` | the rule-based EM scorer (**also the eval scorer — same code**) | `CUSTOM_REWARD_PATH` |
| `tool.py` | the agent's tools over Vector Search | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | MuSiQue questions → train/test parquet; defines the shared `SYSTEM_PROMPT` | `train_files`/`val_files` |
| `eval.py` | held-out benchmark; imports `reward.py` + `tool.py` so eval == training | `EVAL_SCRIPT` |
| `build_corpus.py`, `create_vs_index.py` | build the passage corpus + the Vector Search index | prep jobs |
| `analyze_traces.py` | the **recall × conversion** diagnostic (how you find the bottleneck) | — |
| `probe_vs_access.py` | check index access before paying for a GPU node | — |
| `tests/test_reward.py` | 20 CPU unit tests for the reward, stdlib only | — |

Everything hard (35B MoE parallelism, the multi-turn agent loop, fully-async rollout,
weight sync, multi-node Ray) lives once in [`../../engine/`](../../engine).

## Prerequisites

1. Image built + registered, UC Volume created — [`../../docs/setup.md`](../../docs/setup.md).
2. Base model staged once: `air run --file infra/air/stage_model.yaml -p df1 --watch`.
3. A **Vector Search endpoint** to hold the index (`QA_VS_ENDPOINT`, default
   `wiki-qa-vs`). Create it once in the workspace if it does not exist.
4. Recommended: `air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch`
   — confirms the model's tool-call format before you pay for training.

## Run it — prep → baseline → train → eval → deploy

```bash
# 1. questions + the union passage corpus  (1xA10)
air run --file usecases/agentic-search/air/1_prep_data.yaml    -p df1 --watch

# 2. Delta table + Vector Search index. RETURNS BEFORE THE INDEX IS READY — wait for ready.
air run --file usecases/agentic-search/air/2_build_index.yaml  -p df1 --watch
databricks vector-search-indexes get-index main.mshtelma.wiki_qa_big_corpus_index \
  -p df1 --output json    # wait for status.ready == true

# 3. EVAL the base model = the "before" number  (8xH100)
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch

# 4. TRAIN: GRPO, fully-async, 16xH100, judge-free
air run --file usecases/agentic-search/air/4_train.yaml         -p df1 --watch

# 5. EVAL a checkpoint with the IDENTICAL settings -> the delta is the result
air run --file usecases/agentic-search/air/5_eval.yaml          -p df1 --watch \
  --override env_variables.MODEL_PATH=<ckpt>/actor/model/huggingface \
             env_variables.EVAL_MODEL_PATH=<ckpt>/actor/model/huggingface \
             env_variables.EVAL_OUT=/Volumes/main/mshtelma/verl/eval/agentic_search_step20.json \
             env_variables.EVAL_TRACE_OUT=/Volumes/main/mshtelma/verl/eval/agentic_search_step20_traces.jsonl

# 6. DEPLOY: prints the recipe; SERVE=1 brings up a vLLM endpoint
air run --file usecases/agentic-search/air/6_deploy.yaml        -p df1 --watch
```

Checkpoints land at
`ckpt/agentic-search-grpo/global_step_N/actor/model/huggingface/` (`SAVE_FREQ: '10'` —
every 10 weight syncs). **Evaluate several**: the best held-out checkpoint here was step
20, with a plateau after, so the last checkpoint is not automatically the one you want.

Operational detail — monitoring, capacity, what to grep for:
[`../../docs/running-jobs.md`](../../docs/running-jobs.md).

## The settings that define this use case

Everything below is `env_variables:` in the job files; full reference in
[`../../docs/configuration.md`](../../docs/configuration.md).

**The agent loop**

| setting | value | why |
|---|---|---|
| `MULTI_TURN` | `True` | verl's `ToolAgentLoop` |
| **`MAX_TURNS`** | `12` | ⭐ the hop budget — the headline lever (see below) |
| `TOOL_FORMAT` | `qwen3_coder` | Qwen3.5 emits XML, not JSON. Wrong value = tools silently never fire |
| `MAX_TOOL_RESPONSE_LEN` | `4000` | retrieval returns passages, so the default 512 would truncate every hit |
| `AGENT_NUM_WORKERS` | `8` | parallel agent-loop actors |

**Retrieval** — the tool's whole configuration surface

| setting | value | why |
|---|---|---|
| `QA_VS_ENDPOINT` / `QA_VS_INDEX` | `wiki-qa-vs` / `main.mshtelma.wiki_qa_big_corpus_index` | which index to query. Same workspace ⇒ **ambient auth**, no token, no pip install at query time |
| `QA_SEARCH_TOP_K` | `5` | hits per call |
| `QA_SNIPPET_CHARS` / `QA_TOOL_MAX_CHARS` | `600` / `4000` | keep a tool response readable instead of flooding the context |

**Reward** (rule-based, no judge)

| setting | value | why |
|---|---|---|
| `CUSTOM_REWARD_PATH` | `…/reward.py` | the scorer |
| `REWARD_MANAGER` | `naive` | in-process, no judge to rate-limit |
| `QA_REWARD_METRIC` | `em` | **read by `reward.py` AND `eval.py`** — one metric, no drift |
| `QA_RETRIEVAL_BONUS` | `0.0` | bonus for surfacing the gold passage. Measured **inert** here — see below |
| `TRAINING_NODES` | `2` (= node count) | no judge nodes |

**Async topology**

| setting | value | why |
|---|---|---|
| `TRAIN_MODE` | `async` | disjoint Rollouter/Trainer pools |
| `ROLLOUT_NNODES` | `1` | 1 whole node generates, 1 trains (a 1:1 split — the ratio is *not* learning-neutral) |
| `STALENESS` / `TRIGGER_SYNC_STEP` | `0.1` / `1` | freshness vs overlap |
| `ROLLOUT_PREFIX_CACHING` | `True` | big multi-turn win: the shared system prompt + prior turns are re-prefilled every turn otherwise |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `True` | avoids the vLLM graph-capture crash at `GEN_TP≤8`, *keeping* CUDA graphs |
| `EP` / `GEN_TP` / `TP` | `8` / `8` / `2` | MoE expert sharding; rollout TP stays intra-node |

**Batch / horizon** (`parameters:`)

`rollout_n: 16` (GRPO group size) · `ppo_mini_batch_size: 32` ·
`total_rollout_steps: 3200` · `actor_lr: 2e-6` · `max_prompt_length: 2048` ·
`max_response_length: 512` (per turn — the *episode* budget is
`(2048+512)×12` tokens).

## The one knob to tune first

**`MAX_TURNS`** — the agent's retrieval hop budget. Going 8→12 lifted the *base* model
about 2 EM and was recall-safe, and it is the setting to match to how many hops your
questions actually need. Two consequences worth knowing:

- It is also the primary **backward-memory** cost at `ppo_micro_batch_size_per_gpu=1`,
  because the actor trains on the whole trajectory. Large turn budgets OOM in the actor
  backward before they run out of anything else.
- `EVAL_MAX_TURNS` must equal it, and must be **identical between the baseline and the
  trained eval** — otherwise you are measuring a budget change and calling it learning.

## Why rule-based reward (no judge)?

The answer is a short span, so exact match against the gold is a faithful, free,
deterministic reward — the cheapest agentic-RL loop in the repo, and a good first template.
The [`math`](../math) use case shows the other pattern: an **LLM-judge** reward for
open-ended answers.

One GRPO lesson from this use case, worth internalising before you design a reward: adding
a **retrieval bonus** (extra credit when a retrieved passage contained the gold) did
*nothing*. Recall was already ~80%, so the bonus fired on nearly every sample in a group —
and advantage is `(reward − group_mean) / group_std`, so a bonus that lands in the mean
cancels itself. **A reward term only teaches if it discriminates within the group.**

## Diagnose, don't guess

```bash
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
uv run --with pytest --no-project python -m pytest \
    usecases/agentic-search/tests/test_reward.py -q                     # 20 tests, CPU, <1s
```

`analyze_traces.py` decomposes `EM = recall × conversion` — did retrieval surface the gold
passage, and given that it did, did the model answer correctly? Measured here: recall
~79–81%, with conversion as the gap, which is what said "GRPO should improve *use* of
retrieved context, and turns should protect recall". Without that decomposition you are
guessing across a hundred knobs. Full story: [`../../RESULTS.md`](../../RESULTS.md).
