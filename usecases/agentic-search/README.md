# agentic-search

Trains `Qwen3.5-35B-A3B` with GRPO as a multi-hop search agent. For each question the model runs
a tool loop (`vector_search`, `keyword_search` and `read_article` over a Databricks Vector Search
index) and gives a final answer in `<answer>…</answer>`. The reward is exact match against the
gold answer, with no judge and no reward model, which makes this the cheapest agentic setup in the
repo.

In one run, the best of 13 checkpoints scored 58.5% on the 200-question development set against
54% for the base model. On 500 held-out test questions it scored 37.6% against 34.8%, a gain that
is not statistically significant. A second run with a different seed, its checkpoint named before
training, scored 40.4% on the same test questions (p = 0.004). Details are in
[RESULTS.md](../../RESULTS.md). With your own corpus and questions, the same jobs run unchanged.

## Files

| file | purpose | engine hook |
|---|---|---|
| `reward.py` | exact-match scorer; reads only the model's own `<answer>` (via role spans); its `score_segments` also scores the eval | `CUSTOM_REWARD_PATH` |
| `tool.py` | the three Vector Search tools | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | MuSiQue questions to train/test parquet at pinned revisions; defines `SYSTEM_PROMPT` | `train_files` / `val_files` |
| `eval.py` | the benchmark; imports `reward.py` and `tool.py` and records its own loop policy | `EVAL_SCRIPT` |
| `make_splits.py`, `splits/` | the fixed dev and held-out test question IDs | |
| `build_corpus.py`, `create_vs_index.py` | the passage corpus (fails if a source fails) and its content-versioned index | prep jobs |
| `analyze_traces.py` | EM decomposition over eval traces | |
| `probe_vs_access.py` | checks index access before you pay for a GPU node | |
| `tests/` | CPU tests for the reward, tools and eval | |

## Before you start

1. Build and register the image and create the Volume ([docs/setup.md](../../docs/setup.md)).
2. Stage the base model once: `air run --file infra/air/stage_model.yaml -p <profile> --watch`.
3. Have a Vector Search endpoint (`QA_VS_ENDPOINT`), or let the index job create one with
   `QA_VS_CREATE_ENDPOINT: '1'` (billable), and a SQL warehouse to load the table.
4. Run the tool-format probe (`infra/diagnostics/air/probe_tool_format.yaml`) to confirm the
   model's tool-call format.

## Run

```bash
# 1. questions and the passage corpus (1xA10)
air run --file usecases/agentic-search/air/1_prep_data.yaml -p <profile> --watch

# 2. Delta table and Vector Search index. Returns before the index is ready; it prints the
#    QA_VS_INDEX to use in steps 3-5 and the row count that means "ready".
air run --file usecases/agentic-search/air/2_build_index.yaml -p <profile> --watch \
  --override env_variables.QA_VS_WAREHOUSE_ID=<id>

# 3. base model eval (8xH100)
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p <profile> --watch

# 4. train: GRPO, fully-async, 16xH100
air run --file usecases/agentic-search/air/4_train.yaml -p <profile> --watch

# 5. eval a checkpoint with the same settings
air run --file usecases/agentic-search/air/5_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_MODEL_PATH=<run>/global_step_20 \
             env_variables.EVAL_OUT=<volume>/eval/agentic_search_step20.json \
             env_variables.EVAL_TRACE_OUT=<volume>/eval/agentic_search_step20_traces.jsonl
```

Checkpoints are written every 10 weight syncs (`SAVE_FREQ: '10'`) to
`ckpt/agentic-search-grpo/<RUN_ID>/global_step_N/actor/model/huggingface/`. Evaluate several on
the dev split, pick one, then score it once on the held-out test split. Picking the best of many
on the same questions inflates the number; `scripts/paired_eval.py` reports both the raw and the
selection-adjusted p. [docs/running-jobs.md](../../docs/running-jobs.md) shows how to run the
test split. Deployment is not implemented ([docs/deploy.md](../../docs/deploy.md)).

## Settings

All of these are `env_variables:` in the job files. The full list is in
[docs/configuration.md](../../docs/configuration.md).

| setting | value | notes |
|---|---|---|
| `MULTI_TURN` | `True` | verl's `ToolAgentLoop` |
| `MAX_TURNS` | `12` | the hop budget, and the first knob to tune |
| `TOOL_FORMAT` | `qwen3_coder` | Qwen3.5 writes XML tool calls; with the wrong parser the tools never fire |
| `MAX_TOOL_RESPONSE_LEN` | `4000` | the default of 512 would cut off every passage |
| `QA_VS_ENDPOINT` / `QA_VS_INDEX` | from `config.env` | jobs in the same workspace use ambient auth |
| `QA_SEARCH_TOP_K` | `5` | hits per call |
| `QA_SNIPPET_CHARS` / `QA_TOOL_MAX_CHARS` | `600` / `4000` | keep tool responses readable |
| `REWARD_MANAGER` | `naive` | in-process, no judge |
| `QA_REWARD_METRIC` | `em` | read by `reward.py` and `eval.py` |
| `QA_RETRIEVAL_BONUS` | `0.0` | bonus when a tool surfaced a gold answer string; it did not help in the one run that tried it |
| `TRAINING_NODES` | `2` (the node count) | no judge nodes |
| `TRAIN_MODE` / `ROLLOUT_NNODES` | `async` / `1` | one node generates, one trains |
| `STALENESS` / `TRIGGER_SYNC_STEP` | `0.1` / `1` | |
| `ROLLOUT_PREFIX_CACHING` | `True` | avoids re-prefilling the prompt and earlier turns on every turn |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `True` | vLLM's custom all-reduce fails at intra-node `GEN_TP≤8` on H100; this keeps CUDA graphs |
| `EP` / `GEN_TP` / `TP` | `8` / `8` / `2` | |

Batch and horizon (`parameters:`): `rollout_n: 16`, `ppo_mini_batch_size: 32`,
`total_rollout_steps: 3200`, `actor_lr: 2e-6`, `max_prompt_length: 2048`,
`max_response_length: 512`. In multi-turn mode `max_response_length` is not a per-turn cap: the
launcher turns it into one episode-wide budget of `(2048 + 512) × 12 − 2048 = 28,672` tokens.
Only the evals cap each request (`EVAL_MAX_TOKENS=512`).

## MAX_TURNS

`MAX_TURNS` is the agent's retrieval budget; set it to roughly the number of hops your questions
need. Moving the eval from 8 to 12 turns took the base model from 104 to 108 of 200 (8 gained, 4
lost), which is within noise. It also drives backward-pass memory at
`ppo_micro_batch_size_per_gpu=1`, because the actor trains on the whole trajectory. Keep
`EVAL_MAX_TURNS` the same in the baseline and trained evals, or you are measuring a budget change.

## Reward

The answer is a short span, so exact match is a faithful, free and deterministic reward. The
[math](../math) use case shows the other pattern, an LLM judge for open-ended answers.

A retrieval bonus (extra credit when a tool surfaced the gold answer) did not help in the one run
that tried it. A possible reason, not measured: the answer string is retrieved for about 80% of
questions, so the bonus fires for most samples in a group and moves the group mean instead of
separating rollouts. A reward term can only teach if it varies within a group, so check how often
it fires within groups before relying on it.

## Diagnostics

```bash
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl> \
    --label base --label step20 --out diag/          # [--supporting-from-musique]
.venv/bin/python -m pytest usecases/agentic-search/tests/ -q      # CPU, a few seconds
```

`analyze_traces.py` splits EM by whether a tool surfaced a gold answer string and pairs the two
runs by question ID (the first file is the reference). `--supporting-from-musique` adds the share
of each question's supporting paragraphs that were found. The answer-string check is only a proxy
for retrieval, so treat the split as a hint for the next experiment. More in
[RESULTS.md](../../RESULTS.md).
