# Adding a use case

A new RL task is a few small files and some YAML; nothing under `engine/` changes. You write:

```
usecases/<your-task>/
├── prep_data.py      your data to verl parquet
├── reward.py         a trajectory to a score
├── tool.py           the agent's tools (skip for single-turn)
├── eval.py           the held-out benchmark, reusing reward.py
└── air/
    ├── 1_prep_data.yaml       1×A10
    ├── 3_baseline_eval.yaml   the "before" number
    ├── 4_train.yaml           GRPO
    └── 5_eval.yaml            the "after" number
```

Start from the shipped use case closest to your task:

```bash
cp -r usecases/agentic-search usecases/my-task    # rule-based reward, tools, retrieval
cp -r usecases/math           usecases/my-task    # LLM-judge reward, one tool
```

| your file | env var the job sets |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` (and `CUSTOM_REWARD_NAME`) |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `eval.py` | `EVAL_SCRIPT` |
| parquet from `prep_data.py` | `parameters.train_files` / `val_files` |

## prep_data.py

One row per prompt, written as parquet to the Volume:

```python
{
  "data_source": "my_task",              # picks verl's built-in scorer if you have no custom one
  "agent_name": "tool_agent",            # routes to verl's ToolAgentLoop (multi-turn); omit for single-turn
  "prompt": [                            # raw chat messages, not a rendered string
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user",   "content": question},
  ],
  "ability": "my_task",
  "reward_model": {"style": "rule", "ground_truth": gold},   # the only place the answer goes
  "extra_info": {"split": "train", "index": i, "question": question, ...},
}
```

Define `SYSTEM_PROMPT` here and import it in `eval.py` (`from prep_data import SYSTEM_PROMPT`) so
training and eval use the same prompt. Put `ground_truth` in `reward_model` only: if the model
can see it, you are measuring leakage. Keep `extra_info.index` unique across merged sources. For
text-only data set `image_key: ''` in the job's `parameters:`, or the multimodal path turns
itself back on. Write to a new directory so you never overwrite data an earlier run used. Run it
as a 1×A10 job; if it needs no GPU libraries, use a stock environment with a `dependencies:`
list instead of the custom image, as `usecases/math/air/1_prep_data.yaml` does.

## reward.py

```python
def compute_score(data_source: str = "", solution_str: str = "", ground_truth=None,
                  extra_info: dict | None = None, **kwargs) -> dict[str, float]:
    ...
    return {"score": score, "answer_correct": ..., "well_formed": ...}
```

- verl calls it with exactly these keyword arguments; always accept `**kwargs`.
- Return a dict with a `score` key. GRPO optimises `score`; every other key becomes its own
  MLflow metric, which is how you tell "learning the answer" apart from "learning the format".
- In multi-turn mode `solution_str` is the whole episode: the model's turns, the tool responses
  and the chat-template text. An `<answer>` inside a tool response is not the model's answer, and
  the model can type text that looks like a tool response. The launchers record who wrote which
  characters (`extra_info["role_spans"]`, split with
  [`engine/lib/role_spans.py`](../engine/lib/role_spans.py)); `usecases/agentic-search/reward.py`
  reads the answer from assistant spans only and refuses to score without them.
- Fail closed: a judge outage or a parse failure must not return a passing score. Keep the file
  testable on a CPU without torch, like `usecases/agentic-search/tests/test_reward.py`.
- For an LLM judge, copy `usecases/math/reward.py`. It reads the judge URL from the rendezvous
  file on every call (Ray actors do not reliably inherit the driver's environment), is `async`
  for the `rate_limited` manager, accepts only a strictly valid verdict, never raises (verl turns
  an exception into a 0.0 reward with a different key set), aborts the run when the judge keeps
  failing, and runs calibration cases first (`judge_selfcheck.py` via `PRE_TRAIN_CHECK`).
- A graded reward stays graded: std-normalisation (`NORM_ADV_BY_STD_IN_GRPO`) keeps the order
  and gaps within a group ([tuning.md](tuning.md)).

## tool.py

```python
from verl.tools.function_tool import function_tool

@function_tool("my_search")
def my_search(query: str, top_k: int = 5) -> str:
    """One-line description the model reads.

    Args:
        query: what to search for.
        top_k: how many results.
    """
    return "...text the model sees..."
```

The docstring and type hints become the tool schema the model sees. Tools are global and
stateless: verl offers them to every sample and does not pass `tools_kwargs` through, so
configure them with env vars (`QA_SEARCH_TOP_K`, `QA_VS_INDEX`, ...). Return a string, and cap
its length inside the tool (`QA_TOOL_MAX_CHARS`) as well as through the engine's
`MAX_TOOL_RESPONSE_LEN`. Never raise: return an error string the model can react to (the search
tools keep a raising `*_impl` next to each wrapper, so the eval can tell an outage from a wrong
answer). Inside a job, Databricks services such as Vector Search use ambient auth.
In the training job, set `MULTI_TURN: 'True'`, `MAX_TURNS` and a `TOOL_FORMAT` that matches what
your model emits, and confirm it with `infra/diagnostics/air/probe_tool_format.yaml`: with the
wrong parser the tool calls never decode, and the run looks healthy while learning nothing.

## eval.py

`engine/serve/serve_and_eval.sh` serves `EVAL_MODEL_PATH` with vLLM, waits for `/health`, puts
your script's directory on `PYTHONPATH` and runs it. The script reads `EVAL_BASE_URL` and
`EVAL_MODEL`, shows the model the training prompt and the tool schemas from verl's
`@function_tool` registry, parses tool calls with verl's `TOOL_FORMAT` parser (public
`extract_tool_calls`, no fallback), scores with `import reward`, and writes `EVAL_OUT` (summary
JSON) and `EVAL_TRACE_OUT` (per-question JSONL). Its agent loop is its own code, so it records
how it differs from training's as a versioned `eval_policy` (the shipped evals re-render the chat
each turn, cap each request, allow one tool call per turn and can force a final answer).

Run the baseline and trained evals with the same settings: the two eval jobs differ only in the
model and the output path (a test checks this), and `scripts/paired_eval.py` refuses to pair
artifacts whose `eval_policy` differs. If `EVAL_MAX_TURNS` differs from training's `MAX_TURNS`,
say why (math uses 8 against 4, because 4 cut the base model off before it boxed an answer).

## The job files

Copy the four from the use case you started from and change these fields:

```yaml
code_source:
  snapshot:
    root_path: ../../..              # from usecases/<uc>/air/ this is the repo root
    include_paths:
      - engine
      - usecases/my-task

env_variables:
  FUNCTION_TOOL_PATH: ${CODE_SOURCE_PATH}/usecases/my-task/tool.py
  CUSTOM_REWARD_PATH: ${CODE_SOURCE_PATH}/usecases/my-task/reward.py
  CUSTOM_REWARD_NAME: compute_score
  EVAL_SCRIPT:        ${CODE_SOURCE_PATH}/usecases/my-task/eval.py      # eval jobs

parameters:
  train_files: /Volumes/.../data/my_task/train.parquet
  val_files:   /Volumes/.../data/my_task/test.parquet
  output_dir:  /Volumes/.../ckpt/my-task-grpo
```

Set `experiment_name` too; runs group by it in MLflow. Leave the topology (`TP`, `EP`, `GEN_TP`,
...) alone unless you change the model or the GPU count. A judge-free task sets
`TRAINING_NODES` to the node count; for a judge, copy
`usecases/math/air/{2_stage_judge,4_train}.yaml` and add nodes, with `JUDGE_NODES` stated.

## Before you spend GPU hours

```bash
uv run --with pytest --no-project python -m pytest usecases/my-task/tests/ -q   # the reward, locally
make check && make dry F=usecases/my-task/air/4_train.yaml                       # free

# the resolved verl config, without a cluster
DRY_RUN=1 TRAINING_NODES=1 NUM_NODES=1 LOCAL_WORLD_SIZE=8 POD_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=1 RENDEZVOUS_ROOT=/tmp/rdv \
  MULTI_TURN=True FUNCTION_TOOL_PATH='${CODE_SOURCE_PATH}/usecases/my-task/tool.py' \
  CUSTOM_REWARD_PATH='${CODE_SOURCE_PATH}/usecases/my-task/reward.py' \
  bash engine/train/dispatch_agentic.sh

air run --file infra/diagnostics/air/probe_tool_format.yaml -p <profile> --watch
```

Then run the jobs in order ([running-jobs.md](running-jobs.md)):

1. `1_prep_data.yaml`, then `3_baseline_eval.yaml` at the final eval settings.
2. Check that the reward separates samples. If the base model scores around 95% or around 2%,
   fix the data or the reward first; both shipped use cases changed dataset for this reason
   (HotpotQA to MuSiQue, GSM8K to MATH).
3. `4_train.yaml` with a small `total_rollout_steps` and `SAVE_FREQ=1`, to see the tools fire,
   the reward get called and checkpoints get written.
4. The real run. Evaluate several checkpoints, not just the last.

Parallelism, multi-node Ray, weight sync, the completion certificate, eval serving and judge
co-location are handled in [`engine/`](../engine).
