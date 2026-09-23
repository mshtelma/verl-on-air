# Bring your own task: a new use case in five files

← [verl-on-air](../README.md) · [running-jobs](running-jobs.md) · [configuration](configuration.md) · [tuning](tuning.md)

The claim this repo makes is that once the engine exists, **a new RL task is a handful of
small files and zero infrastructure work**. This page is that claim, made concrete.

You will not edit anything under `engine/`. You will write:

```
usecases/<your-task>/
├── prep_data.py      your data      → verl parquet
├── reward.py         a trajectory   → a score          (the whole task, really)
├── tool.py           the agent's tools                 (optional — skip for single-turn)
├── eval.py           held-out benchmark, reusing reward.py
└── air/
    ├── 1_prep_data.yaml     CPU-ish, 1×A10
    ├── 3_baseline_eval.yaml the "before" number
    ├── 4_train.yaml         GRPO
    └── 5_eval.yaml          the "after" number
```

**Start by copying the closest shipped use case**, not from scratch:

```bash
cp -r usecases/agentic-search usecases/my-task    # rule-based reward, tools, retrieval
cp -r usecases/math           usecases/my-task    # LLM-judge reward, single tool
```

Then work through the five contracts below. Everything the engine needs from you is a
path in an env var — that is the whole integration surface:

| your file | env var the job sets |
|---|---|
| `reward.py` | `CUSTOM_REWARD_PATH` (+ `CUSTOM_REWARD_NAME`) |
| `tool.py` | `FUNCTION_TOOL_PATH` |
| `eval.py` | `EVAL_SCRIPT` |
| parquet from `prep_data.py` | `parameters.train_files` / `val_files` |

---

## 1. `prep_data.py` — data in verl's schema

One row per prompt, written as parquet to the Volume. The schema:

```python
{
  "data_source": "my_task",              # selects verl's built-in scorer when you have no custom one
  "agent_name": "tool_agent",            # routes to verl's ToolAgentLoop (multi-turn); omit for single-turn
  "prompt": [                            # RAW chat, not a rendered string
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user",   "content": question},
  ],
  "ability": "my_task",
  "reward_model": {"style": "rule", "ground_truth": gold},   # ground truth lives ONLY here
  "extra_info": {"split": "train", "index": i, "question": question, ...},
}
```

Rules learned the hard way:

- **Define `SYSTEM_PROMPT` here and import it everywhere else.** `eval.py` does
  `from prep_data import SYSTEM_PROMPT`, so training and eval cannot drift apart.
- **`ground_truth` belongs in `reward_model` only.** If it leaks into the prompt or
  `extra_info` in a way the model sees, you are measuring leakage.
- **`extra_info.index` must be unique** across merged sources.
- **Text-only data:** set `image_key: ''` in the job's `parameters:`, or the multimodal
  path silently re-enables.
- Write to a **new** directory so you never overwrite a dataset a previous run used.
- Pick difficulty deliberately: see §6.

Run it as a 1×A10 job (`1_prep_data.yaml`). If it needs no GPU-side libraries, use the
**stock** environment with a `dependencies:` list instead of the custom image — that is
what `usecases/math/air/1_prep_data.yaml` does.

---

## 2. `reward.py` — the task itself

```python
def compute_score(
    data_source: str = "",
    solution_str: str = "",       # the WHOLE trajectory: every assistant turn + tool response
    ground_truth=None,            # whatever you put in reward_model.ground_truth
    extra_info: dict | None = None,
    **kwargs,                     # always accept this
) -> dict[str, float]:
    ...
    return {"score": score, "answer_correct": ..., "well_formed": ...}
```

The contract:

- **Keyword arguments**, exactly those names — verl calls it that way.
- **Return a dict with `score`.** GRPO optimises `score`; **every other key becomes its
  own MLflow metric**, which is the only way to see "learning the answer" separately from
  "learning the output format". Use that liberally — it is your instrumentation.
- `solution_str` is the **entire episode** in multi-turn mode — the model's turns *and* the
  tool responses *and* chat-template text, in one string. Never search it for the answer or
  for "retrieved" text blindly: an `<answer>` inside a tool response is not the model's answer,
  and text the model typed can look like a tool response. The launchers run a role-span agent
  loop that records who wrote which characters (`extra_info["role_spans"]`, split with
  [`engine/lib/role_spans.py`](../engine/lib/role_spans.py)); `usecases/agentic-search/reward.py`
  reads the answer from assistant spans only and refuses to score without them.
- **Fail closed.** A judge outage or a parse failure must not return a passing score. A
  reward that fails *open* silently teaches the model that broken output is fine.
- Keep it **importable and testable on a CPU** with no torch: see
  `usecases/agentic-search/tests/test_reward.py`, which runs in well under a second under
  `make test` (the whole CPU suite). This is the fastest loop you have; use it before every
  training run.
- If the reward is an **LLM judge**, copy `usecases/math/reward.py`: it resolves the judge
  URL **at call time** from a rendezvous file (a Ray actor does not reliably inherit the
  driver's exports), is `async` so verl's `rate_limited` manager can run it concurrently,
  accepts only a strictly valid verdict, never raises (verl would turn an exception into a
  0.0 reward with a different key set), aborts the run when the judge keeps failing, and
  ships a calibration suite (`judge_selfcheck.py`, run via `PRE_TRAIN_CHECK`).

Graded reward? GRPO's std-normalisation (`NORM_ADV_BY_STD_IN_GRPO`, default `True`) keeps
your scores' order and relative gaps within a group; it only changes how groups are weighted
against each other. Whether `False` suits your reward is an empirical question — see
[tuning.md](tuning.md).

---

## 3. `tool.py` — what the agent can do (skip for single-turn)

```python
from verl.tools.function_tool import function_tool

@function_tool("my_search")
def my_search(query: str, top_k: int = 5) -> str:
    """One-line description the MODEL reads.

    Args:
        query: what to search for.
        top_k: how many results.
    """
    return "...text the model sees..."
```

- The **docstring and type hints generate the OpenAI tool schema** — the model only knows
  what you write there. Vague docstring, vague tool use.
- Tools must be **global and stateless**: verl offers them to every sample, and
  `tools_kwargs` is not threaded through. Per-call configuration comes from env vars
  (`QA_SEARCH_TOP_K`, `QA_VS_INDEX`, …), which also makes them tunable from the YAML.
- **Return a string, and cap its length.** `MAX_TOOL_RESPONSE_LEN` truncates at the
  engine level; clip inside the tool too (`QA_TOOL_MAX_CHARS`) so the model gets a clean
  message rather than a severed one.
- **Never raise.** Return an error string the model can react to; an exception costs you
  the sample.
- Make it runnable standalone (`python3 usecases/<uc>/tool.py`) for a local smoke.
- Talking to a Databricks service (Vector Search, SQL)? Inside a job in the same
  workspace you get **ambient auth** — no token, no `pip install` at query time. Verify
  with a tiny probe script before you pay for a GPU node.

Then, in the training job: `MULTI_TURN: 'True'`, `MAX_TURNS`, and **`TOOL_FORMAT`
matching what your model emits**. Confirm the format with
`infra/diagnostics/air/probe_tool_format.yaml` — the wrong parser means the model
silently never uses its tools, and the run looks healthy while learning nothing.

---

## 4. `eval.py` — the number you actually report

`engine/serve/serve_and_eval.sh` does the infrastructure: it serves `EVAL_MODEL_PATH`
with vLLM (staging it to NVMe first, because UC FUSE random-reads are slow), waits for
`/health`, puts your script's directory on `PYTHONPATH`, and runs it. Your script:

1. reads `EVAL_BASE_URL` + `EVAL_MODEL` (exported for you),
2. shows the model **exactly what training shows it** — the training prompt, and the tool
   schemas training renders, read from verl's `@function_tool` registry rather than copied —
   and parses its output with **verl's own parser** (`TOOL_FORMAT`) through its public
   `extract_tool_calls`, with no fallback,
3. runs its **own** agent loop over them and records how that loop differs from training's as
   a versioned `eval_policy` (both shipped evals: chat re-rendered each turn, a per-request token
   cap, one tool call per turn, optionally a forced final answer),
4. scores with **`import reward`** — the identical scorer training optimised,
5. writes `EVAL_OUT` (summary JSON) and `EVAL_TRACE_OUT` (per-question JSONL).

Two invariants that make the result trustworthy:

- **Same scorer for reward and eval** — one function (e.g. `score_segments`) called by both.
  That makes the *scoring* identical; the eval's agent loop is still its own code, with its own
  recorded policy (turns, per-request caps, forced final answer), so state what differs.
- **Same settings for baseline and trained.** The two eval jobs differ only in the model they
  serve and where they write (a test enforces it), and `scripts/paired_eval.py` refuses to pair
  artifacts whose `eval_policy` differs. The turn budget alone moves the number, so if
  `EVAL_MAX_TURNS` differs from training's `MAX_TURNS`, say why (math: 8 vs 4, because 4 cut
  the base model off before it boxed an answer).

Write the traces. A summary number tells you *whether* something changed; traces tell you
*why*, and they are what a diagnostic like
`usecases/agentic-search/analyze_traces.py` (`EM = P(S)·P(correct | S) + P(¬S)·P(correct | ¬S)`, a hypothesis generator) consumes.

---

## 5. The job files

Copy the four from your donor use case and change five things:

```yaml
code_source:
  snapshot:
    root_path: ../../..              # from usecases/<uc>/air/ that is the repo root
    include_paths:
      - engine
      - usecases/my-task             # <- yours

env_variables:
  FUNCTION_TOOL_PATH: ${CODE_SOURCE_PATH}/usecases/my-task/tool.py      # <- yours
  CUSTOM_REWARD_PATH: ${CODE_SOURCE_PATH}/usecases/my-task/reward.py    # <- yours
  CUSTOM_REWARD_NAME: compute_score
  EVAL_SCRIPT:        ${CODE_SOURCE_PATH}/usecases/my-task/eval.py      # <- eval jobs

parameters:
  train_files: /Volumes/.../data/my_task/train.parquet                  # <- yours
  val_files:   /Volumes/.../data/my_task/test.parquet
  output_dir:  /Volumes/.../ckpt/my-task-grpo
```

Also set `experiment_name` — it is how runs group in MLflow. Leave the topology block
(`TP`/`EP`/`GEN_TP`/…) alone unless you changed model or GPU count.

Judge-free task? Set `TRAINING_NODES` equal to the node count. Need a judge? Copy
`usecases/math/air/{2_stage_judge,4_train}.yaml` and give the job extra nodes.

---

## 6. Before you spend GPU hours — the checklist

```bash
# 1. the reward, on your laptop (uv needs no venv setup; plain `python3 -m pytest` works
#    too if pytest is installed)
uv run --with pytest --no-project python -m pytest usecases/my-task/tests/ -q

# 2. lint + the CPU suite + your job composed against the pinned verl + the real air CLI (free)
make check
make dry F=usecases/my-task/air/4_train.yaml

# 3. the resolved verl config, without a cluster
DRY_RUN=1 TRAINING_NODES=1 NUM_NODES=1 LOCAL_WORLD_SIZE=8 POD_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=1 RENDEZVOUS_ROOT=/tmp/rdv \
  MULTI_TURN=True FUNCTION_TOOL_PATH='${CODE_SOURCE_PATH}/usecases/my-task/tool.py' \
  CUSTOM_REWARD_PATH='${CODE_SOURCE_PATH}/usecases/my-task/reward.py' \
  bash engine/train/dispatch_agentic.sh

# 4. does your model even emit tool calls the way you think?
air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch
```

Then, in order (this is [running-jobs.md](running-jobs.md) §3–4 applied to your task):

1. `1_prep_data.yaml` — build the data.
2. `3_baseline_eval.yaml` — the **before** number, at final settings.
3. **Check the variance gate.** Is the base model in a band where the reward *separates*
   samples? If it scores ~95% or ~2%, stop and fix the data or the reward. This is the
   single most common way an RL project wastes a week: both shipped use cases changed
   dataset for exactly this reason (HotpotQA → MuSiQue; GSM8K → MATH).
4. `4_train.yaml` with a **small** `total_rollout_steps` and `SAVE_FREQ=1` — a smoke that
   proves tools fire, reward is called, checkpoints write.
5. The real run. Then evaluate **several** checkpoints, not just the last.

---

## 7. What you do not have to build

For perspective on the division of labour — none of this is in your five files:

35B MoE parallelism (EP/TP/PP/CP) · Megatron-FSDP vs classic + the offload decision ·
multi-node Ray bring-up and teardown · the fully-async Rollouter/Trainer split, the
MessageQueue and NCCL weight sync · verl's agent loop and tool-call parsing · the
checkpoint-based completion certificate + abort channel · serving a model for eval (NVMe staging, health
waiting) · co-locating and rendezvous-ing an LLM judge across nodes · the vLLM
custom-all-reduce workaround · image build, size gating, registration · MLflow wiring.

That is the point of the split: [`engine/`](../engine) is written once, and a use case
stays small enough to read in one sitting.
