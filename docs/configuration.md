# Configuration reference — every setting, where it lives, what it does

← [verl-on-air](../README.md) · [running-jobs](running-jobs.md) · [tuning](tuning.md) · [training-modes](training-modes.md)

This is the exhaustive list. If you want the *short* list of what actually matters
for learning, read **[tuning.md](tuning.md)** first and come back here for exact names
and defaults. For the per-flag verl/Megatron rationale (why `use_remove_padding=False`,
why `vanilla_mbridge`), see **[verl-config-reference.md](verl-config-reference.md)**.

---

## 1. How a setting reaches the training process

There are exactly **three** channels, and knowing which one a setting uses tells you
how to change it:

```
   air YAML                                  inside the job
 ┌──────────────────┐
 │ env_variables:   │ ──── process env ────>  engine/*.sh read  ${VAR:-default}
 │ parameters:      │ ──── YAML file ──────>  engine/lib/hparams.sh  hp <key> <default>
 │ compute:         │ ──── topology ───────>  NUM_NODES / LOCAL_WORLD_SIZE / POD_RANK
 │ code_source:     │ ──── file snapshot ──>  ${CODE_SOURCE_PATH}
 └──────────────────┘
```

**`env_variables:`** — the main surface. Every knob in this document that is spelled
`IN_CAPITALS` is one of these. They are plain strings: quote numbers (`'8'`) and write
booleans exactly as the script compares them (see the `True`/`1` gotcha in §11).

**`parameters:`** — air materialises this block as a **YAML file** at
`$HYPERPARAMETERS_PATH`, which `engine/lib/hparams.sh` reads with `hp <key> <default>`.
Use it for the run's *identity*: model, data, output dir, batch shape. `hp` preserves
the difference between *absent* (→ default) and *empty* (→ literally empty), which is
how `image_key: ''` means "this dataset is text-only, do not pass `data.image_key`".

**`compute:`** — `num_accelerators` is the **total GPU count**, not nodes. AI Runtime
derives nodes as `num_accelerators / 8` for `GPU_8xH100` and injects `NUM_NODES`,
`LOCAL_WORLD_SIZE`, `POD_RANK`, `MASTER_ADDR`, `MASTER_PORT`. The `command:` runs
**once per node**, which is why `engine/train/dispatch_agentic.sh` exists.

### `${CODE_SOURCE_PATH}` — the one expansion gotcha

air expands `${CODE_SOURCE_PATH}` in **`command:`** but **not** inside
`env_variables:`. So a job can write

```yaml
env_variables:
  FUNCTION_TOOL_PATH: ${CODE_SOURCE_PATH}/usecases/agentic-search/tool.py
```

and the variable arrives at the process *literally*, with the `${...}` unexpanded.
Running the launcher from `command:` does **not** help: the shell expands the
`command:` text, never the *contents* of an environment variable. So every engine
entrypoint resolves the path values it reads with one shared helper,
`resolve_code_path` in [`engine/lib/paths.sh`](../engine/lib/paths.sh): it substitutes
the two spellings of `CODE_SOURCE_PATH` (the value is data — nothing is `eval`'d) and
anchors a relative path at the code snapshot, falling back to the repo root when
`CODE_SOURCE_PATH` is unset, which is why the same YAML values work for a local
`DRY_RUN=1`. The resolved file must exist, or the job stops before doing anything
expensive:

| variable | resolved by | checked before |
|---|---|---|
| `FUNCTION_TOOL_PATH`, `CUSTOM_REWARD_PATH`, `TOOL_CONFIG_PATH`, `AGENT_LOOP_CONFIG_PATH` | `engine/train/dispatch_agentic.sh` | the role split / Ray |
| `EVAL_SCRIPT` | `engine/serve/serve_and_eval.sh` | model staging and vLLM start |

A new path-valued variable needs the same treatment — add it to one of those two lists.

### Overriding anything at submit time, without editing a file

```bash
air run --file usecases/math/air/5_eval.yaml -p df1 \
  --override env_variables.EVAL_LIMIT=0 \
             env_variables.EVAL_MODEL_PATH=/Volumes/.../qwen3_5-35b-math-rl/<RUN_ID>/global_step_24 \
             parameters.actor_lr=3e-6 \
             compute.num_accelerators=16 \
             timeout_minutes=240
```

Dotted paths address any field in the YAML. This is the right way to sweep a knob:
the file stays the documented default, the override records the experiment.

---

## 2. Job-file anatomy

Every one of the 27 job files has the same shape:

| field | meaning | notes |
|---|---|---|
| `experiment_name` | job name + MLflow experiment | keep it stable; it is how runs group |
| `mlflow_experiment_directory` | *optional* — group experiments under one Workspace folder | must start `/Workspace`; unset (as shipped) means your own per-user default |
| `compute.num_accelerators` | **total GPUs** | `16` = 2 nodes of `GPU_8xH100` |
| `compute.accelerator_type` | `GPU_1xA10` · `GPU_1xH100` · `GPU_8xH100` | df1 offers these three |
| `environment.docker_image.url` | the registered custom image | must be **registered** (`make register`) or submit fails |
| `environment.version` + `dependencies` | *stock* runtime instead of a custom image | used by CPU-ish prep jobs (`usecases/math/air/1_prep_data.yaml`) |
| `code_source.snapshot.root_path` | snapshot root, **relative to the YAML's own location** | `../../..` from `usecases/<uc>/air/` = repo root |
| `code_source.snapshot.include_paths` | what to upload | `[engine, usecases/<uc>]` — keep it tight, it is uploaded per submit |
| `env_variables` | the knobs (this document) | strings only |
| `parameters` | run identity + batch shape | read via `hp` |
| `max_retries` | air-level retry | `0` for training (a failed 16-GPU run should not silently re-bill); `>0` for idempotent staging |
| `timeout_minutes` | hard wall | **includes queue time for GPU capacity** — see §12 |
| `command` | what runs on **every** node | `bash ${CODE_SOURCE_PATH}/engine/...` |

`code_source` is why iteration is fast: a launcher or reward edit ships with the next
submit and needs **no image rebuild**. Only changing the *installed stack* (a pip pin,
a system package) requires `make bump && make release`.

---

## 3. `parameters:` — run identity and batch shape

| key | read by | default | what it is |
|---|---|---|---|
| `model_name` | both launchers | `Qwen/Qwen3.5-35B-A3B` (sync) / `Qwen/Qwen3.5-9B` (async) | HF repo id **or** a Volume path. Use a staged Volume path for anything big. |
| `train_files` / `val_files` | both | geo3k parquet | verl parquet, Volume paths |
| `output_dir` | both | `…/ckpt/default` | the experiment's root: each run writes to `<output_dir>/<RUN_ID>/` (`trainer.default_local_dir`), checkpoints at `<output_dir>/<RUN_ID>/global_step_N/actor/model/huggingface/`, plus `run_manifest.json` / `run_result.json` |
| `total_epochs` | both | `1` | passes over the data |
| `train_batch_size` | **sync only** | `32` | prompts per GRPO step. Must satisfy `train_batch_size × rollout_n % trainer_GPUs == 0` — the launcher checks and fails with the arithmetic |
| `ppo_mini_batch_size` | both | `32` (sync) / `16` (async) | optimizer sub-batch; must divide trainer DP |
| `rollout_n` | both | `5` (sync) / `4` (async) | **GRPO group size** — samples per prompt. This is where advantage variance comes from |
| `total_training_steps` | **sync only** | `3` | step cap; **`0` disables it** (the rungs ship `3` as a smoke) |
| `total_rollout_steps` | **async only** | `64` | total rollout **samples** for the run — the async horizon |
| `max_prompt_length` | both | `1024` | the initial prompt's budget (multi-turn: also a sizing unit — §6) |
| `max_response_length` | both | `2048` | single-turn: the response cap. Multi-turn: **not a per-turn cap** — only a sizing unit for the one episode-wide budget `rollout.response_length = (max_prompt_length + max_response_length) × MAX_TURNS − max_prompt_length` (§6), which any one turn may use up. The evals' per-request cap is `EVAL_MAX_TOKENS`, an eval-only choice |
| `actor_lr` | both | `1e-6` | policy learning rate |
| `image_key` | **sync only** | `images` | set `''` for text-only data, or the multimodal path silently re-enables |
| `project_name` / `experiment_name` | both | `verl-on-air` / `grpo-*` | MLflow; `PROJECT_NAME`/`EXPERIMENT_NAME` env override them |

---

### Run identity — set by `make`, required for training

| var | default | what it does |
|---|---|---|
| `RUN_ID` | **required** (`make` sets `<UTC time>-<commit>`) | the run writes to `<output_dir>/<RUN_ID>/`; also keys the rendezvous directory and names eval artifacts. Hand submission: `--override env_variables.RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` |
| `RESUME` | `never` | `never`: a fresh run (refuses if `<output_dir>/<RUN_ID>/` already holds checkpoints); `auto`: continue that run from its latest checkpoint; `<path>/global_step_N`: start from that checkpoint. verl's own default (`resume_mode=auto` on a shared dir) is never used implicitly |
| `MAX_CKPT_TO_KEEP` | keep all | let verl delete all but the N newest checkpoints. Off by default so several can be evaluated |
| `GIT_SHA` / `VOA_IMAGE` | set by `make` | the commit (`-dirty` if tracked files changed) and image the run was submitted from; recorded in `run_manifest.json` and in eval artifacts |

## 4. Mode and node roles — `engine/train/dispatch_agentic.sh`

| var | default | what it does |
|---|---|---|
| **`TRAIN_MODE`** | `async` | **the mode switch**: `async` → `run_grpo_fully_async.sh`, `sync` → `run_grpo_megatron.sh`. See [training-modes.md](training-modes.md) |
| **`TRAINING_NODES`** | `2` | ranks `[0, TRAINING_NODES)` train; the rest serve the judge. Set it **equal to the node count for a judge-free run** |
| `RENDEZVOUS_ROOT` | `/Volumes/main/mshtelma/verl/rendezvous` | UC dir for the rendezvous files, one subdir per `RUN_ID`: judge URL and head address, `training_done`, `ABORT.json`, and the training head's `ray_head_alive` heartbeat / `ray_head_done` (`rc=<code>[ signal=<name>]`; a worker exits with the head's verdict: 0 only for `rc=0`). Writes are atomic; a wait only accepts a file written during this job (`RDV_SKEW_S`, default 300 s of clock skew), so a resumed run never picks up the last attempt's judge |
| `RAY_NODES_TIMEOUT_S` / `RAY_HEARTBEAT_STALE_S` | `900` / `600` | the training head waits this long for every node's GPUs; a worker whose head has not heartbeaten for this long exits 1 instead of idling until the job timeout (a head killed without its cleanup trap leaves Ray's port open) |
| `JUDGE_STAGE_TIMEOUT` / `JUDGE_HEALTH_TIMEOUT` | `3600` / `2400` | the judge's two phases: copying its weights to local NVMe, then loading until `/health` answers. Each fails the judge when exceeded |
| `JUDGE_WAIT_TIMEOUT` | stage + health + `600` | how long training waits for the judge endpoint — derived from the two above so the sides agree; a smaller explicit value is refused |

The dispatcher also **derives `PYTHONPATH`** from the resolved `CUSTOM_REWARD_PATH` and
`FUNCTION_TOOL_PATH` directories, which is what lets `reward.py` and `tool.py` import
each other by bare name (`import reward`) in training *and* in eval.

---

## 5. Topology and parallelism

Read by both launchers. These follow from **model + GPU count**, not from your task —
change them only when you change one of those. The arithmetic is in [sizing.md](sizing.md).

| var | default | meaning |
|---|---|---|
| `TP` | `1` fsdp / `2` classic (sync), `2` (async) | tensor parallel (must divide the model's 16 Q heads) |
| `PP` | `1` | pipeline parallel |
| `CP` | `1` | context parallel |
| **`EP`** | `8` (sync) / `1` (async default; use cases set `8`) | **expert parallel — the dominant lever for this MoE** (92.5% of weights are routed experts) |
| `ETP` | `1` | expert tensor parallel |
| `GEN_TP` | `8` (sync) / `1` (async default; use cases set `8`) | vLLM rollout TP. Keep ≤ GPUs-per-node so rollout TP stays on NVLink |
| `NNODES` / `NGPUS_PER_NODE` / `NODE_RANK` | from `NUM_NODES` / `LOCAL_WORLD_SIZE` / `POD_RANK` | injected; override only for local dry runs |
| `MEGATRON_MODE` | `fsdp` | **sync only**: `fsdp` (ZeRO-3, shards params+grads+optimizer) or `classic` (ZeRO-1, replicates params+grads) |
| `OFFLOAD` | `auto` | **sync only**: `auto` picks `0` at ≥16 GPUs for fsdp / ≥32 for classic, else `1`. CPU offload is **incompatible with Megatron-FSDP** (crashes on DTensors, `aten.is_pinned`) |
| `OFFLOAD_FRACTION` | `1` | fraction of optimizer state offloaded. `OFFLOAD=1` needs **~400–500 GB host RAM per node** |
| `MAX_MODEL_LEN` | `8192`, or the computed episode length when multi-turn | vLLM context cap. Unset, vLLM sizes KV for the model's 262144 config max (~3 GiB KV/request) and fails |
| `ROLLOUT_GPU_MEM_UTIL` | `0.8` (async) / `0.6` (sync) | vLLM `gpu_memory_utilization`. This sizes the **KV cache only** — it is *not* the lever for a weight-sync OOM (see [sizing.md](sizing.md)) |
| `ROLLOUT_ENFORCE_EAGER` | `False` (async) / `0` (sync) | skip vLLM CUDA-graph capture: saves a few GiB, costs generation speed. **Note the different spelling per launcher** (§11) |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `False` | **async only**: NCCL instead of vLLM's custom all-reduce, *keeping* CUDA graphs. Both use cases set `True` — at intra-node `GEN_TP≤8` the custom kernel crashes graph capture on H100 |
| `ROLLOUT_PREFIX_CACHING` | `False` | vLLM prefix cache. Big multi-turn win (shared system prompt + prior turns); safe because verl flushes the cache on weight sync |
| `ROLLOUT_TEMP` | `1.0` | rollout sampling temperature. Higher = more diverse group = denser reward signal |
| `WEIGHT_BUCKET_MB` | unset | **sync only**: weight-sync bucket size. Any bucket must exceed the ~970 MiB embedding; the config path moved between verl releases — read [troubleshooting.md](troubleshooting.md) first |
| `CUDA_DEVICE_MAX_CONNECTIONS` | set by the launcher | `1` for classic (comm/compute overlap); **must be unset for Megatron-FSDP** or the collectives serialise behind compute. The launcher handles this — do not set it in a YAML |

---

## 6. Fully-async knobs (`run_grpo_fully_async.sh`)

| var | default | meaning |
|---|---|---|
| **`ROLLOUT_NNODES`** | `0` | `≥1` = **whole-node split**: the Rollouter gets that many entire nodes, the Trainer the rest. `0` = within-node split |
| `N_GPUS_ROLLOUT` | `4` | GPUs given to the Rollouter when `ROLLOUT_NNODES=0` (single-node split) |
| **`STALENESS`** | `0.1` | `async_training.staleness_threshold`. `0` makes the Trainer wait (synchronous behaviour); `>0` lets the Rollouter run ahead |
| **`TRIGGER_SYNC_STEP`** | `2` | local optimizer updates between weight syncs |
| `REQUIRE_BATCHES` | `1` | mini-batches fetched per update |
| `PARTIAL_ROLLOUT` | `True` | keep partially-generated sequences across a weight sync instead of discarding them |
| `LR_DECAY_STEPS` | `= total_rollout_steps` | **must be explicit**: streaming has no dataloader, so verl cannot derive a step count and Megatron's scheduler asserts |

**The one piece of arithmetic to internalise:**

```
samples between weight syncs = TRIGGER_SYNC_STEP × REQUIRE_BATCHES × ppo_mini_batch_size
total weight syncs           = total_rollout_steps / (that number)
```

and in fully-async mode **`SAVE_FREQ` counts weight syncs (parameter versions)**, not
optimizer steps and not samples. The launcher prints all of this at startup — read that
banner before you walk away from a run.

---

## 7. Agentic / multi-turn tool calling

Read by both launchers (`MULTI_TURN=False` → single-turn, and none of the rest applies).

| var | default | meaning |
|---|---|---|
| **`MULTI_TURN`** | `False` | `True` turns on verl's `ToolAgentLoop`: the model emits tool calls, verl executes them and feeds results back |
| **`MAX_TURNS`** | `4` | max assistant *and* user turns — the agent's tool budget. For a retrieval agent this is the **hop budget** and a top-tier tuning knob |
| **`FUNCTION_TOOL_PATH`** | — | python file of stateless `@function_tool` callables, offered to every sample |
| `TOOL_CONFIG_PATH` | — | YAML of stateful `BaseTool` classes (alternative to the above) |
| `AGENT_LOOP_CONFIG_PATH` | — | YAML registering a **custom agent loop**; the data's `agent_name` column routes to it |
| **`TOOL_FORMAT`** | `hermes` | the tool-call **parser**. Qwen3.5 emits XML → **`qwen3_coder`**. A wrong value silently zeroes all tool use — verify with `infra/diagnostics/air/probe_tool_format.yaml` |
| `AGENT_NUM_WORKERS` | `8` | parallel `AgentLoopWorker` actors |
| `MAX_TOOL_RESPONSE_LEN` | `512` | per-tool-response token cap. Retrieval returns passages → agentic-search raises this to `4000` |

**Episode-length arithmetic** (both launchers compute this identically):

```
episode_len   = (max_prompt_length + max_response_length) × MAX_TURNS
resp_budget   = episode_len − max_prompt_length        # what verl trains on
MAX_MODEL_LEN = episode_len                            # vLLM must hold the whole thing
```

The actor trains on the **whole trajectory** — every assistant turn plus every tool
response — so raising `MAX_TURNS` raises the memory cost of the backward pass. At
`ppo_micro_batch_size_per_gpu=1`, `MAX_TURNS` is effectively the primary backward-memory
knob: turn budgets of 32 and 80 OOM'd in the actor backward on a config where 16 held.

---

## 8. Reward

| var | default | meaning |
|---|---|---|
| **`CUSTOM_REWARD_PATH`** | — | your `reward.py`. verl imports it; its directory goes on `PYTHONPATH` |
| `CUSTOM_REWARD_NAME` | `compute_score` | the function name inside that file |
| **`REWARD_MANAGER`** | verl's default | `naive` (rule-based, in-process) · `rate_limited` (**async**, for an LLM judge) · `dapo`. **async launcher only** |
| `REWARD_MAX_CONCURRENT` | **1 inside verl** | concurrent reward calls. The verl default of 1 means *serial* — **always set this** for a judge (the use case sets `64`) |
| `REWARD_MAX_RPM` / `REWARD_MAX_TPM` | — | request/token rate caps for an external judge API |
| `REWARD_TIMEOUT` | — | per-call timeout, seconds |
| `REWARD_SOURCE` | `judge` (math) | use-case-level: optimise the judge score vs the rule. Read by `usecases/math/reward.py`, not the engine |
| **`NORM_ADV_BY_STD_IN_GRPO`** | verl default `True` | `True` or `False` (anything else stops the launcher), both modes. Whether GRPO divides each group's advantages by the group's reward std. That keeps a graded reward's order and relative gaps — `[0, 0.05, 0.7, 1]` becomes `[−0.89, −0.79, 0.53, 1.14]` — but weights a low-spread group as heavily as a high-spread one; `False` keeps advantages in reward units. An empirical choice ([tuning.md](tuning.md)); the math job sets `True` explicitly, as run |

verl calls your function with keyword args
`(data_source=, solution_str=, ground_truth=, extra_info=)` and expects a **dict** with
a `score` key. Every other key becomes its own MLflow metric — which is how you see
"learning the answer" separately from "learning the output format".

---

## 9. Checkpointing, logging, validation

| var | default | meaning |
|---|---|---|
| **`SAVE_FREQ`** | `-1` (never) | checkpoint interval. **Counts weight syncs in async, optimizer steps in sync.** Pick a divisor of the run's total so the last one is a save point |
| `TEST_FREQ` | `-1` (never) | in-loop validation interval. Both use cases keep this off and evaluate saved checkpoints out of band with the eval job — which is also what makes base-vs-trained apples-to-apples |
| `VAL_BEFORE_TRAIN` | `False` | **sync only**: run validation before training starts |
| `PROJECT_NAME` / `EXPERIMENT_NAME` | from `parameters` | MLflow project/run naming |
| `USE_DIST_CKPT` / `DIST_CKPT_PATH` | `False` | sharded Megatron dist-checkpoint instead of the full-gather HF export. Not needed at 35B; a seam for much larger models. Note it also switches **init** to load from that path |
| `DRY_RUN` | `0` | `1` prints the fully-resolved verl invocation and exits **before** any Ray bootstrap. Works on a laptop; the cheapest possible config check |

---

## 10. Judge server and client (`engine/serve/serve_judge.sh`, `usecases/math/reward.py`)

Only relevant to the judge-reward pattern. Server side, on the judge ranks:

| var | default | meaning |
|---|---|---|
| `JUDGE_ENGINE` | `sglang` | `vllm` or `sglang`. The math use case sets `vllm` to ride the training image. A multi-node judge must be `vllm`: SGLang's own multi-node launch is not implemented, so it is refused |
| `JUDGE_MODEL_PATH` / `JUDGE_MODEL_ID` | one is **required** | a staged Volume dir, or an HF repo id |
| `JUDGE_TP` | `8` | tensor parallel across the judge nodes (`8 × JUDGE_NODES`) |
| `JUDGE_SERVED_NAME` | `judge` | the model name the client asks for |
| `JUDGE_PORT` | `8000` | serving port |
| `JUDGE_GPU_MEM_UTIL` | `0.90` | memory fraction |
| `JUDGE_MAX_MODEL_LEN` | `16384` | judge context: prompt + the trajectory it grades |
| `JUDGE_STAGE_TIMEOUT` | `3600` | the copy to `JUDGE_LOCAL_CACHE` must finish within this, or the judge fails (exit 1) |
| `JUDGE_HEALTH_TIMEOUT` | `2400` | then wait for `/health`; a 744 GB first load is slow |
| `JUDGE_LOCAL_CACHE` | unset | NVMe dir (e.g. `/local_disk0/judge_cache`) to bulk-copy the model off UC FUSE first — much faster than random-reading FUSE |
| `JUDGE_STAGE_PARALLEL` | `8` | parallel copies during that staging |
| `JUDGE_RAY_VERSION` | unset | pin Ray for multi-node serving (`2.48.0`; see the `probe_vllm_multinode` diagnostic) |
| `JUDGE_RAY_PORT` | `6380` | deliberately **not** 6379 — training's Ray owns that |
| `JUDGE_MAX_LIFETIME` | — | self-exit guard, seconds. The judge's exit status says why it stopped: `0` training signalled done (the expected end), `1` it never became healthy, `3` the server died while serving, `4` this lifetime ran out |
| `JUDGE_EXTRA_ARGS` | — | engine passthrough, e.g. `--reasoning-parser glm45 --tool-call-parser glm47` |
| `JUDGE_RENDEZVOUS` / `JUDGE_EXIT_SENTINEL` | set by the dispatcher | where to publish the endpoint / when to shut down |
| `STAGE_ONLY` | `0` | stage the weights and exit without serving |

Client side (runs inside the reward actors):

| var | default | meaning |
|---|---|---|
| `JUDGE_BASE_URL` / `JUDGE_ENDPOINT_FILE` | set by the dispatcher | the endpoint, or the rendezvous file to read it from. The reward resolves the URL **at call time** because Ray does not reliably carry a driver `export` into actor processes |
| `JUDGE_MODEL` | `judge` | served model name |
| `JUDGE_MAX_TOKENS` | `2048` | judge response budget; a verdict cut off by it (`finish_reason=length`) is **invalid**, never graded |
| `JUDGE_TIMEOUT` | `60` | per-attempt HTTP timeout |
| `JUDGE_RETRIES` / `JUDGE_BACKOFF_S` | `2` / `2` | retries for **transient** failures only (connection, timeout, HTTP 429/5xx) |
| `JUDGE_DEADLINE_S` | `REWARD_TIMEOUT − 10` | total budget for one verdict, retries included — must stay below `REWARD_TIMEOUT`, whose expiry replaces the sample's result with a different key set and breaks the batch |
| `JUDGE_TEMPERATURE` | `0` | deterministic grading |
| `JUDGE_DISABLE_THINKING` | `1` | some reasoning models think unconditionally at high effort and blow the parse rate |
| `JUDGE_STRUCTURED_OUTPUT` | `1` | ask vLLM for grammar-constrained JSON (`response_format` json_schema) so LaTeX in `reason` cannot break parsing |
| `JUDGE_TRAJECTORY_CHARS` | `36000` | trajectory budget (~10k tokens); longer working keeps head + tail with a marked cut and logs `judge_input_truncated=1` |
| `JUDGE_FALLBACK` | `rule` | score for a sample with no valid verdict: `rule` (exact match) or `zero`; flagged `judge_fallback=1` |
| `JUDGE_MAX_FAIL_RATE` / `JUDGE_FAIL_WINDOW` / `JUDGE_FAIL_MIN_CALLS` | `0.05` / `200` / `50` | per reward worker: more than 5% of its last 200 calls without a valid verdict **aborts the run** (the abort channel, see `engine/lib/run_control.py`); `1` disables |
| `JUDGE_BLEND_ALPHA` | `0.5` | judge/rule blend weight when `REWARD_SOURCE=blend`; must be in [0, 1] |
| `JUDGE_API_KEY` | `EMPTY` | self-hosted endpoints need no real key |
| `JUDGE_DEBUG` | `0` | log each verdict (`1`/`true` only — `0` is off) |
| `PRE_TRAIN_CHECK` | — | a script the dispatcher runs on training rank 0 once the judge is up, before training (math: `judge_selfcheck.py`); non-zero exit stops the job |

What the reward logs is only meaningful per **valid** verdict: judge coverage is
`judge_valid`; the judge's mean score is `mean(judge_score) / mean(judge_valid)` and its
agreement with the gold answer is `mean(judge_agree) / mean(judge_valid)` (fallback samples
count in neither numerator nor denominator).

---

## 11. Checked before anything runs

Both launchers run [`engine/lib/preflight.py`](../engine/lib/preflight.py) twice: once before they
compute anything from a knob, and again once the geometry is resolved. `DRY_RUN` runs both, so a
dry run checks meaning, not only spelling. `make preflight F=<job.yaml>` does the same on your
machine and prints the job's plan (roles, parallelism, budget, checkpoints) and the upper bound
on the GPU-hours it can bill (GPUs × timeout).

- **Every engine knob is typed**: integers and their bounds, and enums such as `TOOL_FORMAT`,
  `MEGATRON_MODE` and `REWARD_MANAGER`. A bad value stops the job with the reason.
- **Booleans mean what they say.** `true`/`True`/`1`/`yes`/`on` and
  `false`/`False`/`0`/`no`/`off` are normalised to `True`/`False` for both launchers. Before,
  `MULTI_TURN=true` meant off, and the sync launcher wanted `1` for `ROLLOUT_ENFORCE_EAGER`.
  Anything else is an error.
- **A knob only the other mode reads is an error, not a no-op**: `STALENESS` on a sync job,
  `MEGATRON_MODE` on an async one. So is an unknown name with an engine prefix (`ROLLOUT_`,
  `REWARD_`, `AGENT_`, `TOOL_`, …), so the typo `ROLLOUT_TEMPERATURE` for `ROLLOUT_TEMP` is
  caught.
- **The geometry must fit the model**, taken from its `config.json` or from the table of shipped
  models:
  - TP divides every head count (attention, KV and linear attention), PP divides the layers, and
    EP divides the experts;
  - `DP = trainer GPUs / (TP·PP·CP)` is a whole number, and `EP·ETP·PP` divides the trainer GPUs;
  - `GEN_TP` divides the attention heads and the rollout GPUs;
  - the mini-batch splits evenly over DP;
  - Megatron-FSDP never runs with CPU offload, which crashes it — and `OFFLOAD=auto` no longer
    picks that combination.
- **A malformed `parameters:` block stops the job**, instead of every key silently taking its
  default.

---

## 12. Job-level settings that bite

- **`timeout_minutes` includes queue time.** A job waiting for GPU capacity is
  burning its own timeout — a 30-minute smoke that queues 27 minutes for an A10 dies
  `TIMEDOUT` having barely run. Size it generously for scarce accelerator types.
- **`max_retries: 0` for training.** A retry re-bills a multi-node job. Keep retries
  for idempotent, resumable work (model staging skips already-complete shards).
- **Snapshot size.** `include_paths` is uploaded per submit — list only `engine` and
  the one use case, never the repo root.
- **Image tag registration is per tag.** Re-pushing the *same* tag keeps serving the
  already-registered digest, so your change appears not to take effect. After any
  Dockerfile change: `make bump && make release` (`make stale-check` enforces this).

---

## 13. Use-case settings (not engine settings)

These are read by the use-case Python, so they are yours to define when you write a
new use case. Listed here because you need them to *run* the shipped ones.

### agentic-search

| var | default | read by |
|---|---|---|
| `QA_VS_ENDPOINT` | — | `tool.py`, `create_vs_index.py` — Vector Search endpoint name |
| `QA_VS_INDEX` | — | `tool.py` — full index name `catalog.schema.index` |
| `QA_VS_TABLE` / `QA_VS_CATALOG` / `QA_VS_SCHEMA` | `wiki_qa_big_corpus` / `main` / `mshtelma` | `create_vs_index.py` — the table's **base** name: a build creates `<base>_v<h8>` and the index `<base>_v<h8>_index`, `h8` = the corpus content hash, so a new corpus never touches a live table |
| `QA_VS_WAREHOUSE_ID` | **required** | `create_vs_index.py` — the SQL warehouse that loads the table (none is guessed): `make search-index WAREHOUSE_ID=<id>` |
| `QA_VS_CREATE_ENDPOINT` | `0` | `create_vs_index.py` — `1` creates a missing endpoint (billable, persistent); otherwise a missing endpoint is an error, so a typo cannot start one |
| `QA_VS_SQL_TIMEOUT_S` / `QA_VS_WAIT_TIMEOUT_S` | `1800` / `3000` | `create_vs_index.py` — per-statement deadline (then cancelled) / how long `--wait-only` waits |
| `QA_VS_EMBED_MODEL` | `databricks-gte-large-en` | `create_vs_index.py` — managed embedding model |
| `QA_VS_TEXT_COL` / `QA_VS_TITLE_COL` / `QA_VS_ID_COL` | `text` / `title` / `id` | `tool.py` — index column names |
| `QA_SEARCH_TOP_K` | `5` | `tool.py` — hits per search call |
| `QA_SNIPPET_CHARS` / `QA_TOOL_MAX_CHARS` | `600` / `4000` | `tool.py` — snippet and total tool-response caps |
| `QA_REWARD_METRIC` | `em` | `reward.py` + `eval.py` — the same scorer (`score_segments`) reads it in both |
| `QA_RETRIEVAL_BONUS` | `0.0` | `reward.py` — bonus when a retrieved passage held the gold. Measured inert here (RESULTS.md) |
| `QA_FORMAT_SCORE` | `0.0` | `reward.py` — credit for well-formed output alone |
| `QA_DATASETS` / `QA_CORPUS_DATASETS` / `QA_CORPUS_SPLITS` | `musique` / `musique,hotpotqa` / `train,validation` | `prep_data.py`, `build_corpus.py` — sources read at the commits pinned in `prep_data.SOURCES`. A corpus source that fails **fails the build**; `build_corpus.py --allow-partial` writes one its manifest marks incomplete |
| `QA_PREP_TRAIN_LIMIT` / `QA_PREP_VAL_LIMIT` | `0` (all) / `500` | `prep_data.py` |
| `QA_VAL_PARQUET` | `…/qa_musique/test.parquet` | `eval.py` — **override it to your data dir** |
| `QA_HF_CACHE` | `/local_disk0/hf_cache` | prep jobs — NVMe, not FUSE |

### math

| var | default | read by |
|---|---|---|
| `MATH_LEVELS` | all | `prep_data.py` — `3,4,5` keeps the learnable band |
| `MATH_TOOL_OUT_DIR` | `~/data/math_tool` | `prep_data.py` |
| `N_TRAIN` / `N_TEST` | `0` (all) | `prep_data.py` |
| `MATH500_ID` / `MATH500_REVISION` | `HuggingFaceH4/MATH-500` / its pinned commit | `eval.py` — pointing `MATH500_ID` elsewhere requires a 40-hex `MATH500_REVISION` |

### Dataset sources (every prep job, and the math eval sets)

Every Hub dataset is read at a pinned commit (`engine/lib/data_manifest.py`), never a moving
branch. Each prep job writes `DATA_MANIFEST.json` beside its outputs — the corpus writes
`<stem>.manifest.json` — recording sources and revisions, row counts before and after each
filter, the sampling, and every output's sha256. Eval artifacts and `run_manifest.json` carry
that provenance for the files they read, and whether the file still matches its manifest.

| var | default | meaning |
|---|---|---|
| `ALLOW_FALLBACK_SOURCE` | `0` | `1` lets a listed mirror stand in for an unavailable pinned source — accepted **only** if its content matches the pinned data (math prep: the problem/solution digest; AIME: the 30 answers). Without it, an unavailable source is an error, never a silent substitute |

### Eval harness (`engine/serve/serve_and_eval.sh` + any `eval.py`)

| var | default | meaning |
|---|---|---|
| **`EVAL_SCRIPT`** | **required** | absolute path to the use case's `eval.py`; its directory goes on `PYTHONPATH` so eval imports the *same* `reward.py` training used |
| `EVAL_MODEL_PATH` | base model (baseline job); **none** (checkpoint job) | **what to serve**: a model dir, or a checkpoint's `global_step_N` (its HF export is found and verified — verl's completion manifest plus every indexed shard — before anything is staged) |
| `MODEL_PATH` | derived | the eval client's **tokenizer** path — set automatically to the served model; setting it to anything else is an error |
| `EVAL_CKPT_ROOT` | the run's `output_dir` | where the checkpoint job lists complete steps when `EVAL_MODEL_PATH` is missing |
| `EVAL_TP` | `8` | serving tensor parallel |
| `EVAL_SERVE_LEN` | `8192` | served context. Must cover the whole multi-turn episode |
| `EVAL_GPU_UTIL` | `0.85` | vLLM memory fraction |
| `EVAL_STAGE` | `1` | bulk-copy the model UC→NVMe before serving (FUSE random-read is slow) |
| `EVAL_LOCAL_CACHE` | `/local_disk0/eval_model` | that NVMe destination |
| `EVAL_HEALTH_TIMEOUT` | `1800` | wait for `/health` |
| `EVAL_PORT` / `EVAL_MODEL` | `8000` / `eval` | endpoint and served name |
| `EVAL_SERVE_EXTRA_ARGS` | — | extra `vllm serve` flags |
| **`EVAL_MAX_TURNS`** | `8` | eval-time turn budget. **Identical between the baseline and the trained eval** (a test enforces it). Search uses training's `12`; math uses `8` against training's `4`, deliberately — recorded in `eval_policy` |
| `EVAL_FORCE_FINAL_ANSWER` | `1` (search) | on the last turn, tell the model to answer and start its reply with `<answer>` — training has no such turn; `0` makes the last turn an ordinary one. Recorded in `eval_policy` |
| `TOOL_FORMAT` | `qwen3_coder` | verl's parser for the model's tool calls — the eval jobs name the training job's. The tool schemas come from verl's `@function_tool` registry, never a copy; without verl the eval does not start |
| `EVAL_LIMIT` | `0` (all) | number of questions; small values are a cheap harness smoke |
| `EVAL_MAX_TOKENS` | `512` (search) / `1024` (math) | per-**request** cap (math jobs set `3072`) — eval-only: training has one episode-wide budget and no per-turn cap |
| `EVAL_TEMPERATURE` | `0` | greedy, so the comparison is deterministic |
| `EVAL_CONCURRENCY` | `32` | parallel in-flight questions |
| `EVAL_MAX_CONT` | `2`/`3` | continuation attempts on a truncated answer |
| `EVAL_REQ_TIMEOUT` / `EVAL_HTTP_RETRIES` | `600`–`900` / `4` | per-request timeout; retries for **transient** failures only (connection, timeout, HTTP 429/5xx) — a bad request is not retried |
| `EVAL_EXPECT_N` | `EVAL_LIMIT` if set | the number of questions the run must load; anything else makes it **invalid** (math ships `500`) |
| `EVAL_MAX_INFRA_ERRORS` | `0` | questions that may hit an infrastructure failure (inference, retrieval, tool, context limit) before the run is **invalid**. Those questions are never scored |
| `EVAL_OUT` | — | JSON summary path. **Never overwritten** (`EVAL_OVERWRITE=1` to force); carries `valid`, the served model's identity, the dataset fingerprint and the versioned `eval_policy`. Each finished question is also written under `<EVAL_OUT>.parts/` |
| `EVAL_TRACE_OUT` | — | per-question JSONL traces — **required input for `analyze_traces.py`** |

Every eval follows [`engine/serve/eval_contract.py`](../engine/serve/eval_contract.py): readiness
(the served model listed at `/v1/models`, and for search one real retrieval) is checked before the
first question, an outage is recorded as infrastructure instead of a wrong answer, and an **invalid**
run still writes its artifact — marked `"valid": false` with the reasons — and exits non-zero.
Compare only valid artifacts.

### Model staging (`engine/stage_model.py`)

| var | default | meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | HF repo to stage |
| `MODEL_REVISION` | `main` | branch/tag/commit, resolved once to a commit that every file is fetched at; recorded in `<MODEL_DIR>/STAGED.json`. A directory holding another revision is refused. Pin a commit for reproducible re-stages |
| `MODEL_DIR` | `…/verl/models/<basename>` | Volume destination |
| `SCRATCH_DIR` | `/local_disk0/hf_stage` | NVMe staging dir before the FUSE copy |
| `HF_TOKEN` | — | gated repos / rate limits |
| `HF_HUB_DISABLE_XET` | — | set `1` for UC Volumes: FUSE rejects Xet/CAS parallel range writes |

### geo3k infra jobs

| var | default | meaning |
|---|---|---|
| `GEO3K_OUT_DIR` | `…/data/geo3k` | prep destination |
| `N_TRAIN` / `N_TEST` | `64` / `128` | subset size (**`0` = full split**); the variance gate needs `N_TEST >= N_PROMPTS` |
| `EVAL_FILE`, `N_PROMPTS`, `N_SAMPLES`, `TEMPERATURE`, `GEN_TP`, `MAX_TOKENS`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | see `infra/geo3k/baseline_eval.py` | the reward-variance gate probe |

---

## 14. Host-side configuration (`config.env`, consumed by the `Makefile`)

| key | meaning |
|---|---|
| `AIR_PROFILE` | Databricks CLI profile (`df1`). The **DEFAULT profile is not it** — always pass `-p df1` |
| `DOCKERHUB_USER` / `IMAGE_NAME` / `IMAGE_TAG` | the image coordinates; `make retarget` writes them into every custom-image job file (`make lint` fails while any job disagrees); `make bump` increments the tag and retargets |
| `SECRET_SCOPE` / `SECRET_KEY` | Databricks secret holding registry credentials, so `make register` is non-interactive (the interactive fallback reads a TTY and hangs in CI) |
| `UC_CATALOG` / `UC_SCHEMA` / `UC_VOLUME` | Unity Catalog location. Job files carry the resolved path **literally** so any one of them is hand-submittable — if you change it here, grep `**/air/*.yaml` |
| `MAX_IMAGE_GB` | local size gate (`19.5`, **decimal** GB = 10^9 bytes, the stricter reading of AI Runtime's 20 GB limit); `make size` also fails if the image is missing or unmeasurable |
