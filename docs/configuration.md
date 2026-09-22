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
`dispatch_agentic.sh` resolves it itself (`_resolve_path`), substituting
`CODE_SOURCE_PATH` if set and the repo root otherwise — which is also why those paths
work when you run a launcher locally with `DRY_RUN=1`. The four plugin paths
(`FUNCTION_TOOL_PATH`, `CUSTOM_REWARD_PATH`, `TOOL_CONFIG_PATH`,
`AGENT_LOOP_CONFIG_PATH`) get this treatment. `EVAL_SCRIPT` is consumed by
`engine/serve/serve_and_eval.sh`, which is invoked from `command:` — there the shell
has already expanded it.

### Overriding anything at submit time, without editing a file

```bash
air run --file usecases/math/air/5_eval.yaml -p df1 \
  --override env_variables.EVAL_LIMIT=0 \
             env_variables.MODEL_PATH=/Volumes/.../global_step_24/actor/model/huggingface \
             parameters.actor_lr=3e-6 \
             compute.num_accelerators=16 \
             timeout_minutes=240
```

Dotted paths address any field in the YAML. This is the right way to sweep a knob:
the file stays the documented default, the override records the experiment.

---

## 2. Job-file anatomy

Every one of the 26 job files has the same shape:

| field | meaning | notes |
|---|---|---|
| `experiment_name` | job name + MLflow experiment | keep it stable; it is how runs group |
| `mlflow_experiment_directory` | Workspace folder for experiments | must start `/Workspace`; without it experiments scatter to per-user defaults |
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
| `output_dir` | both | `…/ckpt/default` | `trainer.default_local_dir`; checkpoints land at `<output_dir>/global_step_N/actor/model/huggingface/` |
| `total_epochs` | both | `1` | passes over the data |
| `train_batch_size` | **sync only** | `32` | prompts per GRPO step. Must satisfy `train_batch_size × rollout_n % trainer_GPUs == 0` — the launcher checks and fails with the arithmetic |
| `ppo_mini_batch_size` | both | `32` (sync) / `16` (async) | optimizer sub-batch; must divide trainer DP |
| `rollout_n` | both | `5` (sync) / `4` (async) | **GRPO group size** — samples per prompt. This is where advantage variance comes from |
| `total_training_steps` | **sync only** | `3` | step cap; **`0` disables it** (the rungs ship `3` as a smoke) |
| `total_rollout_steps` | **async only** | `64` | total rollout **samples** for the run — the async horizon |
| `max_prompt_length` | both | `1024` | per-turn prompt budget |
| `max_response_length` | both | `2048` | per-turn response budget (multi-turn multiplies it — §6) |
| `actor_lr` | both | `1e-6` | policy learning rate |
| `image_key` | **sync only** | `images` | set `''` for text-only data, or the multimodal path silently re-enables |
| `project_name` / `experiment_name` | both | `verl-on-air` / `grpo-*` | MLflow; `PROJECT_NAME`/`EXPERIMENT_NAME` env override them |

---

## 4. Mode and node roles — `engine/train/dispatch_agentic.sh`

| var | default | what it does |
|---|---|---|
| **`TRAIN_MODE`** | `async` | **the mode switch**: `async` → `run_grpo_fully_async.sh`, `sync` → `run_grpo_megatron.sh`. See [training-modes.md](training-modes.md) |
| **`TRAINING_NODES`** | `2` | ranks `[0, TRAINING_NODES)` train; the rest serve the judge. Set it **equal to the node count for a judge-free run** |
| `RENDEZVOUS_ROOT` | `/Volumes/main/mshtelma/verl/rendezvous` | UC dir for the judge-URL / training-done rendezvous files |
| `JUDGE_WAIT_TIMEOUT` | `2400` | how long training waits for the judge endpoint before failing |

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
| **`NORM_ADV_BY_STD_IN_GRPO`** | verl default `True` | set to **`False` for any graded reward**. With std-normalisation on, 0.05 and 1.0 get the *same* within-group advantage — a graded reward collapses to binary. **async launcher only** |

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
| `JUDGE_ENGINE` | `sglang` | `vllm` or `sglang`. The math use case sets `vllm` to ride the training image |
| `JUDGE_MODEL_PATH` / `JUDGE_MODEL_ID` | one is **required** | a staged Volume dir, or an HF repo id |
| `JUDGE_TP` | `8` | tensor parallel across the judge nodes (`8 × JUDGE_NODES`) |
| `JUDGE_SERVED_NAME` | `judge` | the model name the client asks for |
| `JUDGE_PORT` | `8000` | serving port |
| `JUDGE_GPU_MEM_UTIL` | `0.90` | memory fraction |
| `JUDGE_MAX_MODEL_LEN` | `16384` | judge context: prompt + the trajectory it grades |
| `JUDGE_HEALTH_TIMEOUT` | `2400` | wait for `/health`; a 744 GB first load is slow |
| `JUDGE_LOCAL_CACHE` | unset | NVMe dir (e.g. `/local_disk0/judge_cache`) to bulk-copy the model off UC FUSE first — much faster than random-reading FUSE |
| `JUDGE_STAGE_PARALLEL` | `8` | parallel copies during that staging |
| `JUDGE_RAY_VERSION` | unset | pin Ray for multi-node serving (`2.48.0`; see the `probe_vllm_multinode` diagnostic) |
| `JUDGE_RAY_PORT` | `6380` | deliberately **not** 6379 — training's Ray owns that |
| `JUDGE_MAX_LIFETIME` | — | self-exit guard, seconds |
| `JUDGE_EXTRA_ARGS` | — | engine passthrough, e.g. `--reasoning-parser glm45 --tool-call-parser glm47` |
| `JUDGE_RENDEZVOUS` / `JUDGE_EXIT_SENTINEL` | set by the dispatcher | where to publish the endpoint / when to shut down |
| `STAGE_ONLY` | `0` | stage the weights and exit without serving |

Client side (runs inside the reward actors):

| var | default | meaning |
|---|---|---|
| `JUDGE_BASE_URL` / `JUDGE_ENDPOINT_FILE` | set by the dispatcher | the endpoint, or the rendezvous file to read it from. The reward resolves the URL **at call time** because Ray does not reliably carry a driver `export` into actor processes |
| `JUDGE_MODEL` | `judge` | served model name |
| `JUDGE_MAX_TOKENS` | `2048` | judge response budget |
| `JUDGE_TIMEOUT` | `60` | per-call timeout |
| `JUDGE_TEMPERATURE` | `0` | deterministic grading |
| `JUDGE_DISABLE_THINKING` | `1` | some reasoning models think unconditionally at high effort and blow the parse rate |
| `JUDGE_TRAJECTORY_CHARS` | `8000` | how much trajectory the judge sees — truncate too hard and the judge grades blind |
| `JUDGE_BLEND_ALPHA` | `0.5` | judge/rule blend weight when blending is used |
| `JUDGE_API_KEY` | `EMPTY` | self-hosted endpoints need no real key |
| `JUDGE_DEBUG` | `0` | log prompts/responses |

---

## 11. Two spelling gotchas

1. **Booleans are not normalised across launchers.** `ROLLOUT_ENFORCE_EAGER` is
   compared against `"True"` in `run_grpo_fully_async.sh` but against `"1"` in
   `run_grpo_megatron.sh` (which is why `rung4` sets `'1'`). Everything that is passed
   straight through to Hydra (`MULTI_TURN`, `PARTIAL_ROLLOUT`,
   `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE`, `NORM_ADV_BY_STD_IN_GRPO`,
   `ROLLOUT_PREFIX_CACHING`) uses Python-style `True`/`False`. **Copy the spelling from
   a working job file rather than guessing** — a mistyped boolean reads as "off".
2. **`NORM_ADV_BY_STD_IN_GRPO` only recognises the exact string `False`.** Any other
   value leaves verl's default (`True`) in place.

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
| `QA_VS_TABLE` / `QA_VS_CATALOG` / `QA_VS_SCHEMA` | `wiki_qa_corpus` / `main` / `mshtelma` | `create_vs_index.py` — source Delta table |
| `QA_VS_EMBED_MODEL` | `databricks-gte-large-en` | `create_vs_index.py` — managed embedding model |
| `QA_VS_TEXT_COL` / `QA_VS_TITLE_COL` / `QA_VS_ID_COL` | `text` / `title` / `id` | `tool.py` — index column names |
| `QA_SEARCH_TOP_K` | `5` | `tool.py` — hits per search call |
| `QA_SNIPPET_CHARS` / `QA_TOOL_MAX_CHARS` | `600` / `4000` | `tool.py` — snippet and total tool-response caps |
| `QA_REWARD_METRIC` | `em` | `reward.py` + `eval.py` — **shared by both, so they cannot drift** |
| `QA_RETRIEVAL_BONUS` | `0.0` | `reward.py` — bonus when a retrieved passage held the gold. Measured inert here (RESULTS.md) |
| `QA_FORMAT_SCORE` | `0.0` | `reward.py` — credit for well-formed output alone |
| `QA_DATASETS` / `QA_CORPUS_DATASETS` / `QA_CORPUS_SPLITS` | `hotpotqa` / … | `prep_data.py`, `build_corpus.py` — **the job files override these to MuSiQue** |
| `QA_PREP_TRAIN_LIMIT` / `QA_PREP_VAL_LIMIT` | `0` (all) / `500` | `prep_data.py` |
| `QA_VAL_PARQUET` | `…/qa_search/test.parquet` | `eval.py` — **override it to your data dir** |
| `QA_HF_CACHE` | `/local_disk0/hf_cache` | prep jobs — NVMe, not FUSE |

### math

| var | default | read by |
|---|---|---|
| `MATH_LEVELS` | all | `prep_data.py` — `3,4,5` keeps the learnable band |
| `MATH_TOOL_OUT_DIR` | `~/data/math_tool` | `prep_data.py` |
| `N_TRAIN` / `N_TEST` | `0` (all) | `prep_data.py` |
| `MATH500_ID` | `HuggingFaceH4/MATH-500` | `eval.py` |

### Eval harness (`engine/serve/serve_and_eval.sh` + any `eval.py`)

| var | default | meaning |
|---|---|---|
| **`EVAL_SCRIPT`** | **required** | absolute path to the use case's `eval.py`; its directory goes on `PYTHONPATH` so eval imports the *same* `reward.py` training used |
| `EVAL_MODEL_PATH` | base model | **what to serve** — swap this for a checkpoint |
| `MODEL_PATH` | base model | what the eval client loads the **tokenizer** from (keep it consistent) |
| `EVAL_TP` | `8` | serving tensor parallel |
| `EVAL_SERVE_LEN` | `8192` | served context. Must cover the whole multi-turn episode |
| `EVAL_GPU_UTIL` | `0.85` | vLLM memory fraction |
| `EVAL_STAGE` | `1` | bulk-copy the model UC→NVMe before serving (FUSE random-read is slow) |
| `EVAL_LOCAL_CACHE` | `/local_disk0/eval_model` | that NVMe destination |
| `EVAL_HEALTH_TIMEOUT` | `1800` | wait for `/health` |
| `EVAL_PORT` / `EVAL_MODEL` | `8000` / `eval` | endpoint and served name |
| `EVAL_SERVE_EXTRA_ARGS` | — | extra `vllm serve` flags |
| **`EVAL_MAX_TURNS`** | `8` | eval-time tool budget. **Must match training's `MAX_TURNS`** for a fair comparison — and must be identical between the baseline and the trained eval |
| `EVAL_LIMIT` | `0` (all) | number of questions; small values are a cheap harness smoke |
| `EVAL_MAX_TOKENS` | `512` (search) / `1024` (math) | per-turn response cap |
| `EVAL_TEMPERATURE` | `0` | greedy, so the comparison is deterministic |
| `EVAL_CONCURRENCY` | `32` | parallel in-flight questions |
| `EVAL_MAX_CONT` | `2`/`3` | continuation attempts on a truncated answer |
| `EVAL_REQ_TIMEOUT` / `EVAL_HTTP_RETRIES` | `600`–`900` / `4` | client robustness |
| `EVAL_OUT` | — | JSON summary path |
| `EVAL_TRACE_OUT` | — | per-question JSONL traces — **required input for `analyze_traces.py`** |

### Model staging (`engine/stage_model.py`)

| var | default | meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | HF repo to stage |
| `MODEL_DIR` | `…/verl/models/<basename>` | Volume destination |
| `SCRATCH_DIR` | `/local_disk0/hf_stage` | NVMe staging dir before the FUSE copy |
| `HF_TOKEN` | — | gated repos / rate limits |
| `HF_HUB_DISABLE_XET` | — | set `1` for UC Volumes: FUSE rejects Xet/CAS parallel range writes |

### geo3k infra jobs

| var | default | meaning |
|---|---|---|
| `GEO3K_OUT_DIR` | `…/data/geo3k` | prep destination |
| `N_TRAIN` / `N_TEST` | `64` / `8` | subset size (**`0` = full split**) |
| `EVAL_FILE`, `N_PROMPTS`, `N_SAMPLES`, `TEMPERATURE`, `GEN_TP`, `MAX_TOKENS`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | see `infra/geo3k/baseline_eval.py` | the reward-variance gate probe |

---

## 14. Host-side configuration (`config.env`, consumed by the `Makefile`)

| key | meaning |
|---|---|
| `AIR_PROFILE` | Databricks CLI profile (`df1`). The **DEFAULT profile is not it** — always pass `-p df1` |
| `DOCKERHUB_USER` / `IMAGE_NAME` / `IMAGE_TAG` | the image coordinates; `make bump` rewrites the tag here **and in every job file** |
| `SECRET_SCOPE` / `SECRET_KEY` | Databricks secret holding registry credentials, so `make register` is non-interactive (the interactive fallback reads a TTY and hangs in CI) |
| `UC_CATALOG` / `UC_SCHEMA` / `UC_VOLUME` | Unity Catalog location. Job files carry the resolved path **literally** so any one of them is hand-submittable — if you change it here, grep `**/air/*.yaml` |
| `MAX_IMAGE_GB` | local size gate (`19.5`); AI Runtime rejects images >20 GB |
