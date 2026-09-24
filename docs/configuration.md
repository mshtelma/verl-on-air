# Configuration reference

Every setting the jobs use: where it lives, its default, and what it does. For the few settings
that change results, start with [tuning.md](tuning.md). For the verl and Megatron flags the
launchers set, see [verl-config-reference.md](verl-config-reference.md).

## How a setting reaches the job

- `env_variables:` become the process environment; every capitalised name on this page is one.
  Quote numbers (`'8'`). Booleans accept `True`/`False`, `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`.
- `parameters:` arrive as a YAML file that `hp <key> <default>` (`engine/lib/hparams.sh`) reads:
  model, data, output directory, batch shape. An empty value is not a missing one, so
  `image_key: ''` means text-only data.
- `compute.num_accelerators` is the total GPU count (16 on `GPU_8xH100` is 2 nodes). AI Runtime
  injects `NUM_NODES`, `LOCAL_WORLD_SIZE`, `POD_RANK`, `MASTER_ADDR` and `MASTER_PORT`, and runs
  `command:` once per node.
- `code_source:` uploads a snapshot of your checkout to `${CODE_SOURCE_PATH}`.

air expands `${CODE_SOURCE_PATH}` in `command:` but not in `env_variables:`, so the engine
resolves path values itself (`resolve_code_path` in [`engine/lib/paths.sh`](../engine/lib/paths.sh),
no `eval`; relative paths are anchored at the snapshot, or the repo root for a local
`DRY_RUN=1`). The file must exist or the job stops early:

| variable | resolved by | checked before |
|---|---|---|
| `FUNCTION_TOOL_PATH`, `CUSTOM_REWARD_PATH`, `TOOL_CONFIG_PATH`, `AGENT_LOOP_CONFIG_PATH` | `engine/train/dispatch_agentic.sh` | the role split and Ray |
| `EVAL_SCRIPT` | `engine/serve/serve_and_eval.sh` | model staging and vLLM |

A new path-valued variable has to be added to one of these lists. To sweep a knob, override it
at submit time with a dotted path; the file keeps the default:

```bash
air run --file usecases/math/air/5_eval.yaml -p <profile> \
  --override env_variables.EVAL_LIMIT=0 parameters.actor_lr=3e-6 timeout_minutes=240
```

## Job file fields

| field | meaning |
|---|---|
| `experiment_name` | job name and MLflow experiment; keep it stable so runs group together |
| `mlflow_experiment_directory` | optional Workspace folder for the experiment (must start with `/Workspace`); unset means your per-user default |
| `compute.num_accelerators` | total GPUs (`16` = 2 nodes of `GPU_8xH100`) |
| `compute.accelerator_type` | `GPU_1xA10`, `GPU_1xH100` or `GPU_8xH100` |
| `environment.docker_image.url` | the custom image. It must be registered first (`make register`), and registration is per tag: new content pushed under an old tag is not what jobs run, so after a Dockerfile change run `make bump && make release` |
| `environment.version` + `dependencies` | a stock runtime instead of the custom image, for light jobs such as `usecases/math/air/1_prep_data.yaml` |
| `code_source.snapshot.root_path` | snapshot root, relative to the YAML file (`../../..` from `usecases/<uc>/air/`) |
| `code_source.snapshot.include_paths` | uploaded on every submit: list `engine` and the one use case, not the repo root |
| `max_retries` | `0` for training, because a retry re-bills the whole multi-node job; above 0 only for resumable work such as staging |
| `timeout_minutes` | hard limit, including time spent waiting for GPUs. A 30-minute smoke test that queues for 27 minutes times out having barely run |
| `command` | runs on every node |

Code ships with each submit, so a launcher, reward or tool change needs no image rebuild.

## `parameters:`

| key | read by | default | meaning |
|---|---|---|---|
| `model_name` | both | `Qwen/Qwen3.5-35B-A3B` (sync), `Qwen/Qwen3.5-9B` (async) | HF repo id or a Volume path; use a staged Volume path for large models |
| `train_files` / `val_files` | both | geo3k parquet | verl parquet files on the Volume |
| `output_dir` | both | set by every job | experiment root. Each run writes to `<output_dir>/<RUN_ID>/`: checkpoints under `global_step_N/actor/model/huggingface/`, plus `run_manifest.json` and `run_result.json` |
| `total_epochs` | both | `1` | passes over the data |
| `train_batch_size` | sync | `32` | prompts per step. `train_batch_size × rollout_n` must be divisible by the trainer GPU count (checked at start) |
| `ppo_mini_batch_size` | both | `32` (sync), `16` (async) | optimizer mini-batch; must split evenly over trainer DP |
| `rollout_n` | both | `5` (sync), `4` (async) | GRPO group size: samples per prompt |
| `total_training_steps` | sync | `3` | step cap; `0` removes it. The ladder rungs ship `3` as a smoke test |
| `total_rollout_steps` | async | `64` | rollout samples (prompt groups) for the whole run; this sets the length of an async run |
| `max_prompt_length` | both | `1024` | budget for the initial prompt |
| `max_response_length` | both | `2048` | single-turn: the response cap. Multi-turn: only a unit in the episode budget (see [Multi-turn tool calling](#multi-turn-tool-calling)); no single turn is capped by it. Evals cap each request with `EVAL_MAX_TOKENS` |
| `actor_lr` | both | `1e-6` | policy learning rate |
| `image_key` | sync | `images` | set `''` for text-only data, otherwise the multimodal path stays on |
| `project_name` / `experiment_name` | both | `verl-on-air` / `grpo-*` | MLflow names; `PROJECT_NAME` and `EXPERIMENT_NAME` override them |

## Run identity

`make` sets these. Training needs `RUN_ID`.

| var | default | meaning |
|---|---|---|
| `RUN_ID` | required (`make` sets `<UTC time>-<commit>`) | the run writes to `<output_dir>/<RUN_ID>/`; it also names the rendezvous directory and eval artifacts. By hand: `--override env_variables.RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` |
| `RESUME` | `never` | `never`: a fresh run, refused if `<output_dir>/<RUN_ID>/` already has checkpoints. `auto`: continue that run from its latest checkpoint. `<path>/global_step_N`: start from that checkpoint |
| `MAX_CKPT_TO_KEEP` | keep all | keep only the N newest checkpoints. Off by default so you can evaluate several |
| `GIT_SHA` / `VOA_IMAGE` | set by `make` | the commit (`-dirty` if tracked files changed) and image the run came from, recorded in `run_manifest.json` and in eval artifacts |

## Mode and node roles

Read by `engine/train/dispatch_agentic.sh`.

| var | default | meaning |
|---|---|---|
| `TRAIN_MODE` | `async` | `async` runs `run_grpo_fully_async.sh`, `sync` runs `run_grpo_megatron.sh` ([training-modes.md](training-modes.md)) |
| `TRAINING_NODES` | `2` | ranks below this number train and the rest serve the judge. Set it to the node count for a job without a judge |
| `JUDGE_NODES` | none | required when nodes are left over for a judge; must equal the node count minus `TRAINING_NODES` |
| `RENDEZVOUS_ROOT` | `<volume>/rendezvous` | Volume directory for coordination files, one subdirectory per `RUN_ID`: the judge URL, `training_done`, `ABORT.json`, and the training head's heartbeat and exit record. A wait only accepts files written during this job (`RDV_SKEW_S`, default 300 s of clock skew) |
| `RAY_NODES_TIMEOUT_S` / `RAY_HEARTBEAT_STALE_S` | `900` / `600` | how long the head waits for every node's GPUs, and how long a worker waits on a silent head before exiting 1 |
| `JUDGE_STAGE_TIMEOUT` / `JUDGE_HEALTH_TIMEOUT` | `3600` / `2400` | the judge's copy of its weights to local NVMe, then its load until `/health` answers; a 744 GB model is slow the first time |
| `JUDGE_WAIT_TIMEOUT` | stage + health + `600` | how long training waits for the judge endpoint; a value below stage + health is refused |

The dispatcher also puts the directories of `CUSTOM_REWARD_PATH` and `FUNCTION_TOOL_PATH` on
`PYTHONPATH`, so `reward.py`, `tool.py` and `eval.py` can import each other by bare name.

## Topology and parallelism

These follow from the model and the GPU count, not from the task. The arithmetic is in
[sizing.md](sizing.md).

| var | default | meaning |
|---|---|---|
| `TP` | `1` fsdp, `2` classic (sync); `2` (async) | tensor parallel; must divide the 16 attention heads |
| `PP` | `1` | pipeline parallel |
| `CP` | `1` | context parallel |
| `EP` | `8` (sync), `1` (async; the use cases set `8`) | expert parallel, the main memory lever for this MoE (92.5% of the weights are routed experts) |
| `ETP` | `1` | tensor parallel inside an expert |
| `GEN_TP` | `8` (sync), `1` (async; the use cases set `8`) | vLLM rollout tensor parallel; keep it at or below the GPUs per node so it stays on NVLink |
| `NNODES` / `NGPUS_PER_NODE` / `NODE_RANK` | from `NUM_NODES` / `LOCAL_WORLD_SIZE` / `POD_RANK` | injected; set them only for local dry runs |
| `MEGATRON_MODE` | `fsdp` | sync only: `fsdp` (ZeRO-3, shards params, grads and optimizer) or `classic` (ZeRO-1, replicates params and grads) |
| `OFFLOAD` | `auto` | sync only. `auto` turns offload off for fsdp at 16 or more trainer GPUs (below that it stops, because fsdp cannot offload) and for classic at 32 or more; otherwise it turns it on. CPU offload crashes Megatron-FSDP |
| `OFFLOAD_FRACTION` | `1` | share of the optimizer state offloaded. Offloading needs a lot of host RAM: about 550 GiB per node for the 35B on one node ([sizing.md](sizing.md)) |
| `MAX_MODEL_LEN` | `8192`, or the episode length for multi-turn | vLLM context cap. Without it vLLM sizes the KV cache for the model's 262144-token maximum and fails |
| `ROLLOUT_GPU_MEM_UTIL` | `0.8` (async), `0.6` (sync) | vLLM `gpu_memory_utilization`. It sizes the KV cache only and does not fix a weight-sync OOM ([sizing.md](sizing.md)) |
| `ROLLOUT_ENFORCE_EAGER` | `False` | skip CUDA-graph capture: saves a few GiB, slows generation |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `False` | async only: NCCL instead of vLLM's custom all-reduce, keeping CUDA graphs. Both use cases set `True`, because with `GEN_TP` of 8 or less inside one node the custom kernel crashes graph capture on H100 |
| `ROLLOUT_PREFIX_CACHING` | `False` | async only: vLLM prefix cache. A large speed-up for multi-turn, where the system prompt and earlier turns repeat; safe because verl flushes the cache on every weight sync |
| `ROLLOUT_TEMP` | `1.0` | async only: sampling temperature. Hotter sampling gives more varied groups |
| `WEIGHT_BUCKET_MB` | unset | sync only: weight-sync bucket size. It has to exceed the 970 MiB embedding tensor. Unset by default because its config path moved between verl releases ([troubleshooting.md](troubleshooting.md)) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | set by the launcher | `1` for classic, unset for Megatron-FSDP (otherwise its collectives queue behind compute). Do not set it in a job file |

## Fully-async (`run_grpo_fully_async.sh`)

| var | default | meaning |
|---|---|---|
| `ROLLOUT_NNODES` | `0` | 1 or more: the Rollouter gets that many whole nodes and the Trainer the rest. `0`: split a single node |
| `N_GPUS_ROLLOUT` | `4` | rollout GPUs when `ROLLOUT_NNODES=0` |
| `STALENESS` | `0.1` | `async_training.staleness_threshold`. `0` makes the Trainer wait for fresh samples; higher values let the Rollouter run ahead |
| `TRIGGER_SYNC_STEP` | `2` | optimizer updates between weight syncs |
| `REQUIRE_BATCHES` | `1` | mini-batches per update |
| `PARTIAL_ROLLOUT` | `True` | keep partly generated sequences across a weight sync instead of discarding them |
| `LR_DECAY_STEPS` | `total_rollout_steps` | must be set: streaming has no dataloader to derive it from, and Megatron's scheduler asserts without it |

A run's length follows from these (the launcher prints it at start). `SAVE_FREQ` counts weight syncs:

```
samples per weight sync = TRIGGER_SYNC_STEP × REQUIRE_BATCHES × ppo_mini_batch_size
weight syncs            = total_rollout_steps / samples per weight sync
```

## Multi-turn tool calling

Read by both launchers. With `MULTI_TURN=False` the rest of this table does not apply.

| var | default | meaning |
|---|---|---|
| `MULTI_TURN` | `False` | `True` turns on verl's tool agent loop: the model calls tools, verl runs them and feeds the results back |
| `MAX_TURNS` | `4` | assistant and user turns per episode, i.e. the tool budget. For a retrieval agent this is the hop budget, one of the knobs that matters most |
| `FUNCTION_TOOL_PATH` | none | Python file of stateless `@function_tool` functions, offered to every sample |
| `TOOL_CONFIG_PATH` | none | YAML of stateful `BaseTool` classes, as an alternative |
| `AGENT_LOOP_CONFIG_PATH` | none | YAML that registers a custom agent loop; the data's `agent_name` column selects it |
| `TOOL_FORMAT` | `hermes` | tool-call parser. Qwen3.5 writes XML tool calls, so use `qwen3_coder`. With the wrong parser no tool call ever decodes; check it with `infra/diagnostics/air/probe_tool_format.yaml` |
| `AGENT_NUM_WORKERS` | `8` | parallel agent-loop workers |
| `MAX_TOOL_RESPONSE_LEN` | `512` | token cap per tool response; agentic-search sets `4000` because its tools return passages |

Both launchers size an episode the same way:

```
episode_len   = (max_prompt_length + max_response_length) × MAX_TURNS
response      = episode_len − max_prompt_length     # the budget verl trains on
MAX_MODEL_LEN = episode_len                         # vLLM holds the whole episode
```

The actor trains on the whole trajectory, tool responses included, so `MAX_TURNS` also sets the
memory of the backward pass. It is the first knob to lower after an out-of-memory error there.

## Reward

| var | default | meaning |
|---|---|---|
| `CUSTOM_REWARD_PATH` | none | your `reward.py`; verl imports it and its directory goes on `PYTHONPATH` |
| `CUSTOM_REWARD_NAME` | `compute_score` | the function to call |
| `REWARD_MANAGER` | verl's default | `naive` (a rule, in-process) or `rate_limited` (asynchronous and concurrent, for an LLM judge); all accepted values are in the table below |
| `REWARD_MAX_CONCURRENT` | `1` inside verl | concurrent reward calls per reward worker. verl's default of 1 is serial, so set it for a judge (math uses `64`) |
| `REWARD_MAX_RPM` / `REWARD_MAX_TPM` | none | request and token rate limits for an external judge API |
| `REWARD_TIMEOUT` | none | per-call timeout in seconds |
| `REWARD_SOURCE` | `judge` (math) | math only: optimise the judge score, the rule, or a blend. Read by `usecases/math/reward.py` |
| `NORM_ADV_BY_STD_IN_GRPO` | `True` (verl's default) | whether GRPO divides a group's advantages by the group's reward std; any value other than a boolean stops the launcher. Dividing keeps a graded reward's order and relative gaps (`[0, 0.05, 0.7, 1]` becomes `[−0.89, −0.79, 0.53, 1.14]`) but gives a low-spread group as much weight as a high-spread one. `False` keeps advantages in reward units. Which works better is untested ([tuning.md](tuning.md)); the math job sets `True` explicitly |

verl calls the function with the keyword arguments `data_source`, `solution_str`, `ground_truth`
and `extra_info`, and expects a dict with a `score` key. Every other key becomes its own MLflow
metric, which lets you watch correctness separately from formatting.

## Checkpointing, logging, validation

| var | default | meaning |
|---|---|---|
| `SAVE_FREQ` | `-1` (never) | checkpoint interval: weight syncs in async mode, optimizer steps in sync mode. With `SAVE_FREQ > 0` the final version is always saved, and that is the checkpoint the run is certified against |
| `TEST_FREQ` | `-1` (never) | in-loop validation interval. Both use cases leave it off and evaluate saved checkpoints with the eval job, which keeps base and trained evals identical. The async launcher refuses `0` |
| `VAL_BEFORE_TRAIN` | `False` | sync only: validate before the first step |
| `SEED` | unset | seeds the data order, Megatron and vLLM sampling (`data.seed`, `actor.megatron.seed`, `rollout.seed`). Unset leaves verl's data order unseeded; the search jobs set `42` |
| `PROJECT_NAME` / `EXPERIMENT_NAME` | from `parameters` | MLflow names |
| `USE_DIST_CKPT` / `DIST_CKPT_PATH` | `False` / none | save a sharded Megatron checkpoint instead of the gathered HF export. This also makes the run load its initial weights from `DIST_CKPT_PATH`. Not needed at 35B |
| `ALLOW_UNCERTIFIED` | `False` | async only: for throwaway smoke runs with `SAVE_FREQ=-1`; the result carries verl's exit code and is marked uncertified |
| `DRY_RUN` | `0` | `1` prints the resolved verl command and exits before Ray starts. Works on a laptop |

## Judge server and client

Only used by the judge-reward pattern. Server side, on the judge nodes (`engine/serve/serve_judge.sh`):

| var | default | meaning |
|---|---|---|
| `JUDGE_ENGINE` | `sglang` | `vllm` or `sglang`. The math use case uses `vllm`, which the training image already has. A multi-node judge must use `vllm`; multi-node SGLang is refused |
| `JUDGE_MODEL_PATH` / `JUDGE_MODEL_ID` | one is required | a staged Volume directory, or an HF repo id |
| `JUDGE_TP` | `8` | tensor parallel across the judge nodes (8 × `JUDGE_NODES`) |
| `JUDGE_SERVED_NAME` | `judge` | the model name clients ask for |
| `JUDGE_PORT` | `8000` | serving port |
| `JUDGE_GPU_MEM_UTIL` | `0.90` | GPU memory fraction |
| `JUDGE_MAX_MODEL_LEN` | `16384` | judge context: its prompt plus the trajectory it grades |
| `JUDGE_LOCAL_CACHE` | unset | local NVMe directory (e.g. `/local_disk0/judge_cache`) to copy the model to first; much faster than reading it from the Volume. The copy must finish within `JUDGE_STAGE_TIMEOUT` |
| `JUDGE_STAGE_PARALLEL` | `8` | parallel copies while staging |
| `JUDGE_RAY_VERSION` / `JUDGE_RAY_PATH` | `2.48.0` / `/opt/judge-ray` | the Ray a multi-node judge runs on. vLLM's Ray executor does not work with the Ray 2.58 that verl uses, so the image carries a separate Ray 2.48, placed first on `PYTHONPATH` on the judge nodes only. Nothing is installed at start-up; a mismatch stops the judge |
| `JUDGE_RAY_PORT` | `6380` | not 6379, which training's Ray uses |
| `JUDGE_MAX_LIFETIME` | none | exit after this many seconds. Exit codes: `0` training finished, `1` never became healthy, `3` the server died, `4` lifetime reached |
| `JUDGE_EXTRA_ARGS` | none | passed to the engine, e.g. `--reasoning-parser glm45 --tool-call-parser glm47` |
| `JUDGE_RENDEZVOUS` / `JUDGE_EXIT_SENTINEL` | set by the dispatcher | where to publish the endpoint, and the file that tells the judge to stop |
| `STAGE_ONLY` | `0` | stage the weights and exit |

Client side, in the reward workers (`usecases/math/reward.py`):

| var | default | meaning |
|---|---|---|
| `JUDGE_BASE_URL` / `JUDGE_ENDPOINT_FILE` | set by the dispatcher | the endpoint, or the rendezvous file to read it from. The URL is read at call time because Ray actors do not reliably inherit the driver's environment |
| `JUDGE_MODEL` | `judge` | served model name |
| `JUDGE_MAX_TOKENS` | `2048` | verdict budget; a verdict cut off by it is invalid, never graded |
| `JUDGE_TIMEOUT` | `60` | per-attempt HTTP timeout |
| `JUDGE_RETRIES` / `JUDGE_BACKOFF_S` | `2` / `2` | retries, for transient failures only (connection, timeout, HTTP 429/5xx) |
| `JUDGE_DEADLINE_S` | `REWARD_TIMEOUT − 10` | total time for one verdict, retries included. Keep it below `REWARD_TIMEOUT`: when that expires, verl replaces the sample's result with a different set of keys and the batch fails |
| `JUDGE_TEMPERATURE` | `0` | deterministic grading |
| `JUDGE_DISABLE_THINKING` | `1` | some reasoning models otherwise think at length and fail to return a parseable verdict |
| `JUDGE_STRUCTURED_OUTPUT` | `1` | ask vLLM for schema-constrained JSON so LaTeX in `reason` cannot break parsing |
| `JUDGE_TRAJECTORY_CHARS` | `36000` | trajectory budget in characters (about 10k tokens). Longer work keeps its head and tail with a marked cut and logs `judge_input_truncated=1`. When the judge's tokenizer loads (`JUDGE_TOKENIZER_PATH`, else `JUDGE_MODEL_PATH`), the input is also cut to fit `JUDGE_MAX_MODEL_LEN − JUDGE_MAX_TOKENS` in the judge's own tokens |
| `JUDGE_FALLBACK` | `rule` | score for a sample with no valid verdict: `rule` (exact match) or `zero`, flagged `judge_fallback=1` |
| `JUDGE_MAX_FAIL_RATE` / `JUDGE_FAIL_WINDOW` / `JUDGE_FAIL_MIN_CALLS` | `0.05` / `200` / `50` | per reward worker: if more than 5% of its last 200 calls got no valid verdict, the run is aborted (`engine/lib/run_control.py`). `1` turns the check off |
| `JUDGE_BLEND_ALPHA` | `0.5` | judge weight when `REWARD_SOURCE=blend`, between 0 and 1 |
| `JUDGE_API_KEY` | `EMPTY` | a self-hosted endpoint needs no key |
| `JUDGE_DEBUG` | `0` | log each verdict |
| `PRE_TRAIN_CHECK` | none | a script the dispatcher runs on training rank 0 once the judge is up (math: `judge_selfcheck.py`). A non-zero exit stops the job |

Read judge metrics per valid verdict: divide `mean(judge_score)` and `mean(judge_agree)` by
`mean(judge_valid)`, the coverage. Fallback samples count in neither.

## Checks before a run starts

Both launchers run [`engine/lib/preflight.py`](../engine/lib/preflight.py) before training
(`DRY_RUN=1` included), and `make preflight F=<job.yaml>` runs it locally, printing the job's plan
and an upper bound on the GPU-hours it can bill. It stops a job on a mistyped or out-of-range
knob, a knob only the other mode reads, an unknown name with an engine prefix (so
`ROLLOUT_TEMPERATURE` is caught), a layout that does not fit the model (TP, PP and EP must divide
the heads, layers and experts; DP and the mini-batch must split evenly), FSDP with CPU offload, and
a malformed `parameters:` block.

### Every typed knob

Generated from the preflight schema by `make docs-config`. `make lint` fails if the two disagree.

<!-- BEGIN GENERATED: knobs (scripts/docs_config.py; do not edit by hand) -->

| knob | type | read by | allowed |
|---|---|---|---|
| `ABORT_GRACE_S` | float | both | ≥ 0 |
| `ABORT_POLL_S` | float | both | > 0 |
| `AGENT_LOOP_CONFIG_PATH` | str | both | any string |
| `AGENT_NUM_WORKERS` | int | both | ≥ 1 |
| `ALLOW_UNCERTIFIED` | bool | async | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `ASYNC_WARMUP_BATCHES` | int | sync | ≥ 0 |
| `CERT_SETTLE_S` | float | both | ≥ 0 |
| `CKPT_ENGINE_BACKEND` | str | sync | any string |
| `CP` | int | both | ≥ 1 |
| `CUSTOM_REWARD_NAME` | str | both | any string |
| `CUSTOM_REWARD_PATH` | str | both | any string |
| `DATA_SHUFFLE` | bool | sync | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `DIST_CKPT_PATH` | str | both | any string |
| `EP` | int | both | ≥ 1 |
| `ETP` | int | both | ≥ 1 |
| `EXPERIMENT_NAME` | str | both | any string |
| `FAULT_INJECT` | enum | async | `kill-trainer-after-save` |
| `FUNCTION_TOOL_PATH` | str | both | any string |
| `GEN_TP` | int | both | ≥ 1 |
| `GIT_SHA` | str | both | any string |
| `LR_DECAY_STEPS` | int | async | ≥ 1 |
| `MAX_CKPT_TO_KEEP` | int | both | ≥ 1 |
| `MAX_MODEL_LEN` | int | both | ≥ 1 |
| `MAX_OFF_POLICY` | int | sync | ≥ 0 |
| `MAX_TOOL_RESPONSE_LEN` | int | both | ≥ 1 |
| `MAX_TURNS` | int | both | ≥ 1 |
| `MEGATRON_MODE` | enum | sync | `fsdp` \| `classic` |
| `MULTI_TURN` | bool | both | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `NORM_ADV_BY_STD_IN_GRPO` | bool | both | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `N_GPUS_ROLLOUT` | int | async | ≥ 1 |
| `OFFLOAD` | enum | sync | `auto` \| `0` \| `1` |
| `OFFLOAD_FRACTION` | float | both | ≥ 0, ≤ 1 |
| `PARAM_SYNC_STEP` | int | sync | ≥ 1 |
| `PARTIAL_ROLLOUT` | bool | async | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `PP` | int | both | ≥ 1 |
| `PROJECT_NAME` | str | both | any string |
| `REQUIRE_BATCHES` | int | async | ≥ 1 |
| `RESUME` | str | both | any string |
| `REWARD_MANAGER` | enum | both | `naive` \| `prime` \| `batch` \| `dapo` \| `gdpo` \| `rate_limited` \| `remote` |
| `REWARD_MAX_CONCURRENT` | int | both | ≥ 1 |
| `REWARD_MAX_RPM` | int | both | ≥ 1 |
| `REWARD_MAX_TPM` | int | both | ≥ 1 |
| `REWARD_SOURCE` | str | both | any string |
| `REWARD_TIMEOUT` | float | both | > 0 |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | bool | async | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `ROLLOUT_ENFORCE_EAGER` | bool | both | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `ROLLOUT_GPU_MEM_UTIL` | float | both | > 0, ≤ 1 |
| `ROLLOUT_NNODES` | int | both | ≥ 0 |
| `ROLLOUT_PREFIX_CACHING` | bool | async | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `ROLLOUT_TEMP` | float | async | > 0 |
| `RUN_ID` | str | both | any string |
| `SAVE_FREQ` | int | both | any |
| `SEED` | int | both | ≥ 0 |
| `STALENESS` | float | async | ≥ 0 |
| `TEST_FREQ` | int | both | any |
| `TOOL_CONFIG_PATH` | str | both | any string |
| `TOOL_FORMAT` | enum | both | `hermes` \| `gpt-oss` \| `qwen3_coder` \| `glm` \| `seed` \| `minimax` \| `kimi` \| `deepseek_v4` \| `gemma4` |
| `TP` | int | both | ≥ 1 |
| `TRAINER_MODE` | enum | sync | `sync` \| `separate_async` |
| `TRIGGER_SYNC_STEP` | int | async | ≥ 1 |
| `USE_DIST_CKPT` | bool | both | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `VAL_BEFORE_TRAIN` | bool | sync | `True` \| `False` (also `true`/`1`/`yes`/`on`, `false`/`0`/`no`/`off`) |
| `VOA_IMAGE` | str | both | any string |
| `WEIGHT_BUCKET_MB` | int | sync | ≥ 1 |

<!-- END GENERATED: knobs -->

## Use-case settings

The use-case code reads these, so a new use case defines its own.

### agentic-search

| var | default | read by |
|---|---|---|
| `QA_VS_ENDPOINT` | none | `tool.py`, `create_vs_index.py`: the Vector Search endpoint |
| `QA_VS_INDEX` | none | `tool.py`: full index name, `catalog.schema.index` |
| `QA_VS_CATALOG` / `QA_VS_SCHEMA` / `QA_VS_TABLE` | set in `2_build_index.yaml`; table `wiki_qa_big_corpus` | `create_vs_index.py`: where the corpus table goes. The table name is a base name: a build creates `<base>_v<h8>` and the index `<base>_v<h8>_index`, where `h8` is the corpus content hash, so a new corpus never touches a live table |
| `QA_VS_WAREHOUSE_ID` | required | `create_vs_index.py`: the SQL warehouse that loads the table (`make search-index WAREHOUSE_ID=<id>`) |
| `QA_VS_CREATE_ENDPOINT` | `0` | `create_vs_index.py`: `1` creates a missing endpoint (billable and persistent); otherwise a missing endpoint is an error |
| `QA_VS_SQL_TIMEOUT_S` / `QA_VS_WAIT_TIMEOUT_S` | `1800` / `3000` | `create_vs_index.py`: deadline per SQL statement, and how long `--wait-only` waits |
| `QA_VS_EMBED_MODEL` | `databricks-gte-large-en` | `create_vs_index.py`: the managed embedding model |
| `QA_VS_TEXT_COL` / `QA_VS_TITLE_COL` / `QA_VS_ID_COL` | `text` / `title` / `id` | `tool.py`: index column names |
| `QA_SEARCH_TOP_K` | `5` | `tool.py`: hits per search call |
| `QA_SNIPPET_CHARS` / `QA_TOOL_MAX_CHARS` | `600` / `4000` | `tool.py`: snippet length and total tool-response cap |
| `QA_REWARD_METRIC` | `em` | `reward.py` and `eval.py`, through the shared scorer `score_segments` |
| `QA_RETRIEVAL_BONUS` | `0.0` | `reward.py`: bonus when a retrieved passage contained the gold answer. It did not help in the one run that tried it ([RESULTS.md](../RESULTS.md)) |
| `QA_FORMAT_SCORE` | `0.0` | `reward.py`: credit for well-formed output alone |
| `QA_DATASETS` / `QA_CORPUS_DATASETS` / `QA_CORPUS_SPLITS` | `musique` / `musique,hotpotqa` / `train,validation` | `prep_data.py`, `build_corpus.py`: sources, read at the commits pinned in `prep_data.SOURCES`. A corpus source that fails to load fails the build; `build_corpus.py --allow-partial` writes a corpus that its manifest marks incomplete |
| `QA_PREP_TRAIN_LIMIT` / `QA_PREP_VAL_LIMIT` | `0` (all) / `500` | `prep_data.py` |
| `QA_VAL_PARQUET` | `<volume>/data/qa_musique/test.parquet` | `eval.py`: the question set |
| `QA_HF_CACHE` | `/local_disk0/hf_cache` | prep jobs: HF cache on local NVMe |

### math

| var | default | read by |
|---|---|---|
| `MATH_LEVELS` | all | `prep_data.py`: `3,4,5` keeps problems the base model neither always solves nor always misses |
| `MATH_TOOL_OUT_DIR` | `~/data/math_tool` | `prep_data.py` |
| `N_TRAIN` / `N_TEST` | `0` (all) | `prep_data.py` |
| `MATH500_ID` / `MATH500_REVISION` | `HuggingFaceH4/MATH-500` / a pinned commit | `eval.py`: any other `MATH500_ID` needs a 40-character `MATH500_REVISION` |

### Dataset sources

Every Hub dataset is read at a pinned commit (`engine/lib/data_manifest.py`), and each prep job
writes `DATA_MANIFEST.json` (sources, revisions, row counts per filter, output hashes) next to its
outputs. `ALLOW_FALLBACK_SOURCE=1` lets a listed mirror replace an unavailable source, but only if
its content matches the pinned data; by default an unavailable source is an error.

### Eval harness

`engine/serve/serve_and_eval.sh` plus a use case's `eval.py`.

| var | default | meaning |
|---|---|---|
| `EVAL_SCRIPT` | required | path to the use case's `eval.py`. Its directory goes on `PYTHONPATH`, so the eval imports the same `reward.py` training used |
| `EVAL_MODEL_PATH` | the base model (baseline job); none (checkpoint job) | what to serve: a model directory, or a checkpoint's `global_step_N`, whose HF export is found and verified (verl's completion manifest and every indexed shard) before anything is staged |
| `MODEL_PATH` | derived | the eval client's tokenizer, set to the served model; any other value is an error |
| `EVAL_CKPT_ROOT` | the run's `output_dir` | where the checkpoint job lists complete steps when `EVAL_MODEL_PATH` is missing |
| `EVAL_TP` | `8` | serving tensor parallel |
| `EVAL_SERVE_LEN` | `8192` | served context; must hold a whole multi-turn episode |
| `EVAL_GPU_UTIL` | `0.85` | vLLM memory fraction |
| `EVAL_STAGE` | `1` | copy the model from the Volume to local NVMe before serving |
| `EVAL_LOCAL_CACHE` | `/local_disk0/eval_model` | that NVMe directory |
| `EVAL_HEALTH_TIMEOUT` | `1800` | wait for `/health` |
| `EVAL_PORT` / `EVAL_MODEL` | `8000` / `eval` | endpoint and served name |
| `EVAL_SERVE_EXTRA_ARGS` | none | extra `vllm serve` flags |
| `EVAL_MAX_TURNS` | `8` | eval turn budget. It must be the same for the baseline and the trained eval (a test checks this). Search uses training's `12`; math deliberately uses `8` against training's `4`, as recorded in `eval_policy` |
| `EVAL_FORCE_FINAL_ANSWER` | `1` (search) | on the last turn, tell the model to answer and start its reply with `<answer>`. Training has no such turn; `0` makes the last turn an ordinary one. Recorded in `eval_policy` |
| `TOOL_FORMAT` | `qwen3_coder` | verl's parser for tool calls, the same as the training job's. Tool schemas come from verl's `@function_tool` registry, so the eval does not start without verl |
| `EVAL_LIMIT` | `0` (all) | number of questions; a small value makes a cheap smoke test |
| `EVAL_MAX_TOKENS` | `512` (search), `1024` (math) | cap per request (the math jobs set `3072`). Training has no per-turn cap, only the episode budget |
| `EVAL_TEMPERATURE` | `0` | greedy decoding, so the comparison is deterministic |
| `EVAL_CONCURRENCY` | `32` | questions in flight |
| `EVAL_MAX_CONT` | `2` (search), `3` (math) | continuation attempts on a truncated reply |
| `EVAL_REQ_TIMEOUT` / `EVAL_HTTP_RETRIES` | `600` to `900` / `4` | per-request timeout; retries only for transient failures (connection, timeout, HTTP 429/5xx) |
| `EVAL_EXPECT_N` | `EVAL_LIMIT` if set | how many questions the run must load; any other count makes it invalid (math ships `500`) |
| `EVAL_MAX_INFRA_ERRORS` | `0` | questions allowed to hit an infrastructure failure (inference, retrieval, tool, context limit) before the run is invalid. Those questions are never scored |
| `EVAL_OUT` | none | summary JSON. Never overwritten (`EVAL_OVERWRITE=1` forces it). It records `valid`, the served model's identity, the dataset fingerprint and the `eval_policy`. Each finished question is also written under `<EVAL_OUT>.parts/` |
| `EVAL_TRACE_OUT` | none | per-question JSONL traces, which `analyze_traces.py` reads |
| `EVAL_SPLIT` | none | search: a label for the question set (`dev` or `test`), recorded in the artifact |
| `EVAL_IDS_FILE` | none | search: evaluate exactly these MuSiQue ids, in this order |
| `EVAL_TOOLS` | `1` | search: `0` is the closed-book control (the bare question, no tools) |
| `EVAL_N_SAMPLES` | `1` | search: above 1, ask each question that many times and report how many groups have mixed rewards (the variance probe). Set `EVAL_TEMPERATURE` to the training temperature |

Every eval follows [`engine/serve/eval_contract.py`](../engine/serve/eval_contract.py): readiness is
checked before the first question, an outage counts as an infrastructure failure rather than a
wrong answer, and an invalid run is marked `"valid": false` and exits non-zero. Compare only valid
artifacts.

### Model staging (`engine/stage_model.py`)

| var | default | meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | HF repo to stage |
| `MODEL_REVISION` | `main` | branch, tag or commit, resolved once to the commit every file is fetched at and recorded in `<MODEL_DIR>/STAGED.json`. A directory that holds another revision is refused. Pin a commit for reproducible staging |
| `MODEL_DIR` | `<volume>/models/<basename>` | destination on the Volume |
| `SCRATCH_DIR` | `/local_disk0/hf_stage` | local directory each file is downloaded to before the Volume copy |
| `HF_TOKEN` | none | gated repos and rate limits |
| `HF_HUB_DISABLE_XET` | none | set `1` when writing to a UC Volume, which rejects the parallel writes Xet uses |

### geo3k infra jobs

| var | default | meaning |
|---|---|---|
| `GEO3K_OUT_DIR` | `<volume>/data/geo3k` | prep destination |
| `N_TRAIN` / `N_TEST` | `64` / `128` | subset size (`0` = the full split). The variance check needs `N_TEST >= N_PROMPTS` |
| `EVAL_FILE`, `N_PROMPTS`, `N_SAMPLES`, `TEMPERATURE`, `GEN_TP`, `MAX_TOKENS`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | see `infra/geo3k/baseline_eval.py` | the reward-variance check |

## Host configuration (`config.env`)

The `Makefile` reads `config.env`, and `make lint` fails while any job file disagrees with it.

| key | meaning |
|---|---|
| `AIR_PROFILE` | the Databricks CLI profile every `make` target passes as `-p`. Set your own; do not rely on the CLI's `DEFAULT` profile |
| `DOCKERHUB_USER` / `IMAGE_NAME` / `IMAGE_TAG` | image coordinates. `make retarget` writes them into every custom-image job file; `make bump` increments the tag and retargets |
| `SECRET_SCOPE` / `SECRET_KEY` | the Databricks secret with the registry credentials, so `make register` runs without prompting (the interactive fallback reads a TTY and hangs in CI) |
| `UC_CATALOG` / `UC_SCHEMA` / `UC_VOLUME` | the Unity Catalog Volume. Job files carry the resolved path literally so each can be submitted by hand; `make retarget` rewrites them |
| `VS_ENDPOINT` / `VS_INDEX` | the Vector Search endpoint and index; `make retarget` writes them into the search jobs as `QA_VS_ENDPOINT` and `QA_VS_INDEX` |
| `MAX_IMAGE_GB` | local image size limit: `19.5` decimal GB, under AI Runtime's 20 GB limit. `make size` also fails if the image is missing |
