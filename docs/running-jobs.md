# Running the jobs

Every job in the repo in the order you would run it: what it needs, what it produces and what
to check. Settings are in [configuration.md](configuration.md), the choice of training mode in
[training-modes.md](training-modes.md), first-time setup (CLIs, auth, image, Volume) in
[setup.md](setup.md).

## Quick start

```bash
make doctor && make image && make volume     # one-time: image, credentials, volume
make dev-env && make check                   # lint, CPU tests, verl composition, job files (free)
make preflight F=usecases/agentic-search/air/4_train.yaml   # one job's plan and GPU-hour bound
make smoke                                   # 1xA10, ~2 min
air run --file infra/air/stage_model.yaml -p <profile> --watch   # base model, once (~70 GB)
```

Every job is submitted the same way: `air run --file <job.yaml> -p <profile> --watch`. Always
pass the profile; without `-p` the CLI falls back to your `DEFAULT` profile, which may point at
another workspace or hold an expired token (`make` passes `AIR_PROFILE` from `config.env`). Drop
`--watch` to submit and walk away. Check free capacity with `air list runs --active -p <profile>`.
`make dry F=<job.yaml>` validates one file; `DRY_RUN=1 bash engine/train/run_grpo_megatron.sh`
prints the resolved verl overrides locally.

## The jobs

There are 26 job files: 15 under `infra/` (8 diagnostics, model staging, geo3k prep and
baseline, 4 ladder rungs), 6 for agentic-search and 5 for math. GPUs is
`compute.num_accelerators`. The timeout is the budget written in the file, not a measurement,
and it includes time spent waiting for capacity. Gates exit non-zero unless their check passes
and end with a `PROBE_VERDICT {...}` JSON line; the other diagnostics only measure
([infra/README.md](../infra/README.md)).

| job | GPUs | timeout | what it checks |
|---|---|---|---|
| `infra/diagnostics/air/smoke_test.yaml` | 1×A10 | 30 m | gate: image imports, driver floor, on-device bf16 matmul, CPU RAM, compiler, `AutoBridge` resolves the model |
| `infra/diagnostics/air/diag_cuda.yaml` | 1×A10 | 20 m | CUDA visible, bf16 on device, driver at or above the floor |
| `infra/diagnostics/air/diag_te.yaml` | 1×A10 | 20 m | TransformerEngine multi-tensor path |
| `infra/diagnostics/air/env_probe.yaml` | 1×A10 | 20 m | what the runtime injects (env, PATH, venv) |
| `infra/diagnostics/air/probe_image_engines.yaml` | 1×A10 | 20 m | which model architectures this image's vLLM can serve |
| `infra/diagnostics/air/probe_tool_format.yaml` | 1×A10 | 30 m | gate: `TOOL_FORMAT` parses what your model's chat template writes; run it before any agentic job |
| `infra/diagnostics/air/probe_vllm_multinode.yaml` | 1×A10 | 20 m | how to serve one model across nodes with this vLLM |
| `infra/diagnostics/air/test_rollout_allreduce.yaml` | 8×H100 | 60 m | vLLM's custom all-reduce crash and the two fixes that keep CUDA graphs |
| `infra/air/stage_model.yaml` | 1×A10 | 120 m | stages `Qwen3.5-35B-A3B` (~70 GB) to the Volume |
| `infra/geo3k/air/1_prep.yaml` | 1×A10 | 45 m | geo3k to parquet (64 train, 128 test) |
| `infra/geo3k/air/2_baseline.yaml` | 8×H100 | 90 m | reward-variance gate: how much GRPO signal the data carries |
| `infra/geo3k/air/rung1_2b_fsdp_8gpu.yaml` | 8×H100 | 90 m | full GRPO loop on a 2B dense model (~911 s) |
| `infra/geo3k/air/rung2_9b_fsdp_8gpu.yaml` | 8×H100 | 150 m | 9B dense; co-located rollout memory starts to matter (~1000 s) |
| `infra/geo3k/air/rung3_35b_classic_8gpu.yaml` | 8×H100 | 300 m | 35B MoE, classic ZeRO-1 with CPU offload (~1477 s) |
| `infra/geo3k/air/rung4_35b_fsdp_32gpu.yaml` | 32×H100 | 360 m | 35B MoE, Megatron-FSDP, no offload (~1443 s); on 16 GPUs the co-located weight sync runs out of memory ([ladder.md](ladder.md)) |

| agentic-search job | GPUs | timeout | what it does |
|---|---|---|---|
| `1_prep_data.yaml` | 1×A10 | 120 m | MuSiQue questions to parquet, plus the passage corpus |
| `2_build_index.yaml` | 1×A10 | 60 m | versioned Delta table and Vector Search index; needs `WAREHOUSE_ID`; starts the build and exits |
| `3_baseline_eval.yaml` | 8×H100 | 120 m | evaluates the base model (the "before" number) |
| `4_train.yaml` | 16×H100 | 600 m | GRPO, fully-async, rule-based exact-match reward, no judge |
| `4_train_sync.yaml` | 32×H100 | 600 m | the same run, synchronous. Experimental: it runs out of memory as configured |
| `5_eval.yaml` | 8×H100 | 120 m | evaluates a checkpoint with the baseline's settings |

| math job | GPUs | timeout | what it does |
|---|---|---|---|
| `1_prep_data.yaml` | 1×A10 (stock env) | 45 m | MATH levels 3-5 to tool-agent parquet |
| `2_stage_judge.yaml` | 1×A10 | 600 m | stages the GLM-5.3 judge (~744 GB), resumable |
| `3_baseline_eval.yaml` | 8×H100 | 90 m | base model on all 500 MATH-500 problems |
| `4_train.yaml` | 32×H100 | 480 m | GRPO with the judge in the same job: 2 nodes train, 2 serve the judge |
| `5_eval.yaml` | 8×H100 | 90 m | a trained checkpoint, same settings as the baseline |

Every job also has a `make` target (`make help` lists them with the resolved profile, image and
volume), for example `make search-train` or `make search-eval CKPT=<output_dir>/<RUN_ID>/global_step_20`.
They all pass `--watch`. The rungs and training targets print the job's GPU-hour upper bound and
submit only with `BUDGET_OK=1`. For overrides, call `air run` directly.

## Prove the platform (once per image or workspace)

A failure here costs GPU-minutes; the same failure inside a training run costs GPU-hours.

```bash
make smoke      # image and host facts (~2 min, 1 GPU)
make prep       # geo3k to parquet
make stage      # Qwen3.5-35B-A3B to the Volume, once
air run --file infra/diagnostics/air/probe_tool_format.yaml -p <profile> --watch   # reads the staged model's template
make baseline   # is there GRPO signal in this data?
make rung1 && make rung2 && make rung3 && make rung4
```

In the smoke output, check `driver supports CUDA 13`, `CUDA 13 wheels run on this base` (a real
matmul on the device), `no cuda-compat shadowing`, `cpu ram` (`OFFLOAD=1` needs ~400-500 GB per
node), `AutoBridge resolves the model` (without it `MEGATRON_MODE=fsdp` cannot work) and
`C compiler on PATH` (Triton compiles a launcher stub for Qwen3.5's Gated-DeltaNet layers at
runtime).

Don't skip `make baseline`. GRPO learns only from reward differences within a group of
`rollout_n` samples, and the job reports the fraction of groups with any spread (`pass@1` won't).

## Agentic search, end to end

```bash
air run --file usecases/agentic-search/air/1_prep_data.yaml -p <profile> --watch
make search-index WAREHOUSE_ID=<id>          # or 2_build_index.yaml with env_variables.QA_VS_WAREHOUSE_ID=<id>
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p <profile> --watch
air run --file usecases/agentic-search/air/4_train.yaml -p <profile> --watch
make search-eval CKPT=<output_dir>/<RUN_ID>/global_step_20
```

Prep writes `data/qa_musique/{train,test}.parquet` and `data/qa_musique/corpus_big.parquet`,
each with a manifest recording the pinned source revisions, row counts and file hashes. If a
corpus source fails to load, the job fails instead of writing a smaller corpus.

The index job needs a SQL warehouse (none is guessed) and an existing Vector Search endpoint
(`QA_VS_ENDPOINT`; set `QA_VS_CREATE_ENDPOINT: '1'` to create one, which is billable and
persistent). The table and index are named after the corpus content hash
(`wiki_qa_big_corpus_v<h8>`, `wiki_qa_big_corpus_v<h8>_index`), so a rebuilt corpus never
changes an index a job is using; an existing one with the same name is verified and reused. The
job prints the `QA_VS_INDEX` to put into the train and eval jobs. The shipped job files still
name the index the published results used (`wiki_qa_big_corpus_index`, same corpus, built
before versioning).

The job returns before the index is ready: it loads the table (Change Data Feed on), creates a
Delta Sync index with the managed `databricks-gte-large-en` embeddings and exits, since
embedding ~600k passages takes longer than a job should wait. The index is ready when it is
ONLINE and `indexed_row_count` equals the passage count the job printed:

```bash
databricks api get /api/2.0/vector-search/indexes/<QA_VS_INDEX> -p <profile> | python3 -c \
  'import json,sys; s=json.load(sys.stdin)["status"]; print(s["detailed_state"], s["indexed_row_count"])'
```

Inside a job, `create_vs_index.py --status-only` (exit 0 when ready, 3 when not) and
`--wait-only` do the same. Before a GPU job, `QA_VS_INDEX=<index> python3
usecases/agentic-search/probe_vs_access.py` checks access: it fails unless an ANN and a HYBRID
query return rows, and it is read-only, so it runs locally with `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` set.

The baseline serves the base model with vLLM (TP=8) and writes `eval/musique_base.json` and its
traces. `EVAL_MAX_TURNS` must match in `3_baseline_eval.yaml` and `5_eval.yaml` (both ship
`12`): the turn budget alone moves the score. `--override env_variables.EVAL_LIMIT=20` gives a
cheap harness check first.

Training uses two nodes, one generating (`ROLLOUT_NNODES=1`) and one training, and no judge
(`TRAINING_NODES` equals the node count). It plans 3200 prompt groups, which is 100 weight
syncs, and `SAVE_FREQ: '10'` saves every 10 syncs to
`ckpt/agentic-search-grpo/<RUN_ID>/global_step_N/`. Common overrides:
`parameters.total_rollout_steps=800` (shorter), `env_variables.MAX_TURNS=16` (the hop budget),
`parameters.rollout_n=32` (larger groups). Synchronous training is its own job file
(`make search-train-sync`); see [training-modes.md](training-modes.md).

Evaluate several checkpoints on the dev set (the default questions) and pick one there; the best
is often not the last. Then score that one checkpoint once on the held-out test split, which
`python3 usecases/agentic-search/make_splits.py --out <volume>/data/qa_musique/heldout_test.parquet`
writes:

```bash
air run --file usecases/agentic-search/air/5_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_MODEL_PATH=<output_dir>/<RUN_ID>/global_step_20 \
             env_variables.QA_VAL_PARQUET=<volume>/data/qa_musique/heldout_test.parquet \
             env_variables.EVAL_SPLIT=test env_variables.EVAL_LIMIT=0 env_variables.EVAL_EXPECT_N=500 \
             env_variables.EVAL_OUT=<volume>/eval/test_step20.json \
             env_variables.EVAL_TRACE_OUT=<volume>/eval/test_step20_traces.jsonl
```

Picking on the dev set inflates the dev number, not the test number. Eval artifacts are never
overwritten, so each eval needs its own `EVAL_OUT` (`make search-eval` names them for you).

`analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl> --label base --label step20 --out diag/`
splits EM as `P(S)·P(correct | S) + P(¬S)·P(correct | ¬S)`, where S means a gold answer string
appeared in some tool output, and pairs the runs question by question. S is a proxy, not
supporting-passage recall (`--supporting-from-musique` adds that), so use it to pick the next
experiment. Results: [RESULTS.md](../RESULTS.md). Deployment: [deploy.md](deploy.md) (not implemented).

## Math, end to end

Same shape, but the reward is an LLM judge that the training job serves itself.

```bash
air run --file usecases/math/air/1_prep_data.yaml -p <profile> --watch     # stock env, no custom image
air run --file usecases/math/air/2_stage_judge.yaml -p <profile> --watch   # once, ~744 GB, resumable
air run --file usecases/math/air/3_baseline_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_LIMIT=32 env_variables.EVAL_EXPECT_N=32 \
             env_variables.EVAL_OUT=<volume>/eval/math500_base_smoke.json   # optional smoke
air run --file usecases/math/air/3_baseline_eval.yaml -p <profile> --watch  # all 500 problems
air run --file usecases/math/air/4_train.yaml -p <profile> --watch
air run --file usecases/math/air/5_eval.yaml -p <profile> --watch \
  --override env_variables.EVAL_MODEL_PATH=<output_dir>/<RUN_ID>/global_step_24
```

Jobs cannot reach each other over the network, so the trainer and the judge share one job.
`compute.num_accelerators: 32` gives 4 nodes; `TRAINING_NODES: '2'` sends ranks 0-1 to GRPO and
ranks 2-3 to `engine/serve/serve_judge.sh`, which serves the judge at TP=16 and publishes its
URL to a rendezvous file on the Volume. The reward reads that URL when it is called, and rank 0
writes a `training_done` file on exit so the judge shuts down.

- `REWARD_MAX_CONCURRENT`: verl's default is 1, which makes the judge the bottleneck. The job
  sets 64 per reward worker, and verl runs 8 workers, so up to 512 judge calls are in flight.
- `NORM_ADV_BY_STD_IN_GRPO: 'True'` is verl's default and what the math runs used. Whether
  `False` suits a graded judge score better is untested ([tuning.md](tuning.md)).
- `JUDGE_MAX_FAIL_RATE` is the judge failure budget: if the judge stops answering, the run is
  aborted rather than training on the rule-based fallback. Watch `judge_valid`, and
  `judge_input_truncated` for trajectories longer than `JUDGE_TRAJECTORY_CHARS`.
- Loading 744 GB is slow; watch `JUDGE_HEALTH_TIMEOUT` on the first run. Training waits
  `JUDGE_WAIT_TIMEOUT` (stage + health timeouts + 600 s, printed at start), checks that the judge
  answers (`engine/serve/judge_ping.py`), then runs the use case's `PRE_TRAIN_CHECK`.

## Monitoring

```bash
air get run <run_id> -p <profile>            # status, duration, topology, state history
air logs <run_id> -p <profile>               # driver (rank 0)
air logs <run_id> -p <profile> --node 1      # another node; Ray join failures show up here
air logs <run_id> -p <profile> -v            # verbose, for a failure
air cancel <run_id> -p <profile>             # multi-node jobs bill per node
```

Per-step metrics go to MLflow under `experiment_name`, not to the driver log (in fully-async
mode the trainer logs from a Ray actor). One metric's history:
`databricks api get "/api/2.0/mlflow/metrics/get-history?run_id=<mlflow_run_id>&metric_key=perf/throughput" -p <profile>`.

- A run's verdict is `run_result.json` in its output directory, and the log ends with
  `[certificate] CERTIFIED` or `NOT CERTIFIED` and the reasons. A run passes only if it wrote its
  planned final checkpoint, the checkpoint verifies and nothing raised an abort. Ignore verl's
  exit code in fully-async mode: a clean finish exits non-zero and a crash can exit 0. Sync runs
  are certified the same way, except that a non-zero exit always fails.
- `air cancel` stops the containers without a signal the scripts can catch, so a cancelled run
  writes no `run_result.json`; its `CANCELED` status is the record. Timeouts and lost nodes do
  send SIGTERM and are recorded as `rc=143 signal=TERM`.
- `TIMEDOUT` with no node logs usually means the job waited for capacity, since
  `timeout_minutes` includes queue time.
- `--watch` needs a TTY. In a non-interactive shell it can exit non-zero with no output; submit
  without it and poll `air get run`.
- A geo3k rung that prints "2/3 steps" and SUCCESS is fine: 64 examples at `train_batch_size=32`
  is 2 steps per epoch, and `total_training_steps: 3` is only a ceiling (`0` removes it).

Grep training logs for `Selected Provider is efa` (EFA/RDMA in use; `NET/Socket` means NCCL fell
back to TCP), `Failed to decode tool call` (wrong `TOOL_FORMAT`: the agent is not using its
tools) and `No available memory for the cache blocks` (vLLM's KV cache cannot hold one
`MAX_MODEL_LEN` sequence; fixes in [tuning.md](tuning.md)).

## What lands where

```
/Volumes/<catalog>/<schema>/<volume>/
├── models/Qwen3.5-35B-A3B/        base model                      (stage_model.yaml)
├── models/GLM-5.3/                math judge                      (math/2_stage_judge.yaml)
├── data/geo3k/                    ladder data                     (geo3k/1_prep.yaml)
├── data/qa_musique/               search questions, corpus, heldout_test.parquet
├── data/math_tool/                math questions                  (math/1_prep_data.yaml)
├── ckpt/<experiment>/<RUN_ID>/    run_manifest.json, run_result.json,
│                                  global_step_N/actor/model/huggingface/ (what evals serve)
├── eval/                          eval summaries (EVAL_OUT) and traces (EVAL_TRACE_OUT)
└── rendezvous/<RUN_ID>/           judge endpoint, training_done, ABORT.json, head heartbeat
```

Checkpoints take most of the space: choose `SAVE_FREQ` deliberately, and after evaluating a run
trim it with `make prune-ckpts CKPT=<output_dir>/<RUN_ID>` (it lists what it would delete
until you add `CONFIRM=1`). Your own task: [new-usecase.md](new-usecase.md).
