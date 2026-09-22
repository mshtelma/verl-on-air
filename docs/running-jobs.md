# How to run everything

Operational guide: every job in this repo, in the order you would run it, with what it
needs, what it produces, and what to check. Companion docs:
[configuration.md](configuration.md) (what every setting does) ·
[training-modes.md](training-modes.md) (sync vs async) ·
[tuning.md](tuning.md) (which knobs to move) ·
[setup.md](setup.md) (first-time laptop → cluster setup) ·
[troubleshooting.md](troubleshooting.md) (symptom → fix).

---

## 0. The 60-second version

```bash
# one-time: image + credentials + volume       (see setup.md / build-linux.md)
make doctor && make image && make volume

# free checks, no GPU: linters + every job file against the real air CLI
make check

# cheapest possible proof the platform works
make smoke                                                       # 1xA10, ~2 min

# stage the base model ONCE (~70 GB) — everything else reads it from the Volume
air run --file infra/air/stage_model.yaml -p df1 --watch

# then pick a use case and walk its numbered jobs, e.g.
air run --file usecases/agentic-search/air/1_prep_data.yaml -p df1 --watch
```

Every job is submitted the same way:

```bash
air run --file <path/to/job.yaml> -p df1 --watch
```

`-p df1` is **not optional** — the CLI's DEFAULT profile is a different (dead)
credential. `--watch` streams the driver; drop it to submit and walk away.

---

## 1. Before your first run

| need | how | doc |
|---|---|---|
| `air` + `databricks` CLIs | `uv tool install --force databricks-air --python 3.12` | [setup.md](setup.md) |
| auth | `databricks auth login --host <df1-url> --profile df1` | [setup.md](setup.md) |
| the image, built + **registered** | `make image` (x86_64 Linux host required) | [build-linux.md](build-linux.md) |
| a UC Volume with ~150 GB | `make volume` | [setup.md](setup.md) |
| free GPU capacity | `air list runs --active -p df1` | §7 |

Then, before spending anything:

```bash
make check      # shellcheck + python compile + Dockerfile lint + `air run --dry-run`
                # on all 26 job files. Costs nothing, catches schema/topology/path errors.
make dry F=usecases/math/air/4_train.yaml     # one file only
DRY_RUN=1 bash engine/train/run_grpo_megatron.sh   # print the resolved verl overrides, locally
```

---

## 2. The job catalog

26 jobs: 15 under `infra/` (8 diagnostics + model staging + geo3k prep/baseline + 4 rungs),
6 for agentic-search, 5 for math. "GPUs" is `compute.num_accelerators`; **`timeout` is the
budget written in the file, not a measurement** — and it includes time spent queuing for
capacity (§7).

### infra — platform validation (run these first)

| job | GPUs | timeout | what it proves |
|---|---|---|---|
| `infra/diagnostics/air/smoke_test.yaml` | 1×A10 | 30 m | image imports, driver floor, on-device bf16 matmul, CPU RAM, compiler, `AutoBridge` resolves the model |
| `infra/diagnostics/air/diag_cuda.yaml` | 1×A10 | 20 m | CUDA visible, bf16 on device, driver ≥ floor |
| `infra/diagnostics/air/diag_te.yaml` | 1×A10 | 20 m | TransformerEngine multi-tensor path |
| `infra/diagnostics/air/env_probe.yaml` | 1×A10 | 20 m | what the runtime actually injects (env, PATH, venv) |
| `infra/diagnostics/air/probe_image_engines.yaml` | 1×A10 | 20 m | which model architectures this image's vLLM can serve |
| `infra/diagnostics/air/probe_tool_format.yaml` | 1×A10 | 30 m | **`TOOL_FORMAT` matches what your model emits** — run before any agentic job |
| `infra/diagnostics/air/probe_vllm_multinode.yaml` | 1×A10 | 20 m | how to serve one model across nodes with this vLLM |
| `infra/diagnostics/air/test_rollout_allreduce.yaml` | 8×H100 | 60 m | the vLLM custom-all-reduce crash + the two graph-preserving fixes |
| `infra/air/stage_model.yaml` | 1×A10 | 120 m | stages `Qwen3.5-35B-A3B` (~70 GB) to the Volume |
| `infra/geo3k/air/1_prep.yaml` | 1×A10 | 45 m | geo3k → parquet (64 train / 8 test) |
| `infra/geo3k/air/2_baseline.yaml` | 8×H100 | 90 m | **the reward-variance gate** — how much GRPO signal the data carries |
| `infra/geo3k/air/rung1_2b_fsdp_8gpu.yaml` | 8×H100 | 90 m | whole GRPO loop on a 2B dense model (measured ~911 s) |
| `infra/geo3k/air/rung2_9b_fsdp_8gpu.yaml` | 8×H100 | 150 m | 9B dense; co-located rollout memory starts to matter (~1000 s) |
| `infra/geo3k/air/rung3_35b_classic_8gpu.yaml` | 8×H100 | 300 m | 35B MoE, classic ZeRO-1 + CPU offload (~1477 s) |
| `infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml` | **32**×H100 | 360 m | 35B MoE, Megatron-FSDP, **no offload** (~1443 s) |

> The rung4 **file name says `16gpu`; the file requests 32**. 16 GPUs fits the
> *persistent* state but OOMs on the co-located weight-sync transient, so the passing
> configuration is 32. The name is kept only because `make rung4` points at it —
> see [ladder.md](ladder.md) and [sizing.md](sizing.md) for the byte-level story.

### usecases/agentic-search — the flagship (6 jobs)

| job | GPUs | timeout | what it does |
|---|---|---|---|
| `1_prep_data.yaml` | 1×A10 | 120 m | MuSiQue questions → parquet **+** the union passage corpus |
| `2_build_index.yaml` | 1×A10 | 60 m | Delta table + Vector Search index; **kicks off and exits** (§4.2) |
| `3_baseline_eval.yaml` | 8×H100 | 120 m | EVAL of the **base** model = the "before" number |
| `4_train.yaml` | 16×H100 | 600 m | GRPO, fully-async, judge-free, rule-based EM reward |
| `5_eval.yaml` | 8×H100 | 120 m | EVAL of a **checkpoint**, identical settings → the delta |
| `6_deploy.yaml` | 8×H100 | 60 m | prints the deployment recipe; `SERVE=1` brings up a vLLM endpoint |

### usecases/math — the judge-reward pattern (5 jobs)

| job | GPUs | timeout | what it does |
|---|---|---|---|
| `1_prep_data.yaml` | 1×A10 (stock env) | 45 m | MATH L3–5 → tool-agent parquet |
| `2_stage_judge.yaml` | 1×A10 | 600 m | stages the GLM-5.3 judge (~744 GB), resumable |
| `3_baseline_eval.yaml` | 8×H100 | 90 m | base model on MATH-500 (ships as a 32-question smoke) |
| `4_train.yaml` | **32**×H100 | 480 m | GRPO + **co-located judge**: 2 nodes train, 2 serve the judge |
| `5_eval.yaml` | 8×H100 | 90 m | the trained checkpoint, same settings |

### Makefile shortcuts

Every job has a target, so you never have to remember a path. `make help` prints them all
plus the resolved profile / image / volume.

```bash
# infra
make smoke prep stage baseline        # image pre-flight, data, model, variance gate
make rung1 rung2 rung3 rung4          # the ladder

# agentic-search, in order
make search-prep search-index search-baseline search-train
make search-eval CKPT=/Volumes/.../global_step_20/actor/model/huggingface
make search-deploy

# math, in order
make math-prep math-judge math-baseline math-train
make math-eval CKPT=/Volumes/.../global_step_24/actor/model/huggingface

# ops + free checks
make runs                             # active runs
make logs RUN=<id> [NODE=1]           # stream one node's log
make cancel RUN=<id>                  # multi-node jobs bill per node — cancel promptly
make check / lint / validate          # free pre-flight (no GPU, no submit)
make dry F=<job.yaml>                 # validate one job file
make config MODE=fsdp GPUS=16         # print resolved verl overrides locally
make diff-modes                       # diff fsdp vs classic override sets
```

The `make` targets all pass `--watch`. For anything else — overrides, submitting without
watching — use `air run` directly as shown throughout this page.

---

## 3. Path A — prove the platform (do this once per image / workspace)

Cheapest first. A failure at the top costs GPU-minutes; the same failure found inside a
training run costs GPU-hours.

```bash
# 1. image + host facts (~2 min, 1 GPU)
make smoke

# 2. only if you are going to run an AGENTIC job — the silent killer
air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch

# 3. data + model
make prep       # geo3k -> parquet
make stage      # Qwen3.5-35B-A3B -> Volume (do this ONCE; ~70 GB)

# 4. is there any GRPO signal in this data at all?
make baseline

# 5. the ladder: one new risk per rung
make rung1 && make rung2 && make rung3 && make rung4
```

What to read out of `make smoke`: `driver supports CUDA 13`, `CUDA 13 wheels run on this
base` (a real on-device matmul), `no cuda-compat shadowing`, `cpu ram` (decides whether
`OFFLOAD=1` is viable — it needs ~400–500 GB/node), `AutoBridge resolves the model`
(without it `MEGATRON_MODE=fsdp` cannot work), `C compiler on PATH` (Qwen3.5's
Gated-DeltaNet layers JIT a Triton stub at runtime).

**`make baseline` is the step people skip and shouldn't.** GRPO's gradient comes
entirely from reward *variance* within each group of `rollout_n` samples; a group where
every sample scores the same contributes nothing. That job reports the fraction of
groups with non-zero variance — your effective batch size. `pass@1` will not tell you.

---

## 4. Path B — agentic-search, end to end

The flagship: train a multi-hop search agent with a rule-based exact-match reward.
16×H100 for training, 8 for each eval.

### 4.1 Data + corpus

```bash
air run --file usecases/agentic-search/air/1_prep_data.yaml -p df1 --watch
```

Produces, on the Volume: `data/qa_musique/{train,test}.parquet` (questions) and
`corpus_big.parquet` (the passages to retrieve from).

### 4.2 The Vector Search index — the one asynchronous step

```bash
air run --file usecases/agentic-search/air/2_build_index.yaml -p df1 --watch
```

This job **returns before the index is ready, on purpose**. It creates the Delta table
(Change Data Feed on) and the Delta-Sync index with the managed
`databricks-gte-large-en` embedding model, then exits — embedding ~1M passages takes
longer than a sensible job window, and Vector Search keeps provisioning server-side
whether or not the job is alive.

Wait for the index to report ready before the next step:

```bash
databricks vector-search-indexes get-index main.mshtelma.wiki_qa_big_corpus_index \
  -p df1 --output json | python3 -c 'import json,sys; s=json.load(sys.stdin)["status"]; \
  print(s["ready"], s["indexed_row_count"], s["message"])'
```

Wait for `ready True`. `indexed_row_count` tells you how far the initial snapshot has
got, so you can see progress rather than guess.

Sanity-check access with `usecases/agentic-search/probe_vs_access.py` before paying for
a GPU job — jobs in the same workspace use ambient auth, so no token is needed.

### 4.3 The baseline — the "before" number

```bash
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch
```

Serves the **base** model (vLLM TP8) and runs the same agentic loop and the same scorer
training will use. Writes `eval/musique_base.json` plus
`eval/musique_base_traces.jsonl`.

**`EVAL_MAX_TURNS` must be identical here and in `5_eval.yaml`.** Turn budget alone
moves the number (8 → 12 turns lifted the *base* model ~2 EM), so a mismatch produces a
"training improvement" that is really a budget difference. Both files ship `'12'`.

Cheap first pass: `--override env_variables.EVAL_LIMIT=20` to validate the harness
(tools firing, answers parsed) before the full 200.

### 4.4 Train

```bash
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch
```

2 nodes / 16×H100, fully-async: one whole node generates (`ROLLOUT_NNODES=1`), the
other trains. No judge (`TRAINING_NODES` equals the node count) because the reward is a
pure rule. `SAVE_FREQ: '10'` writes a checkpoint every 10 weight syncs to
`ckpt/agentic-search-grpo/global_step_N/actor/model/huggingface/`.

Common overrides:

```bash
# shorter run
--override parameters.total_rollout_steps=800
# the headline lever
--override env_variables.MAX_TURNS=16
# denser GRPO groups (more compute per prompt)
--override parameters.rollout_n=32
# switch to synchronous/on-policy training (see training-modes.md)
--override env_variables.TRAIN_MODE=sync env_variables.ROLLOUT_NNODES=0
```

### 4.5 Evaluate a checkpoint

```bash
air run --file usecases/agentic-search/air/5_eval.yaml -p df1 --watch \
  --override env_variables.MODEL_PATH=/Volumes/main/mshtelma/verl/ckpt/agentic-search-grpo/global_step_20/actor/model/huggingface \
             env_variables.EVAL_MODEL_PATH=/Volumes/main/mshtelma/verl/ckpt/agentic-search-grpo/global_step_20/actor/model/huggingface \
             env_variables.EVAL_OUT=/Volumes/main/mshtelma/verl/eval/agentic_search_step20.json \
             env_variables.EVAL_TRACE_OUT=/Volumes/main/mshtelma/verl/eval/agentic_search_step20_traces.jsonl
```

Evaluate **several** checkpoints. Training reward is not the deliverable and the best
held-out checkpoint is usually not the last one (here it was step 20, with a plateau
after). Keep `EVAL_OUT`/`EVAL_TRACE_OUT` distinct per step or you will overwrite them.

### 4.6 Understand the result, don't just report it

```bash
python3 usecases/agentic-search/analyze_traces.py <base_traces.jsonl> <trained_traces.jsonl>
```

Decomposes `EM = recall × conversion` — did retrieval surface the gold passage, and given
that it did, did the model answer correctly? That tells you *which* knob to move:
turns/retrieval protect recall, GRPO improves conversion. Full narrative in
[../RESULTS.md](../RESULTS.md).

### 4.7 Deploy

```bash
air run --file usecases/agentic-search/air/6_deploy.yaml -p df1 --watch              # prints the recipe
air run --file usecases/agentic-search/air/6_deploy.yaml -p df1 --override env_variables.SERVE=1
```

Two paths, deliberately kept as a spec: **(A)** register the HF export as a Unity
Catalog model and create a Provisioned Throughput serving endpoint, running the agent
tool-loop in your application layer; **(B)** `SERVE=1` brings the checkpoint up as an
OpenAI-compatible vLLM endpoint on a GPU node for an internal demo. Pick the
checkpoint with `MODEL_PATH`.

---

## 5. Path C — math, end to end (the judge pattern)

Same shape, one extra concern: the reward is an **LLM judge that this job serves
itself**.

```bash
# 1. data (stock environment — no custom image, no GPU work)
air run --file usecases/math/air/1_prep_data.yaml -p df1 --watch

# 2. stage the judge model, once (~744 GB, resumable, max_retries=3)
air run --file usecases/math/air/2_stage_judge.yaml -p df1 --watch

# 3. baseline: ships as a 32-question smoke; run the full 500 for the real number
air run --file usecases/math/air/3_baseline_eval.yaml -p df1 --watch
air run --file usecases/math/air/3_baseline_eval.yaml -p df1 --watch \
  --override env_variables.EVAL_LIMIT=0

# 4. train: 4 nodes = 2 training + 2 judge
air run --file usecases/math/air/4_train.yaml -p df1 --watch

# 5. eval a checkpoint with the SAME eval settings as step 3
air run --file usecases/math/air/5_eval.yaml -p df1 --watch \
  --override env_variables.MODEL_PATH=<ckpt>/actor/model/huggingface \
             env_variables.EVAL_MODEL_PATH=<ckpt>/actor/model/huggingface
```

**How the judge co-location works.** df1 has no cross-*job* connectivity and one image
per job, so trainer and judge must live in one job.
`compute.num_accelerators: 32` gives 4 nodes; `TRAINING_NODES: '2'` sends ranks 0–1 to
GRPO and ranks 2–3 to `engine/serve/serve_judge.sh`, which serves the judge at TP=16 and
publishes its URL to a Unity Catalog rendezvous file. The reward function re-reads that
URL **at call time** (a Ray actor does not reliably inherit the driver's exports), and
rank 0 writes a `training_done` sentinel on exit so the judge shuts itself down.

Three judge settings that matter more than the rest:

- `REWARD_MAX_CONCURRENT` — verl's internal default is **1** (serial). Unset, the judge
  becomes the bottleneck for the whole run. The job sets `64`.
- `NORM_ADV_BY_STD_IN_GRPO=False` — the judge score is *graded*. With GRPO's
  std-normalisation on, 0.05 and 1.0 collapse to the same advantage.
- `JUDGE_TRAJECTORY_CHARS` — truncate too aggressively and the judge grades work it
  cannot see.

Watch `JUDGE_HEALTH_TIMEOUT` on the first run: a 744 GB first load is slow, and the
training ranks fail after `JUDGE_WAIT_TIMEOUT` if the endpoint never appears.

---

## 6. Running the same use case in a different training mode

The mode is one env var — `TRAIN_MODE=async|sync` — read by
`engine/train/dispatch_agentic.sh`. Nothing else about the job changes: same tool, same
reward, same data.

```bash
# fully-async (default): disjoint Rollouter/Trainer GPU pools, generation overlaps training
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch

# synchronous / on-policy: rollout and training co-located on the same GPUs
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch \
  --override env_variables.TRAIN_MODE=sync \
             env_variables.ROLLOUT_NNODES=0 \
             compute.num_accelerators=32
```

`ROLLOUT_NNODES=0` is required in sync mode (there is no separate rollout pool to carve
out) and the dispatcher fails loudly if you forget. Read
[training-modes.md](training-modes.md) before running the sync variant — it explains the
trade-off, why the co-located 35B config needs more GPUs, and exactly which parts of
each mode have been measured versus dry-run-validated.

---

## 7. Watching, debugging, and not wasting money

```bash
air list runs --active -p df1                 # what is running (yours and others')
air get run <run_id> -p df1                   # status, duration, topology
air logs <run_id> -p df1                      # driver (rank 0)
air logs <run_id> -p df1 --node 1             # the other node — where Ray-join failures show
air logs <run_id> -p df1 -v                   # verbose diagnostics on a failure
air cancel <run_id> -p df1                    # multi-node bills per node
```

**Per-step metrics are in MLflow, not the driver log** — in fully-async mode the trainer
logs from a Ray worker actor. Experiments group under the job's
`mlflow_experiment_directory`. To pull a metric history:

```bash
databricks api get "/api/2.0/mlflow/metrics/get-history?run_id=<mlflow_run_id>&metric_key=perf/throughput" -p df1
```

Things worth knowing before you interpret a red run:

- **`TIMEDOUT` with no node logs usually means capacity, not a bug.** `timeout_minutes`
  covers queue time; if the status sits on `Waiting for GPU compute capacity to become
  available`, the fix is a bigger timeout or a less contended accelerator type. Check
  `air get run <id>` for the state history.
- **A fully-async run that finishes cleanly can still exit non-zero.** When the trainer
  completes it cancels the rollouter, which surfaces as a `RayTaskError`. The launcher
  detects this (completion markers / benign-teardown / a *new* checkpoint) and converts
  it to success, while a hard-failure veto (OOM, NCCL, CUDA, assert, engine-init) keeps
  real crashes red. If you see `Treating as SUCCESS` in the log, that is this guard.
- **`--watch` needs a TTY.** In a non-interactive shell it can exit non-zero with empty
  output; submit without `--watch` and poll `air get run` instead.
- **A geo3k rung printing "2/3 steps" and SUCCESS is correct.** The staged subset is 64
  train examples at `train_batch_size=32` = 2 steps/epoch, with
  `total_training_steps: 3` as an unreached ceiling. Set it to `0` for a real run.

Grep a training log for: `Selected Provider is efa` (RDMA active on AWS — `NET/Socket`
instead means a TCP fallback and collapsed throughput), `Failed to decode tool call`
(wrong `TOOL_FORMAT` — your agent is not using its tools), `No available memory for the
cache blocks` (lower `ROLLOUT_GPU_MEM_UTIL` or the episode length).

---

## 8. What lands where

```
/Volumes/main/mshtelma/verl/
├── models/Qwen3.5-35B-A3B/           # staged base model         (stage_model.yaml)
├── models/GLM-5.3/                   # staged judge              (math/2_stage_judge.yaml)
├── data/geo3k/{train,test}.parquet   # infra ladder data         (geo3k/1_prep.yaml)
├── data/qa_musique/*.parquet         # agentic-search questions  (1_prep_data.yaml)
├── data/math_tool/*.parquet          # math questions            (1_prep_data.yaml)
├── ckpt/<experiment>/global_step_N/actor/model/huggingface/   # what the eval jobs serve
├── eval/*.json                       # eval summaries            (EVAL_OUT)
├── eval/*_traces.jsonl               # per-question traces       (EVAL_TRACE_OUT)
└── rendezvous/<master>_<port>/       # judge endpoint + training_done sentinels
```

Checkpoints dominate storage. Keep `SAVE_FREQ` deliberate rather than saving every
sync, and prune old `global_step_*` directories once you have evaluated them.

---

## 9. Bringing your own task

Copy a use case, replace five files, change no engine code:
**[new-usecase.md](new-usecase.md)**.
