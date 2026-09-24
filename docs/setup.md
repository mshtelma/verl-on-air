# Setup

From an empty laptop to a multi-node GRPO run on a Databricks workspace with AI Runtime.
`<profile>` is your Databricks CLI profile, the `AIR_PROFILE` in `config.env`. Once this works,
[running-jobs.md](running-jobs.md) goes through every job.

## Prerequisites

```bash
uv tool install --force databricks-air --python 3.12
air --version && databricks --version && docker --version

databricks auth login --host https://<workspace-url> --profile <profile>
docker login
```

Log in again even if `~/.databrickscfg` has the profile: recent CLI versions reject the old
credential cache, and `air` fails until you do.

GPU quota is per accelerator type. Rung 4 needs 4 free `GPU_8xH100` nodes, the agentic-search
training job 2, the math job 4 (judge included); over the quota, a submit fails with "Workspace
has exceeded its GPU quota".

Check these permissions once, as the identity that submits jobs:

| what | needed by | check |
|---|---|---|
| AI Runtime serverless GPU, `GPU_8xH100` quota | every GPU job | `air list runs -p <profile>` works |
| `USE CATALOG`, `USE SCHEMA`, `CREATE VOLUME` on the schema | `make volume` | `databricks schemas get <catalog>.<schema> -p <profile>` |
| `READ VOLUME`, `WRITE VOLUME` on the Volume | data, models, checkpoints, evals, rendezvous files | `databricks fs ls dbfs:/Volumes/<catalog>/<schema>/<volume> -p <profile>` |
| `CREATE TABLE` on the schema, `CAN USE` on a SQL warehouse | `make search-index` loads the corpus table | `databricks warehouses list -p <profile>` |
| a Vector Search endpoint, or the right to create one | the index build; search training and eval | `databricks vector-search-endpoints list-endpoints -p <profile>` |
| the embedding endpoint `databricks-gte-large-en` | the index build | `databricks serving-endpoints get databricks-gte-large-en -p <profile>` |
| an MLflow experiment location you can write to | training metrics | the job's `mlflow_url` opens |
| a secret scope you can write to | `air register` (Docker Hub credential); optional `HF_TOKEN` | `databricks secrets list-scopes -p <profile>` |

Nothing is created for you except the Volume (`make volume`) and, if you set
`QA_VS_CREATE_ENDPOINT: '1'`, a Vector Search endpoint, which is billable and persistent.

## 1. The Volume

`make volume` creates `/Volumes/<catalog>/<schema>/<volume>` from `UC_CATALOG`, `UC_SCHEMA` and
`UC_VOLUME` in `config.env`. The catalog and schema must exist already.

| what | size | notes |
|---|---|---|
| base model `models/Qwen3.5-35B-A3B` | ~70 GB | staged once (`make stage`) |
| LLM judge `models/GLM-5.3` (math only) | ~744 GB | staged once (`make math-judge`), cached per node on `/local_disk0` |
| data (geo3k, MuSiQue, the passage corpus) | < 1 GB | |
| one 35B training checkpoint | ~0.5 TB (estimate) | HF export (70 GB, measured), bf16 Megatron state (~70 GB), fp32 Adam state (~420 GB) |
| a training run | checkpoint size × `MAX_CKPT_TO_KEEP` (default 3) | `make prune-ckpts CKPT=<run>` trims it afterwards |

Job files carry the Volume path, the image and the Vector Search names literally, so each one
can be submitted by hand. After changing any of them in `config.env`, run `make retarget`;
`make lint` fails while a job file disagrees.

## 2. The image

```bash
make validate   # every job file against the air CLI (free, no image needed)
make build      # linux/amd64, on an x86_64 Linux host (build-linux.md)
make size       # fails above 19.5 GB; the platform rejects images over 20 GB
make push
make register   # 2-6 min
```

`make image` runs the last four in order. `make validate` swaps in a stock environment before
`air run --dry-run`, so job files are checked before the image exists. The build imports the
whole stack in its last layer, so a wrong `transformers` pin fails the build, not a GPU job. It
compiles no CUDA code (the CUDA extensions are prebuilt wheels from verl's wheelhouse), which
keeps the image on the runtime base and under the limit.

A private Docker Hub repository needs credentials to register, and `air` keeps them in a
Databricks secret. Create it once, interactively:

```bash
air register image <dockerhub-user>/verl-megatron-air:<tag> -p <profile>
# Docker registry username: <dockerhub-user>
# Docker registry password/PAT: ****
# Databricks secret scope name [docker-credentials-...]: <scope>
# Databricks secret key name  [dockerio-...]:            <key>
```

Then set `SECRET_SCOPE=<scope>` and `SECRET_KEY=<key>` in `config.env`, and `make register`
passes them as `--scope/--key`. Without them it falls back to the interactive flow, which reads
the terminal and hangs in CI. Rotate the secret with the interactive flow too; its format is
internal to `air`.

The base image, `databricksruntime/air:dcs-base-aws-runtime-cu13` (CUDA 13.0.3), is published
for AWS workspaces only. verl's wheelhouse builds TransformerEngine, apex and flash-attn for
cu130 only, so a CUDA 12 stack would mean compiling them and overshooting the size limit.

## 3. Smoke test

```bash
make smoke      # 1×A10, ~2 min
```

It checks the image and the driver floor on an A10; read H100 host facts (RAM, EFA, NVLink) on an
H100 job, where the first rung prints them.

| line | what it tells you |
|---|---|
| `image tag baked at build time` | the tag the image was built with; if it differs from `config.env`, the job runs an old image |
| `driver supports CUDA 13` | the driver can run the cu130 stack (CUDA 13.0 needs ≥ 580.65.06) |
| `CUDA works on device` | a real bf16 matmul, where an ABI or driver mismatch shows up |
| `no cuda-compat shadowing` | a `cuda-compat` on `LD_LIBRARY_PATH` would cause CUDA error 803 |
| `cpu ram` | the A10's RAM only. `OFFLOAD=1` (rung 3, the sync search job) needs ~550 GiB per H100 node |
| `EFA / RDMA (AWS)` | warns on an A10, as expected; multi-node H100 jobs need the EFA devices |
| `AutoBridge resolves the model` | required: `MEGATRON_MODE=fsdp` only works through Megatron-Bridge |
| `C compiler on PATH` | Triton compiles a small launcher for Qwen3.5's Gated-DeltaNet kernels at runtime |

## 4. Data, model and the ladder

```bash
make prep     # geo3k to parquet: 64 train / 128 test (N_TRAIN=0 for the full split)
make stage    # Qwen3.5-35B-A3B (~70 GB) to the Volume, 20-40 min; once, not per run

make rung1    # Qwen3.5-2B dense, Megatron-FSDP, 8xH100: the whole code path, cheapest
make rung2    # Qwen3.5-9B dense, Megatron-FSDP, 8xH100
make rung3    # 35B-A3B MoE, classic Megatron + CPU offload, 8xH100: verl's tested config
make rung4    # 35B-A3B MoE, Megatron-FSDP, no offload, 32xH100
```

Run the rungs in order: a failure on rung 1 costs 8 GPU-minutes, on rung 4 GPU-hours. Each
changes several settings at once ([ladder.md](ladder.md)); rung 4 needs 32 GPUs because 16 hold
the steady state but not the co-located weight-sync peak ([sizing.md](sizing.md)). Rungs and
training targets print a GPU-hour bound and submit only with `BUDGET_OK=1`. The rungs stop at
`total_training_steps: 3`; set it to `0` for a real run.

## 5. Watching a run

```bash
make runs                       # recent runs
make logs RUN=<run_id>          # rank 0
make logs RUN=<run_id> NODE=1   # the other node; Ray join failures show up here
make cancel RUN=<run_id>        # multi-node jobs bill per node
```

Metrics go to MLflow under the job's `experiment_name`, in your default location. To group them,
add `mlflow_experiment_directory: /Workspace/Users/<you>/verl-on-air` to a job file.

## 6. Changing code, and your own task

Every job uploads `engine/` plus its use case (or `infra/` directory) as a `code_source`
snapshot, and the image holds no repository code, so a launcher, reward or tool change goes out
with the next submit. Only a change to the installed stack needs `make bump && make release`.

A new task needs a prep script that writes verl's parquet schema and a `compute_score` for
`CUSTOM_REWARD_PATH` ([new-usecase.md](new-usecase.md)). Run `make baseline` on your data first:
it reports the fraction of sample groups whose rewards differ, which is your effective batch
size (a group where every sample scores the same teaches GRPO nothing, and pass@1 hides it).
