# Setup

← [verl-on-air](../README.md) · [running-jobs](running-jobs.md) · [build-linux](build-linux.md) · [troubleshooting](troubleshooting.md)

End-to-end, from an empty laptop to a multi-node GRPO run on `df1`. For the job-by-job
operational guide once this is done, see [running-jobs.md](running-jobs.md).

## 0. Prerequisites

```bash
# CLIs
uv tool install --force databricks-air --python 3.12
air --version
databricks --version
docker --version

# Auth
databricks auth login --host https://<df1-workspace-url> --profile df1
docker login                       # Docker Hub user: michaelshtelma587
```

> **Stale-credential gotcha.** A pre-existing `~/.databrickscfg` entry is not
> enough. Recent CLI versions reject the old file cache:
> ```
> Error: error getting token: cache: stored credentials from older CLI versions
> are no longer used; run `databricks auth login` to sign in again
> ```
> `air` uses the same auth, so this blocks `air register image` and every
> `air run` — re-run `databricks auth login --profile df1` before starting.
> Verify with a cheap read: `databricks schemas get main.mshtelma -p df1`.

Check quota before you start — rung 4 needs **4 free `GPU_8xH100` nodes** (32 GPUs), and
the use-case training jobs need 2 (agentic-search) or 4 (math, judge included). The
workspace quota is per accelerator type (df1: 24 `GPU_8xH100` nodes); a job over it fails
at submit with "Workspace has exceeded its GPU quota".

```bash
make runs                           # recent runs; someone else's job may be holding nodes
```

### Permissions a fresh workspace needs

Check each once, as the identity that will submit jobs (`<profile>` = your `AIR_PROFILE`):

| what | needed by | check |
|---|---|---|
| AI Runtime serverless GPU enabled, `GPU_8xH100` quota | every GPU job | `air list runs -p <profile>` works |
| `USE CATALOG` + `USE SCHEMA` + `CREATE VOLUME` on `UC_CATALOG.UC_SCHEMA` | `make volume` | `databricks schemas get <catalog>.<schema> -p <profile>` |
| `READ VOLUME` + `WRITE VOLUME` on the Volume | data, models, checkpoints, evals, rendezvous | `databricks fs ls dbfs:/Volumes/<catalog>/<schema>/<volume> -p <profile>` |
| `CREATE TABLE` on the schema; a SQL warehouse you can use (`CAN USE`) | `make search-index` (loads the corpus table) | `databricks warehouses list -p <profile>` |
| a Vector Search endpoint you can use, or the right to create one | `make search-index`; search train/eval query it | `databricks vector-search-endpoints list-endpoints -p <profile>` |
| the embedding endpoint the index uses (`databricks-gte-large-en`) | the index build (Databricks embeds the corpus) | `databricks serving-endpoints get databricks-gte-large-en -p <profile>` |
| an MLflow experiment location you can write | every training job's metrics | the job's `mlflow_url` opens |
| a secret scope you can write | `air register` (Docker Hub credential); optional `HF_TOKEN` | `databricks secrets list-scopes -p <profile>` |

Nothing here is created implicitly except the Volume (`make volume`) and, with
`--create-endpoint`, a Vector Search endpoint -- which is billable and persistent.

## 1. Create the UC volume

The `main.mshtelma` schema already exists in df1, so this only creates the
volume inside it (`make volume` does not create catalogs or schemas):

```bash
make volume
# -> /Volumes/main/mshtelma/verl
```

What it holds, roughly:

| what | size | notes |
|---|---|---|
| base model `models/Qwen3.5-35B-A3B` | ~70 GB | staged once (`make stage`) |
| LLM judge `models/GLM-5.3` (math only) | ~744 GB | staged once (`make math-judge`), cached per node at `/local_disk0` |
| data (geo3k, MuSiQue, the passage corpus) | < 1 GB | |
| one training checkpoint of the 35B | ~0.5 TB (estimate) | HF export (70 GB, measured) + bf16 Megatron state (~70 GB) + fp32 Adam state (~12 B/param, ~420 GB) |
| a training run | checkpoint size x `MAX_CKPT_TO_KEEP` (default 3) | `make prune-ckpts CKPT=<run>` trims it afterwards |
| eval artifacts + traces | MBs per eval | |

All `air/*.yaml` files carry this path **literally** rather than templating it, so
they stay readable and hand-submittable. If you change catalog/schema/volume (or the
Vector Search endpoint/index) in `config.env`, `make retarget` rewrites every job file,
and `make lint` fails while any disagrees.

## 2. Build, gate, push, register the image

```bash
make validate   # schema-check every **/air/*.yaml against the real air CLI (seconds, free)
make build      # linux/amd64 is forced; an arm64 image will not run
make size       # HARD GATE: fails above 19.5 GB (DCS rejects >20 GB)
make push
make register   # 2-6 min, blocks (uses the stored secret; see below)
```

Or all four: `make image`.

### Registry credentials for `air register image`

A custom image must be registered before any workload can use it, and a private
Docker Hub repo needs credentials for that. `air` keeps them in a **Databricks
secret**, not in the YAML.

Create them once, interactively:

```bash
air register image michaelshtelma587/verl-megatron-air:v1 -p df1
# Docker registry username: michaelshtelma587
# Docker registry password/PAT: ****
# Databricks secret scope name [docker-credentials-...]: msh
# Databricks secret key name  [dockerio-...]:            dockerhubsecret
# -> To reuse these credentials in future commands, pass:
#      --scope msh --key dockerhubsecret
```

Then record them in `config.env` so every later registration is non-interactive:

```
SECRET_SCOPE=msh
SECRET_KEY=dockerhubsecret
```

`make register` uses `--scope/--key` whenever both are set, and only falls back to
the interactive flow otherwise. That fallback **reads the controlling TTY and will
hang** in CI or any non-interactive shell, which is why the values belong in
`config.env`.

The secret's payload format is internal to `air`, so do not hand-craft it with
`databricks secrets put-secret` — re-run the interactive flow to rotate it. To
check it exists (names only, never values):

```bash
databricks secrets list-secrets msh -p df1
```

> Re-register only when you push a **new tag** or rotate credentials — not on
> every run.

`make validate` is worth running first because it costs nothing. It swaps the
custom image for a stock environment before calling `air run --dry-run`, so it
exercises the schema, compute topology, `code_source` and `parameters` *before*
the image exists — otherwise every file just fails with "Image not registered"
and masks any real error. `make check` runs it together with the local linters.

Three things the build does for you:

- **Verifies itself.** The final layer imports the whole stack and asserts
  `transformers.Qwen3_5MoeForConditionalGeneration` exists. A wrong
  `transformers` pin fails the *build*, not a 16-GPU job 40 minutes in.
- **Asserts `Python.h` is reachable** from `/opt/venv`'s interpreter. This base
  sets `UV_PYTHON_INSTALL_DIR=/opt/uv/python`, so apt's `python3-dev` may install
  headers for the wrong interpreter and megatron-core would then fail
  cryptically. The build fails early, printing the actual paths.
- **Compiles nothing CUDA.** TransformerEngine, apex and flash-attn come as
  prebuilt cu130/torch-2.11/cp312 wheels from verl's wheelhouse, pinned by
  direct URL. That is what lets us sit on the 4.1 GB *runtime* base instead of
  the 10.3 GB *devel* one and stay under the size cap.

If `make size` fails, `make layers` shows the biggest layers.

### Why df1, and the CUDA story

The cloud your workspace runs on decides the base image, because the `-cu13` tags
are published for **AWS only**:

| tag | cloud | CUDA | NCCL |
|---|---|---|---|
| `dcs-base-aws-runtime-cu13` | AWS | **13.0.3** | 2.28.3 +cuda13.0 |
| `dcs-base-azure-runtime` | Azure | 12.9.1 | 2.27.3 +cuda12.9 |

Our stack is torch **cu130**, so on df1 the toolchain matches natively and there
is nothing to reason about. (On df2 it also works, but only via an argument about
bundled CUDA libs, soname major-versioning, the host driver floor and
`cuda-compat` absence — see git history for that chain.)

Downgrading to a CUDA 12 torch is not the cheap escape it appears to be: verl's
wheelhouse ships **cu130-only** TransformerEngine/apex/flash-attn, so it would
mean source-building them and blowing the 20 GB cap.

The smoke test still asserts the driver floor (CUDA 13.0 needs ≥ 580.65.06),
absence of `cuda-compat` on `LD_LIBRARY_PATH` (which would cause CUDA Error 803),
and a real on-device bf16 matmul — node pools get rebuilt, and those are host
facts the image cannot control.

## 3. Pre-flight on 1×A10 (~2 min, do not skip)

```bash
make smoke
```

Read these out of the output:

| line | why it matters |
|---|---|
| `driver supports CUDA 13` | the load-bearing assumption of cu130 wheels on the CUDA 12.9 Azure base |
| `CUDA 13 wheels run on this base` | an actual on-device matmul — an ABI/driver mismatch surfaces here, not at import |
| `no cuda-compat shadowing` | a `cuda-compat` on `LD_LIBRARY_PATH` would cause CUDA Error 803 |
| `cpu ram` | decides whether `OFFLOAD=1` (rung 3) is viable at all — needs ~550 GiB |
| `infiniband (Azure RDMA)` | warns on 1×A10 (no RDMA hardware — expected); must show `mlx5_*` on H100 for rung 4 |
| `AutoBridge resolves the model` | if this fails, `MEGATRON_MODE=fsdp` cannot work; Megatron-FSDP is only reachable via Megatron-Bridge |
| `C compiler on PATH` | Qwen3.5's Gated-DeltaNet layers are Triton kernels and Triton JITs a host launcher stub at runtime |

## 4. Data and model

```bash
make prep     # geo3k -> parquet (64 train / 128 test; set N_TRAIN=0 for the full split)
make stage    # Qwen3.5-35B-A3B (~70 GB) -> UC volume, ~20-40 min
```

Stage the model **once**. Pulling 70 GB from HF on every training run costs
10-20 min of paid H100 time per run.

## 5. Climb the ladder

Each rung adds one new *risk* (the table in [ladder.md](ladder.md) lists what changes
between rungs -- sometimes more than one setting). Do not skip — a failure at rung 1
costs 8 GPU-minutes; the same failure discovered at rung 4 costs 16 GPU-hours. Every
rung (like every training target) prints its GPU-hour upper bound and submits only with
`BUDGET_OK=1`.

```bash
make rung1    # Qwen3.5-2B  dense  FSDP  8xH100  — full code path, cheapest
make rung2    # Qwen3.5-9B  dense  FSDP  8xH100  — FSDP sharding starts to matter
make rung3    # 35B-A3B MoE CLASSIC + offload 8xH100 — upstream's tested config
make rung4    # 35B-A3B MoE MEGATRON-FSDP no-offload 32xH100 — the headline
```

Rungs 1 and 2 are dense, so they run `EP=1`. Rung 3 vs rung 4 is the interesting
comparison: same model, same data, same reward, two different sharding strategies. Note
the rung4 file is named `…16gpu.yaml` but requests **32** — 16 fits the persistent state,
the co-located weight-sync transient does not. Why: [sizing.md](sizing.md).

All four are capped at `total_training_steps: 3`. Set it to `0` in the YAML to
remove the cap for a real run.

## 6. Watch it

`air run --watch` streams the driver. Otherwise:

```bash
make runs                       # active runs
make logs RUN=<run_id>          # driver (rank 0)
make logs RUN=<run_id> NODE=1   # the other node — where Ray-join failures show up
make cancel RUN=<run_id>        # multi-node jobs bill per node; cancel promptly
```

Metrics land in MLflow under `experiment_name`. `trainer.logger` is
`['console','mlflow']`; AI Runtime injects the MLflow context, so no tracking
URI is needed.

Experiments land in your own workspace's default MLflow location. To group them
under one folder instead, add `mlflow_experiment_directory: /Workspace/Users/<you>/verl-on-air`
to a job file (optional; it must start with `/Workspace`).

## 7. Iterating without rebuilding

**Every** job now uses `code_source: snapshot` (`engine/` plus the one use case or infra
subdir), so a launcher, reward or tool edit ships with the next submit and needs no image
rebuild. Only changing the installed stack (a pip pin, a system package) needs
`make bump && make release`.

The image carries **no** repository code (from v9; earlier tags baked a copy at `/app`), so
what runs is always the snapshot of the tree you submitted from -- and a code change can never
be masked by, or blamed on, a stale copy in the image.

## 8. Your own dataset and reward

Both sit behind seams, so neither needs an engine change: a prep script that emits verl's
parquet schema, and a `compute_score` function pointed at by `CUSTOM_REWARD_PATH`. The
contracts, a copyable starting point, and a GPU-free checklist are in
**[new-usecase.md](new-usecase.md)**.

Before committing a dataset, run `make baseline`: it reports the fraction of sample-groups
with **non-zero reward variance**. GRPO's advantage is relative to the group, so a group
where all samples score identically contributes no task-reward gradient — that fraction is
your effective batch size, and `pass@1` will not tell you what it is.
