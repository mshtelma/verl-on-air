# Setup

End-to-end, from an empty laptop to a 16×H100 GRPO run on `df1`.

## 0. Prerequisites

```bash
# CLIs
uv tool install --force databricks-air --python 3.12
air --version
databricks --version
docker --version

# Auth
databricks auth login --host https://<df1-workspace-url> --profile df1
docker login                       # Docker Hub user: mshtelma
```

Check quota before you start — rung 4 needs **2 free `GPU_8xH100` nodes**:

```bash
air list runs --active -p df1       # someone else's job may be holding them
```

## 1. Create the UC volume

`make volume` creates the **volume** but not the catalog or schema, so confirm
`main.mshtelma` exists in df1 first:

```bash
databricks schemas get main.mshtelma -p df1     # 404 -> create it first:
# databricks schemas create mshtelma main -p df1
```

Then:

```bash
make volume
# -> /Volumes/main/mshtelma/verl
```

Needs ~150 GB: ~70 GB model + data + checkpoints.

All `air/*.yaml` files carry this path **literally** rather than templating it, so
they stay readable and hand-submittable. If you change catalog/schema/volume in
`config.env`, grep `air/` and `scripts/` and update to match — `make help` prints
the resolved path as a cross-check.

## 2. Build, gate, push, register the image

```bash
make build      # linux/amd64 is forced; an arm64 image will not run
make size       # HARD GATE: fails above 19.5 GB (DCS rejects >20 GB)
make push
make register   # 2-6 min, blocks; private repo so it prompts for a Docker Hub PAT
```

Or all four: `make image`.

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

`df1` is **AWS** (`dbc-559ffd80-2bfc.cloud.databricks.com`); `df2` is Azure
(`adb-4599328495546933.13.azuredatabricks.net`). That decides the base image,
because the `-cu13` tags are published for **AWS only**:

| tag | cloud | CUDA | NCCL |
|---|---|---|---|
| `dcs-base-aws-runtime-cu13` | AWS (df1) | **13.0.3** | 2.28.3 +cuda13.0 |
| `dcs-base-azure-runtime` | Azure (df2) | 12.9.1 | 2.27.3 +cuda12.9 |

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
make prep     # geo3k -> parquet (64 train / 8 test; set N_TRAIN=0 for the full split)
make stage    # Qwen3.5-35B-A3B (~70 GB) -> UC volume, ~20-40 min
```

Stage the model **once**. Pulling 70 GB from HF on every training run costs
10-20 min of paid H100 time per run.

## 5. Climb the ladder

Each rung changes exactly one variable from the previous one. Do not skip — a
failure at rung 1 costs 8 GPU-minutes; the same failure discovered at rung 4
costs 16 GPU-hours.

```bash
make rung1    # Qwen3.5-2B  dense  FSDP  8xH100  — full code path, cheapest
make rung2    # Qwen3.5-9B  dense  FSDP  8xH100  — FSDP sharding starts to matter
make rung3    # 35B-A3B MoE CLASSIC + offload 8xH100 — upstream's tested config
make rung4    # 35B-A3B MoE MEGATRON-FSDP no-offload 16xH100 — the headline
```

Rungs 1 and 2 are dense, so they run `EP=1`. Rung 3 vs rung 4 is the
interesting comparison and the actual demo: same model, same data, same reward,
two different sharding strategies, and the reason one needs 16 GPUs is
[docs/sizing.md](sizing.md).

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

> The reference repo this was modelled on also set
> `mlflow_experiment_directory`, which is **not** in the public workload YAML
> reference. Confirm with `air -h config` before adding it.

## 7. Iterating without rebuilding

Rung 4 uses `code_source: snapshot` for `scripts/`, so launcher edits ship with
the job. `make rung4` picks up your local changes with no image rebuild.

Rungs 0-3 run the copy baked at `/app/scripts`. Add the same `code_source`
block to those YAMLs if you want the same fast loop.

## 8. Phase 2: your dataset and reward

The pipeline deliberately keeps both behind seams.

**Dataset.** Write a prep script producing verl's parquet schema (copy
`scripts/prep_geo3k.py`). The `data_source` column selects the scorer, so it
must match whatever you register. For a text-only dataset set `image_key: ''` in
the YAML `parameters:` to drop the multimodal path.

**Reward.** `scripts/reward/custom_reward.py` is a working, tested
drop-in — currently a dict-returning clone of the geo3k rule. Enable it with:

```yaml
env_variables:
  CUSTOM_REWARD_PATH: /app/scripts/reward/custom_reward.py
  CUSTOM_REWARD_NAME: compute_score
```

verl calls it with keyword args
`(data_source=, solution_str=, ground_truth=, extra_info=)` — verified against
`verl/workers/reward_manager/naive.py:131`. Return a **dict** with a `score`
key; every other key becomes its own MLflow metric, which is the only way to
see "learning the answer" separately from "learning the output format".

**Before you commit to a dataset, run `make baseline`.** It reports the
fraction of sample-groups with non-zero reward variance. GRPO normalises reward
within each group, so a group where all `n` samples score identically yields
advantage 0 and contributes **no gradient** — that fraction *is* your effective
batch size, and `pass@1` does not tell you what it is.

One measured subtlety that matters here: geo3k's accuracy term is gated on
`\boxed{}` extraction, so a *correct but unboxed* answer scores **0.00, not
0.90**. Emitting the box is a precondition for any reward at all. Run
`python3 scripts/reward/custom_reward.py` to see the full reward surface.
