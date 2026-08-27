# verl-on-air

**GRPO reinforcement learning on `Qwen3.5-35B-A3B` (MoE) with verl's
Megatron/mcore backend, on Databricks AI Runtime serverless GPU.**

The headline configuration runs **Megatron-FSDP (ZeRO-3) across 16×H100 with no
CPU offload**. That combination is the point of the repo: it is the smallest
offload-free topology for this model, and classic Megatron cannot reach it at
any node count below 32. The arithmetic is in [docs/sizing.md](docs/sizing.md).

Status: **skeleton — builds and submits, not yet validated on hardware.** Every
claim traceable to upstream source or docs is cited; every unverified assumption
is flagged in [docs/troubleshooting.md](docs/troubleshooting.md).

---

## Quick start

```bash
databricks auth login --host https://<df1-url> --profile df1
docker login                     # Docker Hub: mshtelma

make check                       # lint + validate all air YAML vs the real CLI (free)
make image                       # build + size gate + push + register
make setup                       # volume + 1xA10 smoke + geo3k + stage 70 GB model
make rung1                       # cheapest end-to-end check   (8xH100, ~15 min)
make rung4                       # the headline run            (16xH100)
```

`make help` lists everything.

### Validated against the live `df1` workspace

`make validate` runs `air run --dry-run` on all eight workload files, swapping the
custom image for a stock environment so the schema is checked *before* the image
exists (otherwise every file fails with "Image not registered" and hides real
errors). Currently all 8 pass, which confirms against the real CLI:

- `num_accelerators: 16` + `GPU_8xH100` → 2 nodes (per `air -h config.compute`)
- rung 4's `code_source.snapshot.root_path: ..` resolves and packages correctly
- `mlflow_experiment_directory` is a genuine field (it is absent from the public
  YAML reference, so it was previously flagged as unverified)
- `parameters`, `env_variables`, `max_retries`, `timeout_minutes` all accepted

## The validation ladder

Each rung changes **one** variable from the previous. A failure at rung 1 costs
8 GPU-minutes; the identical failure found at rung 4 costs 16 GPU-hours.

| | model | backend | GPUs | offload | purpose |
|---|---|---|---|---|---|
| `make rung1` | Qwen3.5-2B (dense) | Megatron-FSDP | 8 | no | full code path, cheapest possible |
| `make rung2` | Qwen3.5-9B (dense) | Megatron-FSDP | 8 | no | FSDP sharding starts to matter |
| `make rung3` | **35B-A3B** (MoE) | classic Megatron | 8 | **yes** | reproduces verl's own tested config |
| `make rung4` | **35B-A3B** (MoE) | **Megatron-FSDP** | **16** | **no** | **headline** |

Rungs 1–2 are dense, so `EP=1`. Rungs 3 and 4 are the same model, data and
reward with two different sharding strategies — that contrast *is* the demo.

## Why 16 GPUs, and why FSDP

92.5 % of this model's 34.8 B parameters are routed-expert weights (256 experts
× 40 layers). Per-GPU persistent HBM, no offload:

| GPUs | mode | params | grads | Adam | ref | vLLM | +overhead | verdict |
|---|---|---|---|---|---|---|---|---|
| 8 | classic | 10.6 | 10.6 | 52.2 | 10.6 | 8.7 | 110.6 | **OOM** |
| 8 | fsdp | 8.7 | 8.7 | 52.2 | 8.7 | 8.7 | 105.0 | **OOM** |
| 16 | classic | 10.6 | 10.6 | 26.1 | 10.6 | 8.7 | 84.5 | **OOM** |
| **16** | **fsdp** | **4.4** | **4.4** | **26.1** | **4.4** | **8.7** | **65.9** | **OK** |
| 32 | fsdp | 2.2 | 2.2 | 13.1 | 2.2 | 8.7 | 46.3 | roomy |

Classic Megatron's distributed optimizer is **ZeRO-1**: it shards optimizer
state but *replicates* params and grads, which is why more nodes never fix it.
Megatron-FSDP (`optim_grads_params`) is **ZeRO-3** and shards all three.

`EP=8` at 16 GPUs gives `expert-DP = 2`, which is what FSDP needs to shard the
experts at all. At 8 GPUs `expert-DP = 1` and FSDP becomes pure overhead —
measured upstream in NVIDIA/Megatron-LM issue #2772, where EP8+FSDP was both
slower *and* hungrier than EP8 alone.

Reproduce every number: `python3 docs/sizing.py`.

## Algorithm and reward

**GRPO**, no critic — the advantage is a group-relative baseline over
`rollout_n=5` samples per prompt, so no value network exists. KL to the
reference policy is a loss term (`kl_loss_coef=0.01`, `low_var_kl`), not folded
into the reward.

**Reward is rule-based and verifiable** — no reward model, nothing to train.
geo3k scoring is `0.9 × answer_correct + 0.1 × format_ok`. Measured surface:

| response | score | acc | fmt |
|---|---|---|---|
| `<think>…</think> … \boxed{42}` correct | 1.00 | 1 | 1 |
| `<think>…</think> … \boxed{7}` wrong | 0.10 | 0 | 1 |
| `the answer is \boxed{42}` correct | 0.90 | 1 | 0 |
| `42` — correct but **unboxed** | **0.00** | 0 | 0 |

Note the last row: the accuracy term is **gated on `\boxed{}` extraction**, so a
correct-but-unboxed answer earns nothing. Emitting the box is a precondition for
any reward, not a 10 % style bonus. Verify with
`python3 scripts/reward/custom_reward.py`.

> **Before a real training run, `make baseline`.** GRPO's gradient comes
> entirely from reward variance *within* a group; if all `n` samples score
> alike, advantage is 0 and that prompt teaches nothing. The baseline job
> reports the fraction of groups with non-zero variance — that fraction is your
> effective batch size. `pass@1` does not tell you what it is, and
> `Qwen3.5-35B-A3B` is a strong post-trained model on an easy dataset.

## Version set

Taken verbatim from verl v0.9.0's `docker/Dockerfile.stable.vllm`, a
mutually-tested combination. Do not bump one alone.

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | df1 = AWS. CUDA 13.0.3, matching our wheels. ~4.7 GB; *runtime*, not devel — that is the size headroom |
| torch | 2.11.0 / cu130 | ABI everything else is built against; matches the base's CUDA 13 |
| vllm | 0.24.0 | first with Qwen3.5 rollout support |
| transformers | 5.5.3 | `Qwen3_5MoeForConditionalGeneration`; asserted at build time |
| verl | v0.9.0 | `--no-deps` (its metadata pins numpy<2, vllm≤0.12) |
| megatron-core | `core_v0.18.0` | only source build; pybind11 ext, no CUDA |
| megatron-bridge | 0.5.2 (`r0.5.0`) | **required** for Megatron-FSDP |
| TransformerEngine | 2.16.1 | prebuilt wheel |
| apex / flash-attn | 0.1 / 2.8.3 | prebuilt wheels |
| opencv-python-headless | **4.12.0.88** | 5.x bundles FIPS libcrypto that aborts on `import cv2` |

All native wheels come prebuilt from
[verl's wheelhouse](https://verl-project.github.io/verl-wheelhouse/simple/),
pinned by **direct URL** — nothing CUDA compiles at build time, and index
resolution can't grab the unrelated PyPI package named `apex`.

### Why df1 (AWS) and not df2 (Azure)

Both workspaces exist; this repo targets **df1** because the `-cu13` base images
are **published for AWS only**:

| tag | cloud | CUDA | NCCL |
|---|---|---|---|
| `dcs-base-aws-runtime-cu13` | AWS (df1) | **13.0.3** | 2.28.3 +cuda13.0 |
| `dcs-base-azure-runtime` | Azure (df2) | 12.9.1 | 2.27.3 +cuda12.9 |

Our stack is torch **cu130**, so on df1 the toolchain simply matches. On df2 it
would still work, but only via a four-step argument (cu130 wheels bundle their own
CUDA 13 runtime; sonames are major-versioned so the 12.9 libs can't shadow them;
only `libcuda.so.1` comes from the host and Azure's 580.105.08 driver clears CUDA
13.0's 580.65.06 floor; and no `cuda-compat` sits on `LD_LIBRARY_PATH` to trigger
Error 803). df1 removes the need for that argument entirely.

Downgrading to a CUDA 12 torch is *not* the cheap alternative it looks like:
verl's wheelhouse ships **cu130-only** builds of TransformerEngine/apex/flash-attn,
so it would mean source-building them — slow, and it would blow the 20 GB cap.

To move to df2 anyway: set `AIR_PROFILE=df2` in `config.env` **and** change the
Dockerfile `FROM` to `dcs-base-azure-runtime`. Networking differs too — Azure is
InfiniBand (NCCL speaks IB verbs natively) while AWS is EFA (NCCL reaches it
*through* `aws-ofi-nccl`, so a plugin failure costs RDMA outright rather than just
the SHARP optimisation). Rung 4 sets `NCCL_DEBUG=INFO`; confirm `NET/OFI ...
Provider is efa` rather than `NET/Socket`.

The smoke test asserts the driver floor, absence of `cuda-compat` shadowing, and
an actual on-device bf16 matmul.

## Layout

```
config.env                     Makefile config (profile, image, UC paths)
docker/Dockerfile              the image; self-verifying final layer
air/00_smoke_test.yaml         1xA10 pre-flight: imports, arch, CPU RAM, compiler
air/01_prep_geo3k.yaml         geo3k -> UC volume parquet
air/02_stage_model.yaml        HF -> UC volume (~70 GB, once)
air/03_baseline_eval.yaml      measure GRPO reward variance before training
air/1*.yaml air/2*.yaml        the four rungs
scripts/run_grpo_megatron.sh   THE launcher: fsdp|classic, topology, Ray
scripts/lib/hparams.sh         air `parameters:` (YAML, not JSON) -> shell
scripts/lib/ray_cluster.sh     multi-node Ray head/worker + clean teardown
scripts/reward/custom_reward.py  PHASE 2 HOOK — tested drop-in reward
docs/sizing.md  docs/sizing.py   the memory analysis, and code to reproduce it
docs/setup.md                  step by step
docs/troubleshooting.md        symptom -> cause -> fix
```

## Phase 2: your dataset and reward

Both are behind seams already:

- **Reward** — set `CUSTOM_REWARD_PATH=/app/scripts/reward/custom_reward.py`.
  verl calls it with `(data_source=, solution_str=, ground_truth=, extra_info=)`
  (verified against `verl/workers/reward_manager/naive.py:131`). Return a dict
  with a `score` key; other keys become individual MLflow metrics, which is the
  only way to separate "learned the answer" from "learned the format".
- **Dataset** — copy `scripts/prep_geo3k.py`. The `data_source` column selects
  the scorer. Set `image_key: ''` for text-only data to skip the vision path.

See [docs/setup.md §8](docs/setup.md).

## Credits

Databricks AI Runtime packaging patterns — image size limits, the FIPS/opencv
trap, Ray multi-node teardown, YAML hyperparameters — are adapted from
[hiouchiy/databricks-air-verl-qwen35](https://github.com/hiouchiy/databricks-air-verl-qwen35).
Training configuration follows verl's own
`examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` and
`run_qwen2-7b_math_megatron_fsdp.sh`.
