# verl 0.10 trial

This is an isolated experiment on branch `trial/verl-0.10`, in the worktree
`/home/michael.shtelma/verl-on-air-010`. The original checkout and its v9 image stay available.
The upstream package calls itself **0.10.0.dev**; this is a pinned development snapshot,
not a released 0.10.0 or a claim of production stability. The bounded df1 image,
2B training/checkpoint/resume, and TP8 Flash judge qualifications all passed on
2026-10-05. The full 24-GPU math workload remains unlaunched.

Current image: `michaelshtelma587/verl-megatron-air:v12-verl010-glmflash-fix1`.
It uses vLLM 0.30.0, Transformers 5.16.1, and a guarded verl weight-serializer
repair. It is published, registry-verified, and registered AVAILABLE on **df1**.
Its digest is recorded in `docker/IMAGE.lock`:

```text
sha256:7bb2fbf97763e4aae1c79ed7f6c5515a25519c868bbbce002476c722d19f388a
```

The [Flash judge results](glm-flash-judge-trial.md) record successful run IDs,
checkpoint evidence, and serving measurements. The initial image
`michaelshtelma587/verl-megatron-air:v10-verl010-dev`, built at commit `0a153cc`
with vLLM 0.29 and Transformers 5.12.1, remains available as the control:

```text
sha256:8725d4ea7c69090fcff8347c9337f5afbb37e1fca2f75ac594e7cc17fdbce890
```

## Current v12 stack

| Component | Current trial | Previous v9 image |
|---|---|---|
| verl | `0.10.0.dev`, commit `8718ca30a3f002f93b7c4fd99b9b2506718681bc` | 0.9.0 |
| PyTorch | 2.13.0, CUDA 13.0 | 2.11.0, CUDA 13.0 |
| vLLM | 0.30.0 | 0.24.0 |
| Transformers | 5.16.1 | 5.5.3 |
| Megatron-Core | 0.19.2, commit `4b4acac9a1d28ea6829c8d4f566d75698a21249d` | 0.18.0 |
| NVIDIA Megatron-Bridge | 0.6.2 | 0.5.2 |
| TransformerEngine | 2.16.1, torch 2.13 wheel | 2.16.1, torch 2.11 wheel |
| FlashAttention | 2.8.3, torch 2.13 wheel | 2.8.3, torch 2.11 wheel |
| Ray for training | 2.55.1 | 2.58.0 |
| TransferQueue | 0.1.10 | 0.1.8 |

The four wheelhouse inputs and the OpenCV wheel are SHA256-pinned in `docker/artifacts.lock`. Apex's package
version remains 0.1, but its binary is replaced with the torch 2.13 build. The AWS AIR base
image digest is retained. All custom-image jobs on this branch use the separate trial tag.

## Changes needed for AIR

- Both classic Megatron and Megatron-FSDP now use NVIDIA Megatron-Bridge. Remove legacy
  `mbridge`, its patches/audit, and `vanilla_mbridge` overrides. `use_mbridge=True` is still a
  valid upstream setting. Remove the deleted `grad_offload` setting and add Megatron Energon.
- Keep the existing fully-async trainer for this comparison. It remains in the pinned source
  but is deprecated upstream; migrating to the V1 trainer is a separate change.
- Include FlashInfer's `cu13` extra so the CUDA 13 CUTLASS libraries are present. Import
  `verl.models.mcore` before NVIDIA Bridge in both image and GPU checks, matching the actual
  training initialization. Upstream handles the bundled optional FA4/CUTLASS API mismatch
  there; NVIDIA Bridge and the FlashAttention-2 binary remain required imports.
- Use actual PyPI torch dependencies: `nvidia-cusparselt-cu13==0.8.1`, rather than upstream's
  0.8.0 override. Keep numpy within mistral-common's supported range (2.3.5), and pandas
  below 3 (2.3.3) for MLflow 3.15.0.
- Use `cuda-tile==1.6.0`, whose released PyPI wheel satisfies FlashInfer's `>=1.4.0` requirement.
  Upstream's 1.6.0rc5 is a placeholder sdist that downloads from `pypi.nvidia.com`; that
  endpoint is unavailable on this build network.
- Keep a FIPS-compatible OpenCV binary, now **4.13.0.92** to satisfy vLLM's new minimum.
  SHA256-pin the `manylinux2014` wheel in `artifacts.lock` and install it last. The resolver's
  preferred `manylinux_2_28` wheel bundles a different libcrypto with the FIPS self-test abort
  code; a version pin alone does not choose the correct binary.
- Retain the CUDA 13.2.86 runtime JIT compiler/header alignment. The separate judge
  environment includes Ray 2.48.0; current TP8 Flash serving uses multiprocessing
  on one node and does not exercise distributed judge Ray.
- Fetch verl by its immutable commit for CPU composition checks. `0.10.0.dev` is a package
  version, not a Git release tag.
- Repair the pinned `split_weight_chunks` implementation to serialize strided NVIDIA
  Bridge weights with `contiguous()` and avoid data copies for metadata-only relays.
  The image build checks the source identity and runs four numerical regressions;
  H100 training and resume then verify the repaired path with actual weight syncs.

Dependency inputs are in `docker/requirements.in`. Regenerate the index-package lock with:

```bash
.venv/bin/python scripts/lock_requirements.py
```

This exports the pinned upstream frozen `uv.lock`, applies the documented AIR differences,
then resolves against PyPI or the same detected package mirror used by `make build`.
The Dockerfile verifies native hashes, torch ABI, runtime CUDA compilation, imports, and
installed-package agreement with the locks. Those gates remain required for the trial.

## Initial v10 qualification

**545 CPU tests passed, 3 expected failures** from existing prime_math limitations before
adding the bounded job. After that addition, all **76 affected tests passed**, including
composition of all eight training jobs and their checkpoint/topology invariants.
`make lint` passed for all 27 jobs, shell scripts, Python files, Dockerfile, and config docs.
All 24 probe tests passed again after requiring the renamed vLLM native extensions in
the GPU smoke.

The image built successfully and passed its native-import, torch ABI, CUDA/CCCL compilation,
Qwen3.5 architecture, and dependency-lock gates. Its local size is **16,115,014,799 bytes
(16.12 GB)**, below the 19.5 GB gate. The actual fully-async training entry point and the
repository's tool-agent loop also import successfully from the image.

On this build host, which has no NVIDIA driver, vLLM's generic platform emits a missing
`vllm._C` warning. The wheel contains the renamed `_C_stable_libtorch` and
`_moe_C_stable_libtorch` binaries, and the CUDA platform imports those names. Loading them
locally stops at the absent `libcuda.so.1`; the GPU smoke now requires both imports explicitly.
This local check does not establish that generation works.

The user selected `df1`. Registration is **AVAILABLE**, and the initial 1×A10 image smoke
passed on 2026-10-05, run **602487753336548**, with a `PASS` probe verdict and no required
check failures. The initial smoke had a 10-minute timeout and no retries.

## Current v12 qualification on df1

All bounded GPU checks passed with the repaired image. The A10 smoke used source
commit `5d107d4`; the H100 runs used
`22878fe8af6e85c67453b189ff3447a754391c54`.

| Qualification | AIR run | Result |
|---|---|---|
| A10 image smoke | [603122941791377](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/603122941791377) | PASS, no required failures |
| Qwen3.5-2B training | [480787754760751](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/480787754760751) | SUCCESS, certified `global_step_2`, raw exit 0 |
| Exact checkpoint resume | [276390230259878](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/276390230259878) | SUCCESS, restored trainer and dataloader from step 2; certified `global_step_4`, raw exit 0 |
| GLM-5.3-Flash TP8 judge | [97695668337323](https://dbc-559ffd80-2bfc.cloud.databricks.com/jobs/runs/97695668337323) | PASS, 8 calibration cases and 32 test requests at concurrency 16 |

V12 is **16,138,980,912 bytes (16.14 GB)** and passes native hashes, torch ABI,
CUDA/CCCL compilation, GLM model-registry/import gates, all 304 exact index pins,
and four numerical serializer regressions. Local lint, all eight training
compositions, all 28 AIR validations on df1, and all 43 affected
dispatch/lifecycle/image-lock tests passed. Initial H100 submissions were rejected
by workspace quota; capacity subsequently opened and all checks above completed.

The Flash judge served on eight H100s, so this branch's configured math allocation
is **32 → 24 GPUs**. The measured concurrent throughput was **1.53 requests/second**,
p95 latency **14.35 seconds**, and cold startup **18.5 minutes**. See the
[Flash trial results](glm-flash-judge-trial.md) for complete evidence and limitations.

The qualification job is `infra/diagnostics/air/verl010_train.yaml`:

| Setting | Value |
|---|---|
| Hardware | One 8×H100 node: 4 trainer GPUs and 4 rollout GPUs |
| Model/data | Qwen3.5-2B / existing geo3k train and test parquet |
| Generation | 8 prompt groups × 2 responses, response cap 256 |
| Training | DP=4, mini-batch 4; 2 updates and 2 weight syncs |
| Checkpoints | Save every update; require a verified `global_step_2` |
| Isolation | `/Volumes/main/mshtelma/verl/ckpt/verl010-smoke/<RUN_ID>` |
| Cost bound | 15 minutes, no retries: at most 2 GPU-hours per attempt |

The successful training and resume used RUN_ID
`verl010-glmflash-fix1-retry-20261005T231405Z-22878fe`. The second submission used
`RESUME=auto` and `total_rollout_steps=16`; the actor's learning-rate horizon stayed
at 16 in both runs. Logs explicitly show trainer and rollout dataloader restoration
from step 2. Certificates verify complete checkpoint manifests and HF exports at
steps 2 and 4, each **4,426,558,832 bytes**, with identities `766faa2fa02925bb` and
`61e5cda90ad71f68`. Both raw exits were 0 and both certificates have no problems.
Training took 702 seconds and resume took 737 seconds, within their 15-minute bounds.

## Commands

```bash
cd /home/michael.shtelma/verl-on-air-010
make lint
make test
make compose-check
make certs vendor build
make size push

# Use the profile explicitly chosen for this trial.
TRIAL_PROFILE=df1
make register AIR_PROFILE="$TRIAL_PROFILE"
make smoke AIR_PROFILE="$TRIAL_PROFILE"
TRIAL_RUN_ID="verl010-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
make trial-train AIR_PROFILE="$TRIAL_PROFILE" RUN_ID="$TRIAL_RUN_ID" BUDGET_OK=1

# Only after the first run completes and its checkpoint verifies:
air run --profile "$TRIAL_PROFILE" --watch \
  --file infra/diagnostics/air/verl010_train.yaml \
  --override env_variables.RUN_ID="$TRIAL_RUN_ID" env_variables.RESUME=auto \
    env_variables.GIT_SHA="$(git rev-parse HEAD)" \
    env_variables.VOA_IMAGE=michaelshtelma587/verl-megatron-air:v12-verl010-glmflash-fix1 \
    parameters.total_rollout_steps=16
```

A passing single-node 2B run verifies basic generation, training, weight sync, checkpoint
save and resume. The separate Flash probe verifies serving and strict grading on a TP8 node.
These checks do not establish 35B MoE capacity, full tool-agent execution, equal math judge
quality, 512-call reward capacity, lower total GPU-hours, or multi-node EFA transport.
Torch now bundles NCCL 2.29.7 while the AIR base's
OFI plugin was built against NCCL 2.28.3; require `NET/OFI` in a real multi-node run before
promoting this image for that workload. The results in `RESULTS.md` remain measurements of
the previous stack.
