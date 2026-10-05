# verl 0.10 trial

This is an isolated experiment on branch `trial/verl-0.10`, in the worktree
`/home/michael.shtelma/verl-on-air-010`. The original checkout and its v9 image stay available.
The upstream package calls itself **0.10.0.dev**; this is a pinned development snapshot,
not a released 0.10.0 or a claim of production stability.

Image: `michaelshtelma587/verl-megatron-air:v10-verl010-dev`.
This is the initial vLLM 0.29 image at commit `0a153cc`. The current branch's
[Flash judge phase](glm-flash-judge-trial.md) now uses `v12-verl010-glmflash-fix1`,
vLLM 0.30 and Transformers 5.16.1; the original image remains available as the control.
Published and verified against the registry on 2026-10-05. The digest recorded in
`docker/IMAGE.lock` is:

```text
sha256:8725d4ea7c69090fcff8347c9337f5afbb37e1fca2f75ac594e7cc17fdbce890
```

## Initial v10 stack

| Component | Trial | Previous v9 image |
|---|---|---|
| verl | `0.10.0.dev`, commit `8718ca30a3f002f93b7c4fd99b9b2506718681bc` | 0.9.0 |
| PyTorch | 2.13.0, CUDA 13.0 | 2.11.0, CUDA 13.0 |
| vLLM | 0.29.0 | 0.24.0 |
| Transformers | 5.12.1 | 5.5.3 |
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
- Retain the CUDA 13.2.86 runtime JIT compiler/header alignment and separate judge Ray 2.48.0.
  vLLM 0.29 with that judge Ray still needs qualification.
- Fetch verl by its immutable commit for CPU composition checks. `0.10.0.dev` is a package
  version, not a Git release tag.

Dependency inputs are in `docker/requirements.in`. Regenerate the index-package lock with:

```bash
.venv/bin/python scripts/lock_requirements.py
```

This exports the pinned upstream frozen `uv.lock`, applies the documented AIR differences,
then resolves against PyPI or the same detected package mirror used by `make build`.
The Dockerfile verifies native hashes, torch ABI, runtime CUDA compilation, imports, and
installed-package agreement with the locks. Those gates remain required for the trial.

## Qualification

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
check failures. H100 training/checkpoint qualification is continuing with the newer Flash
image. The initial smoke had a 10-minute timeout and no retries.

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

After a successful initial run, resume the same RUN_ID with `RESUME=auto` and
`total_rollout_steps=16`. The learning-rate horizon remains 16 for both submissions;
the resumed run must reach a verified `global_step_4`. Check `run_result.json`, the completed
checkpoint manifests, HF weights, and logs showing the resumed step rather than trusting
verl's process exit status alone.

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
save and resume. It does not establish 35B MoE capacity, tool-agent compatibility, distributed
judge behavior, or multi-node EFA transport. Torch now bundles NCCL 2.29.7 while the AIR base's
OFI plugin was built against NCCL 2.28.3; require `NET/OFI` in a real multi-node run before
promoting this image for that workload. The results in `RESULTS.md` remain measurements of
the previous stack.
