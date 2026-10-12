# PyTorch 2.13 and verl 0.10 compatibility

This repository uses PyTorch 2.13 with a pinned verl **0.10.0.dev0** build.
It is a development snapshot, so treat the results below as qualification of
this particular stack. The Dockerfile and lock files record the exact source
and binary versions needed to rebuild it.

## Tested stack

| Component | Current stack | Earlier stack |
|---|---|---|
| verl | 0.10.0.dev0 | 0.9.0 |
| PyTorch / CUDA | 2.13.0 / 13.0 | 2.11.0 / 13.0 |
| vLLM | 0.30.0 | 0.24.0 |
| Transformers | 5.16.1 | 5.5.3 |
| Megatron-Core | 0.19.2 | 0.18.0 |
| NVIDIA Megatron-Bridge | 0.6.2 | 0.5.2 |
| TransformerEngine | 2.16.1, torch 2.13 wheel | 2.16.1, torch 2.11 wheel |
| FlashAttention | 2.8.3, torch 2.13 wheel | 2.8.3, torch 2.11 wheel |
| Ray for training | 2.55.1 | 2.58.0 |
| TransferQueue | 0.1.10 | 0.1.8 |

The OfficeQA image adds BM25 retrieval to the stack qualified by the initial
training and Flash judge probes. Its digest is in
[IMAGE.lock](../docker/IMAGE.lock), native-wheel hashes are in
[artifacts.lock](../docker/artifacts.lock), and Python package versions are in
[requirements.lock](../docker/requirements.lock).

## Compatibility changes

The upgrade requires changes across the stack, rather than a torch-only bump:

- Both classic Megatron and Megatron-FSDP use NVIDIA Megatron-Bridge. The
  launchers omit the removed `vanilla_mbridge` and `grad_offload` settings.
- The existing fully-async trainer remains in use for this experiment. Upstream
  deprecates that path; adopting the V1 trainer needs separate qualification.
- Native CUDA extensions are built for the torch 2.13 ABI. FlashInfer includes
  its CUDA 13 dependencies, and the image checks compilation with the CUDA
  headers and JIT compiler installed on AIR.
- The package lock resolves compatible NumPy, pandas, CUDA and FlashInfer
  dependencies. The OpenCV wheel is pinned by platform and hash because another
  wheel of the same version aborts on AIR's FIPS hosts. See
  [troubleshooting](troubleshooting.md) for that failure.
- A guarded serializer patch makes strided NVIDIA Bridge weights contiguous
  before transfer and avoids data copies for metadata-only relays. The build
  checks the upstream file and exercises numerical reconstruction; the GPU
  training and resume probes also exercise weight synchronization.
- The image retains a separate Ray environment for multi-node judge serving.
  The qualified eight-GPU Flash judge uses single-node multiprocessing.

The [Dockerfile](../docker/Dockerfile) documents the implementation. To update
the package lock after changing its inputs:

```bash
.venv/bin/python scripts/lock_requirements.py
```

The build checks native hashes, torch ABI, CUDA compilation, required imports
and agreement with the package locks. See [build instructions](build-linux.md).

## What has been tested

| Check | Outcome |
|---|---|
| A10 image smoke | Passed the required on-device and import checks |
| Qwen3.5-2B training | Completed two updates, weight synchronization and checkpoint saves on one eight-H100 node |
| Checkpoint resume | Restored trainer and dataloader state at step 2, then saved step 4 |
| GLM-5.3-Flash judge | Served on eight H100s; passed eight calibration cases and 32 requests at concurrency 16 |
| OfficeQA | Audited 100-update Qwen3.5-35B-A3B experiment and both paired evaluations; the [report](../results/officeqa/2026-10-11-final-report.md) explains the manual completion review |

The initial image was about 16.14 GB, below AIR's 20 GB limit. The build applies
a 19.5 GB gate; run `make size` for the image you plan to register. The full
24-H100 math configuration has not been qualified end to end. MuSiQue and
the geo3k ladder were measured on PyTorch 2.11 / verl 0.9. Their job files now
target the current image, but those workloads have not been rerun on it.

## Run the qualification probes

Configure your workspace and image with [setup](setup.md), then run:

```bash
make dev-env
make check
make image
make smoke
make prep        # geo3k data for the 2B training probe
make math-judge  # full Flash weights for the serving probe
make trial-train BUDGET_OK=1
make trial-judge BUDGET_OK=1
```

`trial-train` splits one eight-H100 node between four trainer and four rollout
GPUs and downloads Qwen3.5-2B from Hugging Face. It limits responses to 256
tokens, runs two updates and saves each one.
The 15-minute timeout and zero retries bound an attempt to two H100-hours.
Check the completion certificate before using a checkpoint for a resume test.
The [Flash judge guide](glm-flash-judge-trial.md) describes the serving probe.

These short runs check basic runtime behavior. A new model, a different topology
or a combined judge/training workload still needs its own end-to-end check.
For multi-node workloads, confirm the expected network transport in the NCCL
logs; a single-node probe does not exercise it.
