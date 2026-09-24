# Troubleshooting

Grouped by when you hit them. Unless a row says otherwise, the fix is already in the repo; the
row is here so you recognise the failure if a change brings it back.

## Data prep and model staging

| symptom | cause | fix |
|---|---|---|
| `TypeError: HfFileSystem.find() got multiple values for keyword argument 'maxdepth'` inside `datasets.load_dataset` | the stock environment preinstalls a `datasets` that satisfies a loose pin but does not match its `huggingface_hub` | pin both to current majors: `datasets>=4.0`, `huggingface_hub>=0.35`, `fsspec>=2024.9` (the prep jobs do) |
| `CAS service error : IO Error: Operation not supported (os error 95)` when downloading to a Volume | UC Volumes (a FUSE mount) take sequential writes only; HF's Xet and `hf_transfer` backends write parallel ranges | download to local scratch, then copy, with `HF_HUB_DISABLE_XET=1`. `stage_model.py` does both, one file at a time, so scratch only needs room for the largest shard (~5 GB) |
| model staging times out | 70 GB from one A10 takes about 25 minutes, plus queue time | re-run it: staging is resumable and skips files whose hash matches the Hub's |
| `safetensors` error in a Ray worker at model load | a truncated shard | `stage_model.py` checks every shard in the index before it exits; re-stage |
| a job needs a gated HF model | secrets are referenced as `scope/key` in the YAML, never inline | `secrets: { HF_TOKEN: '<scope>/hf_token' }`, as in `infra/air/stage_model.yaml` |

## Building the image

The build needs your PyPI index and GitHub, reached from inside the build container. `make
doctor` checks both with a real HTTPS request from a container, which is the test that matters:
a host that resolves a name says nothing about the container.

| symptom | cause | fix |
|---|---|---|
| `Failed to fetch: https://pypi.org/simple/...` with `dns error` | the host reaches PyPI through a proxy, and the container does not inherit your pip or uv config | `make build` detects the index and passes it as a BuildKit secret; override with `make build PIP_INDEX_URL=https://<proxy>/simple` ([build-linux.md](build-linux.md)) |
| `invalid peer certificate: UnknownIssuer` on a github.com URL | TLS inspection signs with an internal CA that the container does not trust | `make certs && make build`: the host CA bundle goes into `./certs` and the image trusts it (`UV_SYSTEM_CERTS=1`). Or `make vendor && make build`: the host fetches the GitHub artifacts (checked against `docker/artifacts.lock`) and the container never contacts GitHub |
| `Failed to fetch https://pypi.org/simple/setuptools/` while installing from a git source | uv resolves a source tree's `build-system.requires` separately, and a per-command `--index-url` does not reach it | install through `uvi` (it exports the index as environment) with `--no-build-isolation`, as the Dockerfile's source installs do |
| host resolves names, the container does not | the systemd-resolved stub (`127.0.0.53`) is unreachable from containers | `/etc/docker/daemon.json`: `{"dns": ["8.8.8.8", "1.1.1.1"]}`, then `sudo systemctl restart docker` |
| nothing resolves; the host uses `HTTPS_PROXY` | the build does not inherit the proxy | `make build BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"` |
| container DNS works, the build still cannot connect | BuildKit network isolation | `make build BUILD_ARGS="--network=host"` |
| occasional network failures | transient errors | every network step retries (`docker/retry.sh`, 6 attempts) and fails the build when the retries run out |
| `failed to compute cache key: not found` | `.dockerignore` is an allow-list, and a new `COPY` path is not on it | add the path to `.dockerignore` |
| `nvidia-cuda-crt-cu13==0.0.1` fails with `THIS PROJECT ... IS DEPRECATED` | since CUDA 13, NVIDIA publishes these packages without the `-cu13` suffix; the suffixed names are placeholder sdists | use `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-crt` |
| `/bin/sh: ...: not found` right after a `` `# comment` `` in a `RUN` | a backtick comment in command position leaves an empty command, so the next assignment runs as a command | put comments on `#` lines above the `RUN`; `scripts/lint_dockerfile.py` flags the pattern |
| `air register image` hangs, then times out | the image is over the 20 GB limit | run `make size` before pushing, and check that `UV_NO_CACHE=1` took effect (uv's cache alone is ~11 GB) |
| `no space left on device` at a layer commit | a large `COPY` | bind-mount wheel directories per `RUN`; never `COPY` them |
| a job dies after about a second with `No module named pip` | AI Runtime imports `pip` and `yaml` before your command runs | keep them in the image (the Dockerfile's first step installs both) |
| `ImportError: undefined symbol: _ZN3c105Error...` | a CUDA extension built against a different torch ABI | take every native wheel from verl's wheelhouse (torch 2.11, cu130); check `torch._C._GLIBCXX_USE_CXX11_ABI` |
| `Python.h: No such file` in the megatron-core step | the base sets `UV_PYTHON_INSTALL_DIR=/opt/uv/python`, so apt's `python3-dev` can belong to another interpreter than `/opt/venv`'s | the build stops early and prints the interpreter's include directory; install headers for that interpreter or export `CPPFLAGS=-I<include dir>` |
| an unrelated `apex` gets installed | PyPI has a different package named `apex` | install NVIDIA's apex from its pinned URL (`docker/artifacts.lock`) |
| `'environment.version' requires inline 'dependencies'` | `environment.version` used on its own | pair `version:` with a non-empty `dependencies:` list |
| every `air run --dry-run` fails with `Image not registered` | air checks registration after the schema, so an unregistered image hides schema errors | `make validate` swaps in a stock environment and checks the schema alone |

## A fix that does not take effect

Registration is cached per image tag. Push new content under an existing tag and jobs keep
getting the digest registered first; the registration log shows `Using cached image:
sha256:...`. After any change under `docker/` or `certs/`:

```bash
make bump        # next tag in config.env and in every custom-image job
make release     # rebuild, size gate, push, register
```

`make push` runs `make stale-check`, which refuses an image not built from the current inputs
and a tag already pushed from other inputs. `make register` requires the registry to serve the
digest `docker/IMAGE.lock` records. `make smoke` prints the tag baked into the image first; if
it disagrees with `config.env`, you are running an old image. Code under `engine/`,
`usecases/`, `infra/` and `scripts/` never needs a new tag, because jobs upload it as a
snapshot.

## Registering and pushing

| symptom | cause | fix |
|---|---|---|
| `air register image` asks for a username and PAT every time | no stored credentials, or `SECRET_SCOPE`/`SECRET_KEY` missing from `config.env` | run the interactive flow once and record the scope and key it prints ([setup.md](setup.md)) |
| registration hangs in CI or a piped shell | the interactive flow reads the terminal | set `SECRET_SCOPE`/`SECRET_KEY`; `scripts/bootstrap_linux.sh` warns when they are missing |
| `status=PENDING` for several minutes | normal: the platform pulls and replicates the image (2-6 min, longer for a large image) | wait. If it never finishes, the image is too large or the credentials cannot pull it |
| a job says `Image not registered` after a successful registration | registration is per tag and per user | register every new tag |
| `denied: requested access to the resource is denied` on push | the stored Docker credential belongs to another account, has expired, or is read-only | `bash scripts/check_dockerhub_push.sh <dockerhub-user> <image-name>` names the cause; usually `docker login -u <dockerhub-user>` with a Read & Write token |
| push denied although login works | the repository belongs to another organisation, or the free plan's private-repo quota is used up | create the repository on Docker Hub first, or make it public |

`make push` checks push scope before uploading ~16 GB. Don't write the registry secret with
`databricks secrets put-secret`: its format is internal to `air`. Rotate it with the
interactive flow.

## A job dies at start

| symptom | cause | fix |
|---|---|---|
| `FATAL FIPS SELFTEST FAILURE`, then `Fatal Python error: Aborted` on `import cv2` | `opencv-python-headless` 5.x bundles a FIPS-enforcing libcrypto, and transformers imports cv2 through `mistral_common` | the Dockerfile installs `opencv-python-headless==4.12.0.88` as its last pip step; keep it last. `OPENSSL_*` variables cannot fix this one |
| `ssl.SSLError: [CRYPTO] unknown error (_ssl.c)` | AI Runtime hosts run a FIPS kernel, and non-FIPS crypto fails to initialise | `OPENSSL_FORCE_FIPS_MODE=0` and `OPENSSL_FIPS=0`, set in the image (the stock-environment jobs set them in their YAML). What that trades off: [security.md](security.md) |
| Ray: `expected a valid path like mymodule.provider_class` | `RAY_RUNTIME_ENV_HOOK` set to an empty string | unset it |
| `No module named 'triton'`, or a Gated-DeltaNet kernel fails to compile | Triton compiles a small C launcher at runtime and needs `cc` | keep `build-essential` in the image; the smoke test checks for a compiler |

## CUDA compilation at runtime

vLLM compiles Qwen3.5's Gated-DeltaNet prefill kernel with nvcc (through FlashInfer) the first
time it runs, so the image needs a working CUDA toolchain at run time, not only at build time.

| symptom | cause | fix |
|---|---|---|
| `Ninja build failed` with exit status 127 under `/root/.cache/flashinfer/`, then `EngineDeadError` and "no materializable trajectories" | nvcc is not on `PATH`: the pip CUDA toolchain lives under `site-packages/nvidia/cu13` | the Dockerfile links it into `/usr/local/cuda` and puts `/usr/local/cuda/bin` on `PATH` |
| `CUDA compiler and CUDA toolkit headers are incompatible`, or `ptxas fatal: Unsupported .version 9.2` | the `nvidia-cuda-*` packages are at different versions; nvcc, ptxas and the headers must agree on MAJOR.MINOR | keep them on one version. The Dockerfile installs the set at `CUDA_TOOLCHAIN_VERSION` (13.2.86) after every other package and asserts they agree; `make smoke` compiles a CCCL kernel for `sm_90a` the way FlashInfer does |

The first rollout on a fresh node still spends a minute or so compiling into
`/root/.cache/flashinfer`. That cache does not carry over between jobs.

## Megatron and verl settings

| symptom | cause | fix |
|---|---|---|
| segfault in `transformer_engine::multi_tensor_scale` on every rank after the first rollout; Ray reports `SYSTEM_ERROR ... connection error code 2` with no Python traceback | `use_precision_aware_optimizer=True` under Megatron-FSDP: gradient clipping calls TE's fused scale on the precision-aware buffers | the launcher enables it in classic mode only. A missing Python traceback does not mean OOM; look for a native stack earlier in the log |
| `use_megatron_fsdp` has no effect | `vanilla_mbridge=True`: only the Megatron-Bridge path passes it on | fsdp mode sets `vanilla_mbridge=False`; don't combine the two |
| FSDP throughput far below expectation, no error | `CUDA_DEVICE_MAX_CONNECTIONS=1` serialises FSDP collectives behind compute | the launcher unsets it in fsdp mode and sets `1` only in classic mode; don't set it in a job file |
| a Megatron-FSDP error about gradient accumulation fusion | the two are incompatible | `gradient_accumulation_fusion=False` (fsdp mode sets it) |
| shape or stride errors in attention, or Gated-DeltaNet rejecting packed input | Qwen3.5's Gated-DeltaNet has no packed-sequence (THD) support in Megatron-LM | `use_remove_padding=False` on both `model.` and `actor.megatron.`, and `use_dynamic_bsz=False` everywhere (both launchers set these) |
| `real_train_batch_size (N) must be divisible by minimal possible batch (M)` | `train_batch_size × rollout_n` is not divisible by the trainer GPUs | the launcher checks this first and prints the numbers; change `train_batch_size` or `rollout_n` |
| `'set' object is not subscriptable` while wrapping the model | a bug in verl's FSDP2 backend (`_no_split_modules` is a set) | the Megatron path does not hit it; with FSDP2, pass an explicit list of layer classes |

## Memory

| symptom | cause | fix |
|---|---|---|
| OOM building the optimizer, 35B on 8 GPUs | ZeRO-1 replicates params and grads, and ~390 GB of Adam state has nowhere to go | expected: use `OFFLOAD=1` (rung 3) or Megatron-FSDP on more GPUs (rung 4). See [sizing.md](sizing.md) |
| host OOM, or the job is killed while building the optimizer with `OFFLOAD=1` | the node has less than ~550 GiB of RAM | read `cpu ram` on an H100 job; if it is short, use rung 4's layout (no offload) |
| OOM during rollout only | vLLM's memory fraction is too high next to the resident training state | lower `ROLLOUT_GPU_MEM_UTIL` (0.6 to 0.5). `free_cache_engine=True` already keeps the two peaks from adding up |
| `No available memory for the cache blocks` | the KV cache cannot hold a single `MAX_MODEL_LEN` sequence | raise `ROLLOUT_GPU_MEM_UTIL` if nothing else lives on those GPUs, raise `GEN_TP`, or lower `MAX_MODEL_LEN` ([tuning.md](tuning.md)) |
| OOM in log-prob or entropy | a 248,320-token vocabulary: un-chunked logits are ~3 GB per micro-batch | `entropy_from_logits_with_chunking=True` (set for actor and ref) |
| `tensor too large to fit in the bucket` during the weight sync | one tensor is larger than the actor-to-vLLM bucket (the embedding is 248320×2048) | set `WEIGHT_BUCKET_MB=6144`. If Hydra rejects `rollout.checkpoint_engine.update_weights_bucket_megabytes`, try `rollout.update_weights_bucket_megabytes` |
| OOM at the `on_step_end` weight sync in fsdp mode, in `uneven_dtensor_to_full_tensor` during "Converting to HuggingFace" | co-located sync: ZeRO-3 gathers each parameter into a full tensor (~1.9 GiB for the largest expert) while the woken vLLM holds ~15-17 GiB of weights | add GPUs to thin the shards (35B: 16 to 32). `ROLLOUT_GPU_MEM_UTIL` does not help, because it sizes the KV cache, which is asleep during the sync; `enforce_eager` saves about 2 GiB; offload is not available with FSDP. Details: [ladder.md](ladder.md) |
| `custom_all_reduce.cuh:455 'invalid argument'` at CUDA-graph capture, or a hang that ends in `RPC call to sample_tokens timed out` (8-way TP inside one H100 node) | vLLM's intra-node custom all-reduce fails on these nodes | turn it off, and NCCL does the all-reduce: `--disable-custom-all-reduce` for servers (`serve_and_eval.sh` and `serve_judge.sh` pass it), `disable_custom_all_reduce=True` for offline `LLM(...)`, `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=True` for training rollout |
| OOM in the actor backward (`fla` `l2norm_bwd`) with Megatron-FSDP at TP=1 and 12-turn episodes | Gated-DeltaNet activations for ~28k-token episodes | TP=2 halves them; lowering `MAX_TURNS` shortens the episode. The sync search recipe still does not fit as configured ([training-modes.md](training-modes.md)) |

## Multi-node

| symptom | cause | fix |
|---|---|---|
| a job stays `RUNNING` after training finished, and keeps billing | a worker ran `ray start --block` | workers poll the head's GCS port and exit when it goes away; don't bring back `--block` |
| a worker keeps running after the head fails | cleanup placed after the launch line is skipped under `set -e` | the head installs its `EXIT` trap before `ray start` |
| `only N/16 GPUs registered` after 15 minutes | a worker could not reach the head | read the other node's log (`make logs RUN=<id> NODE=1`); `MASTER_ADDR` and `MASTER_PORT` are injected only on multi-node jobs |
| NCCL reports `NET/Socket`, and throughput collapses | EFA did not bind, so NCCL fell back to TCP | set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET` (rung 4 does), grep for `NET/OFI ... Provider is efa`, and check `ls /sys/class/infiniband` on the node |
| `NET/Plugin ... failed to load`, with `aws-ofi-nccl` skipped | torch's bundled NCCL and the EFA plugin disagree on the NCCL version | this loses RDMA, since EFA reaches NCCL through that plugin. Keep the build arg `OVERRIDE_NCCL=0`, which leaves torch's NCCL in place |
| `NET/OFI ... initialization failed` warnings, or no EFA devices, on an A10 job | A10 hosts have no EFA hardware | harmless for a one-GPU job; silence it with `NCCL_NET_PLUGIN: "none"` |

## Training signal

| symptom | cause | fix |
|---|---|---|
| reward flat, loss near 0, nothing learns | every sample in a group scores the same, so the advantage is 0 and only the KL term acts | run `make baseline` for the fraction of groups whose rewards differ. Raise `rollout_n`, use harder data, or start from another checkpoint |
| reward stuck at exactly 0.00 on geo3k | no `\boxed{}` in the responses; geo3k's accuracy term needs it, so a correct unboxed answer scores 0.00, not 0.90 | keep the `<think>`/`\boxed{}` instruction in the prompt (`infra/geo3k/prep_geo3k.py` adds it); `python3 infra/geo3k/reward.py` shows how answers score |
| reward near 1.0 from the first step | the model already solves the task | harder data, or the `-Base` checkpoint |
| MLflow shows only `score` | the custom reward returned a float | return a dict with a `score` key; every other key becomes its own metric |
| the log ends with `[certificate] NOT CERTIFIED` | the run did not write and verify its planned final checkpoint (a crash the Rollouter swallowed, an abort, a timeout), whatever verl's exit code was | the reasons are in `run_result.json` next to the run's checkpoints ([running-jobs.md](running-jobs.md)) |

## Checkpointing with Megatron-FSDP

From verl's Megatron-FSDP documentation:

- checkpoints are DTensor checkpoints under `dist_ckpt`;
- `use_distributed_optimizer=True` is required (verl's default);
- `CUDA_DEVICE_MAX_CONNECTIONS` must be unset or greater than 1 (the launcher handles it);
- optimizer state cannot be saved on its own: include `model` whenever `optimizer` is in
  `checkpoint.save_contents`;
- `checkpoint.async_save=True` is not supported for FSDP DTensor checkpoints;
- PEFT with Megatron-FSDP is not supported upstream.

The ladder rungs run with `SAVE_FREQ=-1`, so the first real save is where any of this shows up.
