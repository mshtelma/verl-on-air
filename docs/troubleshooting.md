# Troubleshooting

Ordered by when you hit them. Entries marked **[inherited]** are documented
failures from the reference implementation
([hiouchiy/databricks-air-verl-qwen35](https://github.com/hiouchiy/databricks-air-verl-qwen35))
that this repo already works around — listed so you recognise them if a change
reintroduces one. Entries marked **[predicted]** are reasoned-about but not yet
observed on this stack.

## Setup jobs (data prep / model staging)

Both **observed on df1** and fixed in this repo.

| symptom | cause | fix |
|---|---|---|
| `TypeError: HfFileSystem.find() got multiple values for keyword argument 'maxdepth'` from inside `datasets.load_dataset` | env 5 **preinstalls** a `datasets` that satisfies a loose `>=3.0`, so uv installs nothing (the log shows a 6 s dependency step) and you inherit a mutually incompatible `datasets`/`huggingface_hub` pair | pin **both** to current majors so a coherent pair is actually installed: `datasets>=4.0`, `huggingface_hub>=0.35`, `fsspec>=2024.9`. `prep_geo3k.py` now prints resolved versions first so this is one glance instead of a stack-trace hunt. |
| `RuntimeError: Data processing error: CAS service error : IO Error: Operation not supported (os error 95)` when downloading to a Volume | `os error 95` = `EOPNOTSUPP`. UC Volumes are a FUSE mount that supports sequential writes but **not** the parallel range / sparse-file writes that HF's Xet/CAS and `hf_transfer` backends use. Pointing `snapshot_download(local_dir=<volume>)` at it fails. | `stage_model.py` downloads each file to **local scratch** (full POSIX) then stream-copies it to the Volume sequentially, and `HF_HUB_DISABLE_XET=1`. Per-file rather than per-snapshot bounds scratch to the largest shard (~5 GB) instead of 70 GB. |
| model staging times out part-way through 70 GB | 1×A10 + `timeout_minutes` too low; 67 GiB took **25 min** measured | staging is **resumable** — files already present with a matching byte size are skipped, so just re-run. (The first successful run skipped 4 files left by a failed attempt.) |
| `safetensors` error inside a Ray worker at model load | truncated staging | `stage_model.py` verifies every shard in `model.safetensors.index.json` exists before exiting, so this should surface at staging time instead. |

## `THIS PROJECT ... IS DEPRECATED` / a package that only has an sdist tombstone

**Observed** when pinning the CUDA toolchain explicitly:

```
× Failed to build `nvidia-cuda-crt-cu13==0.0.1`
  ⚠️ THIS PROJECT 'nvidia-cuda-crt-cu13' IS DEPRECATED.
    Please use 'nvidia-cuda-crt' instead.
```

In the CUDA 13 era NVIDIA **unified these package names**; the `-cu13`-suffixed
variants are deprecation tombstones. Measured on the index:

| package | artefacts |
|---|---|
| `nvidia-cuda-nvcc`, `-runtime`, `-crt` | real wheels, 13.3.x |
| `nvidia-cuda-nvcc-cu13`, `-crt-cu13` | **only a `0.0.1` sdist** that fails on purpose |

Two lessons worth generalising:

1. **An index page returning HTTP 200 does not mean the package is usable.** A
   probe of `/simple/nvidia-cuda-nvcc-cu13/` returned 200 and was taken as
   confirmation; the page existed, the package was a tombstone. Same family of
   mistake as "a Docker credential exists" vs "it can push", and "DNS resolves" vs
   "TLS completes". Check the *artefacts*, not the endpoint.
2. **Prefer no pin to a wrong pin** for transitively-supplied packages. These are
   now unpinned so the resolver keeps whatever vllm/torch require; the actual
   guarantee is the `nvcc --version` + `cuda_runtime.h` assertion in Dockerfile
   step 5b, which fails the build loudly if they ever go missing.

## DNS / egress during the build

**Observed and root-caused.** A build on a Databricks corp Linux box died with:

```
Setting up build-essential (12.10ubuntu1) ...        <- apt SUCCEEDED
...
error: Failed to fetch: `https://pypi.org/simple/pybind11/`
  Caused by: dns error: failed to lookup address information
```

`apt` worked, then PyPI did not resolve. **Cause:** the box reaches PyPI only
through an internal proxy configured in `~/.pip/pip.conf`
(`index-url = https://pypi-proxy.dev.databricks.com/simple`). A build container
does **not** inherit that config, so `uv` fell back to `pypi.org`, which is not
routable from there.

Measured reachability from such a box:

| host | reachable | needed? |
|---|---|---|
| `pypi-proxy.dev.databricks.com` | yes | **yes** — the index |
| `github.com` / `objects.githubusercontent.com` | yes | **yes** — verl wheelhouse + git-sourced megatron-core |
| `pypi.org` | **no** | no, when a proxy is configured |
| `download.pytorch.org` | **no** | **no longer needed** — see below |

### The fix is automatic

`make build` detects the local index and passes it through:

```make
PIP_INDEX_URL ?= $(shell python3 -m pip config get global.index-url)
```

It is a build **ARG**, never `ENV` — the proxy is a build-time concern only, and
training nodes have different egress. Override explicitly if needed:

```bash
make build PIP_INDEX_URL=https://mirror.internal/simple
```

### Why download.pytorch.org is no longer required

The Dockerfile used to pull torch from `download.pytorch.org/whl/cu130`, which is
blocked on these boxes. It turns out **PyPI's own `torch==2.11.0` already IS the
CUDA 13 build** — its wheel metadata requires `nvidia-cudnn-cu13`,
`nvidia-nccl-cu13`, `nvidia-cusparselt-cu13`, `nvidia-nvshmem-cu13`. So the plain
PyPI wheel carries exactly the cu130 ABI the wheelhouse binaries were compiled
against, and one fewer host has to be reachable. `torchvision==0.26.0` and
`torchaudio==2.11.0` are on PyPI too (all three verified against the index).

`TORCH_INDEX_URL` is therefore empty by default and torch comes from
`PIP_INDEX_URL`. The build **hard-fails** if `torch.version.cuda` is not `13.x`,
because a CUDA 12 torch would import fine and then die on a GPU node hours later
with `undefined symbol: _ZN3c105Error...`.

### TLS interception: `invalid peer certificate: UnknownIssuer`

**Observed.** DNS was fine, the internal PyPI index worked, and then:

```
Caused by: Failed to fetch: `https://github.com/verl-project/verl-wheelhouse/.../transformer_engine-...whl`
Caused by: invalid peer certificate: UnknownIssuer
```

Two facts explain it:

1. Corporate TLS inspection presents a certificate signed by an **internal CA**.
   The host trusts it (which is why `curl` and `git` work in your shell); a fresh
   container's trust store does not.
2. **`uv` links rustls with BUNDLED webpki roots and ignores the system trust
   store entirely.** So even adding the CA to the image is not enough on its own.

Note the asymmetry: `pypi-proxy.dev.databricks.com` is internal and not
intercepted, so it worked — which is why this only surfaced at the github step.

**Fix (preferred):**

```bash
make certs      # copies the host CA bundle into ./certs
make build      # image trusts it, and UV_NATIVE_TLS=1 makes uv use it
```

The Dockerfile appends anything in `./certs` to `/etc/ssl/certs/ca-certificates.crt`,
runs `update-ca-certificates`, and sets `UV_NATIVE_TLS=1` plus `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `GIT_SSL_CAINFO`. `./certs` holds only a `.gitkeep` by
default, so this is a no-op on open networks. Host bundles are gitignored — they
are host-specific and do not belong in the repo.

**Fix (bulletproof) — remove the need to reach github at all:**

```bash
make vendor     # fetch on the HOST, which already trusts the CA
make build
```

`scripts/vendor_artifacts.sh` downloads the four wheelhouse wheels into
`vendor/wheels/` and clones `megatron-lm` (`core_v0.18.0`) and `mbridge` (pinned
rev) into `vendor/src/`. The Dockerfile prefers those over the URLs whenever
present, so the container never talks to github. This also makes builds
reproducible and much faster to repeat.

> `make doctor` now performs a real **HTTPS handshake** (`curl -r 0-1`) against
> the detected index *and* the TransformerEngine wheel URL, from inside a
> container. The earlier version only did `getent hosts`, which is exactly why it
> reported egress as healthy while the build then failed on TLS.

### `Failed to fetch https://pypi.org/simple/setuptools/` on a git/source install

**Observed** while everything else used the proxy correctly:

```
uv pip install --no-deps "verl @ git+https://github.com/volcengine/verl.git@v0.9.0"
  Failed to resolve requirements from `build-system.requires`
  No solution found when resolving: `setuptools>=61.0`, `wheel`
  Failed to fetch: `https://pypi.org/simple/setuptools/` ... dns error
```

A source install makes uv resolve the package's `build-system.requires` in a
**separate, isolated resolution**. A per-command `--index-url` on the outer
install does *not* reach that inner resolution, so it fell back to `pypi.org`.

Fixed two ways, belt and braces:

1. the index is now set as **ENV** (`UV_DEFAULT_INDEX` / `UV_INDEX_URL`), so every
   uv invocation inherits it, inner resolutions included. Both are **cleared at the
   end of the build**, so the runtime image never carries a build-time mirror the
   training nodes cannot reach;
2. `--no-build-isolation` on the three source installs (verl, megatron-core,
   mbridge) — `setuptools`/`wheel` are already present from step 1, so there is no
   inner resolution to perform at all.

> Lesson generalised: prefer environment configuration over per-command flags for
> anything uv might do in a sub-resolution.

### `UV_NATIVE_TLS` deprecation warning

`warning: The UV_NATIVE_TLS environment variable is deprecated ... Use
UV_SYSTEM_CERTS instead.` The Dockerfile now sets `UV_SYSTEM_CERTS=1` only.
`SSL_CERT_FILE` is also set and is honoured by older uv regardless.

### Other causes, if the index is not the problem

| finding | cause | fix |
|---|---|---|
| host resolves, container does not | systemd-resolved stub (`127.0.0.53`) unreachable from containers | `/etc/docker/daemon.json`: `{"dns": ["8.8.8.8", "1.1.1.1"]}`, then `sudo systemctl restart docker` |
| nothing resolves, host has `HTTPS_PROXY` | build does not inherit the proxy | `make build BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128"` |
| intermittent, succeeds on retry | transient resolver failure | already handled — every network step goes through `docker/retry.sh` (6 attempts, linear backoff) |
| container DNS fine, build still fails | BuildKit network isolation | `make build BUILD_ARGS="--network=host"` |

> `make doctor` probes the **detected** index plus `github.com` from inside a
> container. It deliberately does not require `pypi.org` when a proxy is set.

> The step that originally failed had **no retry at all** while torch/vllm/deps
> had six. Every network step is now wrapped, and exhaustion is a hard failure —
> the old `cmd && break || { warn; }` idiom returned 0 after total failure, which
> would have shipped a silently broken image.

> If you add a `COPY` to the Dockerfile, add it to `.dockerignore` too. That file
> is allow-list style (`*`, then `!scripts`, `!docker/retry.sh`), so an unlisted
> path fails with `failed to compute cache key: not found`.

## STALE IMAGE — re-pushing the same tag does nothing

**This cost a full debug cycle, so read it first when "my fix didn't work".**

`air register image` caches **per image tag**. Re-pushing the same tag with new
content does *not* replace it — the platform keeps serving the digest it
registered:

```
[INFO] Using cached image: sha256:23d37a3c2...     <- the OLD content
```

Symptom: a fix is built, pushed and re-registered, yet the job behaves exactly as
before. In our case the nvcc fix was in the image locally, but every job kept
getting the pre-fix `:v1`. Proof, from a 2-minute diagnostic job:

```
HF_HUB_ENABLE_HF_TRANSFER = 1      <- only set in the OLD image
HF_XET_HIGH_PERFORMANCE   = None   <- the NEW env var was absent
/usr/local/cuda/bin exists : False <- step 5b had never run
```

**Fix / prevention:**

```bash
make bump        # v1 -> v2 in config.env AND every air/*.yaml
make release     # rebuild -> size gate -> push -> register the NEW tag
```

Three guards now exist:

1. `make bump` — one command; rewrites `config.env` plus all YAMLs (they carry the
   image literally on purpose, so they stay hand-submittable).
2. `make stale-check` — fails if the local image for the current tag was built
   *before* the newest change under `docker/` or `scripts/`.
3. The tag is **baked into the image** as `VERL_ON_AIR_IMAGE_TAG`, and `make smoke`
   prints it as its very first check. If it disagrees with `config.env`, you are
   running an old image.

> Rule of thumb: **any** change to `docker/` or `scripts/` that must reach a GPU
> node needs `make bump`. Changes to `scripts/` alone can instead be delivered via
> `code_source: snapshot` (rung 4 and the diag job already do this), which
> bypasses the image entirely — that is why iterating on the launcher is fast while
> iterating on the Dockerfile is not.

## Registering the image

| symptom | cause | fix |
|---|---|---|
| `air register image` prompts for a username/PAT every time | credentials not yet stored, or `SECRET_SCOPE`/`SECRET_KEY` missing from `config.env` | run the interactive flow once, then record the scope/key it prints. `make register` then uses `--scope/--key`. |
| registration **hangs forever** in CI / a piped shell | `--interactive-authenticate` reads the controlling TTY | never use it non-interactively. Set `SECRET_SCOPE`/`SECRET_KEY`; `scripts/bootstrap_linux.sh` warns up front if they are absent. |
| `status=PENDING` for many minutes | normal — the platform pulls and replicates the image. Docs say 2-6 min; a ~16 GB image sits at the slow end | wait. If it never completes, the image is probably too large (see the 20 GB limit) or the credentials cannot pull it. |
| registration succeeds but a workload says `Image not registered` | registration is **per image tag**, per user | re-register after pushing a new tag. Same tag re-pushed with new content also needs re-registration. |
| need a gated HF model in a job | secrets go in the YAML as `scope/key`, not inline | `secrets: { HF_TOKEN: 'msh/hf_token' }` — see `air/02_stage_model.yaml`. |

> Do **not** hand-craft the registry secret with `databricks secrets put-secret`:
> its payload format is internal to `air`. Re-run the interactive flow to rotate.

## Pushing to Docker Hub

| symptom | cause | fix |
|---|---|---|
| `denied: requested access to the resource is denied` after a successful build | the stored credential is for a different account, expired, or is a **Read-only** PAT. Note that a credential merely *existing* in `~/.docker/config.json` proves none of this | `bash scripts/check_dockerhub_push.sh michaelshtelma587 verl-megatron-air` names the exact cause. Usually: `docker login -u michaelshtelma587` with a **Read & Write** PAT from <https://app.docker.io/settings/personal-access-tokens> |
| push denied though login looks fine | repo belongs to another org, or free-plan private-repo quota is exhausted | create `michaelshtelma587/verl-megatron-air` on Docker Hub first, or make it public |

`make push` now verifies push scope via Docker Hub's token endpoint **before**
uploading ~16 GB, and `make doctor` performs the same check. Both are read-only.

## Image build / registration

| symptom | cause | fix |
|---|---|---|
| `air register image` hangs then times out | image >20 GB; the platform cannot replicate it | `make size` before pushing. Confirm `UV_NO_CACHE=1` applied (the uv cache alone is ~11 GB and pushed a comparable image to 31 GB). **[inherited]** |
| `no space left on device` at layer commit | large `COPY` into a layer | never `COPY` a wheelhouse; bind-mount it per-`RUN`. **[inherited]** |
| Job dies at ~1 s, `No module named pip` | AI Runtime's harness imports `pip` and `yaml` before your command runs; a bare base venv has neither | already handled — step 1 of the Dockerfile installs `pip` + `pyyaml`. **[inherited]** |
| every `air run --dry-run` fails with `Image not registered` | `air` validates the YAML schema first but then checks image registration, so before the first `make register` this masks any real schema error | use `make validate`, which swaps in a stock environment so the schema is checked independently of the image. |
| `environment: Value error, 'environment.version' requires inline 'dependencies'` | `environment.version` cannot be used alone | always pair `version:` with a non-empty `dependencies:` list. |
| `ImportError: undefined symbol: _ZN3c105Error...` | CXX11-ABI mismatch between torch and a CUDA extension | do not mix wheel sources. All native wheels here come from verl's wheelhouse, built against torch 2.11/cu130. Check `torch._C._GLIBCXX_USE_CXX11_ABI`. |
| `Python.h: No such file` during the megatron-core step | the base sets `UV_PYTHON_INSTALL_DIR=/opt/uv/python`, so `/opt/venv`'s interpreter may be a uv standalone build while apt's `python3-dev` installed headers for the *system* python | already asserted — step 1 of the Dockerfile fails the build with the interpreter path and its `sysconfig` include dir. Follow the message: install headers for that interpreter, or export `CPPFLAGS=-I<build>/include/pythonX.Y`. |
| CUDA `Error 803`, or NCCL `no GPUs found` while `nvidia-smi` works | a `cuda-compat` ahead of the driver's `libcuda` pins userspace *below* the kernel driver | never add `cuda-compat` to this image. The base ships none on `LD_LIBRARY_PATH`; the smoke test asserts it stays that way. |
| `torch.cuda.is_available()` is False on an H100 node | host driver older than CUDA 13.0's 580.65.06 minimum, so the cu130 wheels cannot init | the smoke test checks the driver explicitly and prints the floor. AWS documents 580.126.16; if a pool is older you cannot use cu130 wheels, and the CUDA-12 fallback means source-building TE/apex/flash-attn (the wheelhouse is cu130-only). |
| build installs an unrelated `apex` | PyPI has a package literally named `apex` that is not NVIDIA's | already handled — the Dockerfile pins wheels by **direct URL**, never by index resolution. |
| `Python.h: No such file` | missing dev headers for megatron-core's pybind11 ext | already handled (`python3.12-dev`). **[inherited]** |

## Runtime — process dies immediately

| symptom | cause | fix |
|---|---|---|
| `FATAL FIPS SELFTEST FAILURE`, `Fatal Python error: Aborted` on `import cv2` | `opencv-python-headless` 5.x bundles a FIPS-enforcing `libcrypto`; `transformers` imports cv2 via `mistral_common` | already handled — pinned to `4.12.0.88` as the **last** pip op. `OPENSSL_*` env vars cannot fix it; the blob is vendored. **[inherited]** |
| `ssl.SSLError: [CRYPTO] unknown error (_ssl.c)` | air hosts run a FIPS kernel; non-FIPS crypto fails to init | `OPENSSL_FORCE_FIPS_MODE=0` (set in the image and every YAML). |
| Ray: `expected a valid path like mymodule.provider_class` | `RAY_RUNTIME_ENV_HOOK` set to `""` | never set it to empty. Unset it entirely. **[inherited]** |
| `No module named 'triton'` / GDN kernel compile failure | Triton JITs a host C launcher stub at runtime and needs `cc`/`gcc` | already handled — `build-essential` is deliberately **kept** in the image. The smoke test asserts a compiler is on PATH. |

## Runtime — CUDA JIT (FlashInfer / Triton)

**Observed on rung 1**, after a clean build and a passing smoke test:

```
subprocess.CalledProcessError: Command '['ninja', '-v', '-C',
  '/root/.cache/flashinfer/0.6.12/90a/cached_ops/gdn_prefill_sm90', ...]'
  returned non-zero exit status 127
-> RuntimeError: Ninja build failed
-> vllm.v1.engine.exceptions.EngineDeadError
-> RuntimeError: Sync replay buffer selected terminal groups with no
   materializable trajectories
```

Read it inside-out: `127` = **command not found**. vLLM's FlashInfer
**JIT-compiles the Qwen3.5 Gated-DeltaNet prefill kernel at runtime** and needs
`nvcc`. Everything downstream (`EngineDead`, "no materializable trajectories") is
just the rollout engine dying and GRPO finding empty groups.

The root error was a reasoning slip: *"we install only prebuilt wheels, so nothing
CUDA compiles"* is true at **build** time and false at **run** time. Exactly the
same shape as Triton needing a host C compiler at runtime — which we did account
for, and which is why `build-essential` is deliberately kept.

Measured on a node with `scripts/diag_cuda.py`:

| fact | value |
|---|---|
| `which nvcc` | **None** |
| nvcc actually present at | `<site-packages>/nvidia/cu13/bin/nvcc` (13.2.86, runs) |
| `/usr/local/cuda/bin` | **MISSING** |
| `/usr/local/cuda/include` | only `nvtx3` |
| `CUDA_HOME` | `/usr/local/cuda` — i.e. an incomplete tree |

So nothing needed installing; the toolchain merely had to be discoverable. The
Dockerfile now symlinks the pip tree's `bin/` and `nvvm/` into `/usr/local/cuda`,
grafts its headers into `/usr/local/cuda/targets/x86_64-linux/include`, and puts
`/usr/local/cuda/bin` on `PATH`. `CUDA_HOME=/usr/local/cuda` stays valid, the base
image's `lib64` is untouched, and the build asserts both `nvcc --version` and the
presence of `cuda_runtime.h`.

`make smoke` now compiles a real `sm_90` test kernel with `nvcc`, so this class of
failure costs two A10-minutes instead of eight H100-minutes.

One wrinkle in the graft, hit on the first attempt:

```
ln: /usr/local/cuda/targets/x86_64-linux/include/nvtx3: cannot overwrite directory
```

The base image ships `include/nvtx3` as a real **directory** and the pip tree has
one too, so a bare `ln -sfn "$f" "$INC/"` fails. The loop therefore **skips any
entry the base already provides** — which is also the safer merge order: the base
image's headers win and we only add what is missing (`cuda_runtime.h` and
friends). The step prints `headers: N linked, M already provided` so the outcome is
visible.

> First rollout on a fresh node still pays a one-off JIT cost while FlashInfer
> builds the GDN kernels into `/root/.cache/flashinfer`. That cache does not
> persist between jobs.

### ...and then: "CUDA compiler and CUDA toolkit headers are incompatible"

Making nvcc discoverable moved rung 1 from **exit 127** to **exit 1** — nvcc now ran,
and failed differently:

```
cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:
  error "CUDA compiler and CUDA toolkit headers are incompatible,
         please check your include paths"
```

CCCL enforces, roughly:

```c
CUDA_VERSION != __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10
```

so **MAJOR.MINOR must agree**; the patch level is free. The transitive resolution
had produced a skewed set:

| package | version | contributes |
|---|---|---|
| `nvidia-cuda-nvcc` | 13.2.86 | `__CUDACC_VER_*` → **13020** |
| `nvidia-cuda-runtime` | 13.0.96 | `CUDA_VERSION` → **13000** |
| `nvidia-cuda-crt` | 13.3.73 | — |

torch is cu130, so the toolkit is 13.0 and **nvcc was the odd one out**. Step 5a2
pins `nvidia-cuda-nvcc` and `nvidia-cuda-crt` to `13.0.88`, after every other pip
install so nothing can re-upgrade them, and asserts numerically that
`__CUDACC_VER == CUDA_VERSION`. `runtime`/`nvrtc` are left to torch — 13.0.96 and
13.0.88 both yield 13000, which is all CCCL checks.

**Why the first smoke check missed it:** it compiled a kernel including only
`cuda_runtime.h`, which succeeds *even with a skewed toolchain*. The version assert
lives in **CCCL** headers, which that probe never pulled in. Both the build step and
`make smoke` now compile `#include <cuda/std/type_traits>` with FlashInfer's own
`-I` paths and `-gencode=arch=compute_90a,code=sm_90a` — i.e. they reproduce the
real compile.

The strengthened check was validated **against the known-bad image** before the fix
was built, and reproduced the 8xH100 failure on a 2-minute A10 job:

```
FAIL  nvcc usable for runtime JIT (FlashInfer GDN): ... headers are incompatible
ok    CUDA toolchain versions -> nvcc=13.2.86 runtime=13.0.96 crt=13.3.73
      <-- nvcc/runtime MAJOR.MINOR differ, CCCL will reject
```

> General lesson: **a probe must exercise the same code path as the real workload.**
> "nvcc runs" and "nvcc compiles what FlashInfer compiles" are different claims, and
> only the second one was worth anything.

## Runtime — Megatron / config

| symptom | cause | fix |
|---|---|---|
| `use_megatron_fsdp` appears to do nothing | `vanilla_mbridge=True` — verl only threads it through the Megatron-**Bridge** provider path, legacy mbridge silently ignores it | fsdp mode sets `vanilla_mbridge=False`. Never combine `MEGATRON_MODE=fsdp` with `vanilla_mbridge=True`. |
| FSDP throughput far below expectation, no error | `CUDA_DEVICE_MAX_CONNECTIONS=1` serialises FSDP collectives behind compute | the launcher **unsets** it in fsdp mode and exports `=1` only in classic. If you hand-edit, preserve this. |
| Megatron-FSDP crash mentioning gradient accumulation fusion | incompatible with Megatron-FSDP | `gradient_accumulation_fusion=False` (set in fsdp mode). |
| shape/stride errors in attention, or GDN complaining about packed input | Qwen3.5 GDN has no THD support in Megatron-LM | `use_remove_padding=False` on **both** `model.` and `actor.megatron.`, plus `use_dynamic_bsz=False` everywhere. Non-negotiable. |
| `real_train_batch_size (N) must be divisible by minimal possible batch (M)` | `train_batch_size × rollout_n` not divisible by world GPUs | the launcher pre-checks this and fails fast with the arithmetic. Adjust `train_batch_size` or `rollout_n`. **[inherited]** |
| `'set' object is not subscriptable` during model wrap | verl bug where `model._no_split_modules` resolves to a `set` | FSDP2-path bug from the reference repo; the Megatron path here does not hit it. If you switch to the FSDP2 backend, pass an explicit layer-class list. **[inherited]** |

## Runtime — memory

| symptom | cause | fix |
|---|---|---|
| OOM at optimizer construction, 8 GPUs, 35B | ZeRO-1 replicates params+grads; ~390 GB of Adam state has nowhere to go | this is expected. Use `OFFLOAD=1` (rung 3) or 16 GPUs + fsdp (rung 4). See [sizing.md](sizing.md). |
| Opaque host OOM / job killed during optimizer build with `OFFLOAD=1` | node has <~550 GiB RAM | read `cpu ram` from `make smoke`. If short, rung 4 (no offload) is the only option. |
| OOM only during rollout | vLLM `gpu_memory_utilization` too high alongside resident training state | lower `ROLLOUT_GPU_MEM_UTIL` (0.6 → 0.5). `free_cache_engine=True` is already on so the two peaks do not sum. |
| OOM in log-prob / entropy | vocab is 248320; un-chunked logits are ~3 GB per micro-batch | `entropy_from_logits_with_chunking=True` (already set for actor and ref). |
| `tensor too large to fit in the bucket` during weight sync | a single tensor exceeds the actor→vLLM transfer bucket; embedding is 248320×2048 | set `WEIGHT_BUCKET_MB=6144`. Left unset by default because the config path moved between verl releases — if hydra rejects `rollout.checkpoint_engine.update_weights_bucket_megabytes`, try `rollout.update_weights_bucket_megabytes`. **[inherited]** |

## Multi-node

| symptom | cause | fix |
|---|---|---|
| Job stays `RUNNING` forever after training finishes; billing continues | worker used `ray start --block` and never exits | already handled — workers poll the head's GCS port and `exit 0` when it disappears. Never reintroduce `--block`. **[inherited]** |
| Worker lingers after the head **fails** | cleanup placed after the launch line; `set -e` + `pipefail` skips it | already handled via `trap cleanup EXIT` on the head. **[inherited]** |
| `only N/16 GPUs registered after 15 min` | worker could not reach the head | check `make logs RUN=... NODE=1`. Confirm `MASTER_ADDR`/`MASTER_PORT` are injected (multi-node only — they are absent on single-node). |
| NCCL reports `NET/Socket`; throughput collapses | EFA did not bind, fell back to TCP | rung 4 sets `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET`; grep for `NET/OFI ... Provider is efa`. Check `ls /sys/class/infiniband` on the node (the smoke test prints it). Do **not** set `OVERRIDE_NCCL=1` before confirming `NET/OFI` works without it. |
| `NET/Plugin ... failed to load` / aws-ofi-nccl skipped | version skew between torch's bundled NCCL and the plugin the base built against `libnccl2` | **not benign on AWS** — EFA reaches NCCL *through* `aws-ofi-nccl`, so losing the plugin loses RDMA outright. (Azure differs: NCCL speaks IB verbs natively there, so the same failure would only cost SHARP.) On the `-cu13` base both are on the ~2.28 line, so skew should be small; keep `OVERRIDE_NCCL=0`. |
| no EFA devices on a 1×A10 job | G-family hosts have no EFA hardware | expected and harmless — the smoke test warns rather than fails. EFA is only required for multi-node H100. |
| `NET/OFI ... initialization failed` WARN ×3 on A10 jobs | same cause: the base ships the plugin, the hardware isn't there | harmless; NCCL falls back to sockets and a 1-GPU job doesn't care. Silence with `NCCL_NET_PLUGIN: "none"`. |

## Training signal

| symptom | cause | fix |
|---|---|---|
| Reward is flat; loss ~0; nothing learns | every sample in each group scores identically → advantage 0 → **no gradient** | not a bug. Run `make baseline`: it reports the fraction of groups with non-zero reward variance. Raise `rollout_n`, use harder data, or switch checkpoint. |
| Reward stuck at exactly 0.00 | responses contain no `\boxed{}`. geo3k's accuracy term is gated on `\boxed{}` extraction, so a *correct but unboxed* answer scores 0.00, not 0.90 | ensure the prompt carries the `<think>`/`\boxed{}` instruction (`scripts/prep_geo3k.py` injects it). Verify the surface with `python3 scripts/reward/custom_reward.py`. |
| Reward pinned near 1.0 from step 0 | model already solves the task | saturated — see `make baseline` verdict. Move to `math_dapo`/`aime` or the `-Base` checkpoint. |
| MLflow shows only `score` | you returned a float from a custom reward | return a dict with a `score` key; other keys become separate metrics. |

## Checkpointing (Megatron-FSDP specifics)

Per verl's Megatron-FSDP docs, on this path:

- checkpoints are **DTensor** checkpoints under `dist_ckpt`
- requires `use_distributed_optimizer=True` (verl's default)
- requires `CUDA_DEVICE_MAX_CONNECTIONS` unset or >1 (handled by the launcher)
- **optimizer state cannot be saved alone** — include `model` whenever
  `optimizer` is in `checkpoint.save_contents`
- `checkpoint.async_save=True` is not covered for FSDP DTensor checkpoints
- PEFT + Megatron-FSDP save/load is not covered upstream

The ladder ships with `SAVE_FREQ=-1` (no saving) to keep smoke runs fast. Set
`SAVE_FREQ` before a real run, and expect the first save to be where any of the
above surfaces.
