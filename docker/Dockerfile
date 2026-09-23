# syntax=docker/dockerfile:1.10
# ^ REQUIRED, and must be the first line: the self-verification step at the end
#   uses a `RUN ... <<'PY'` heredoc (BuildKit frontend >= 1.4), and the package
#   index arrives through a secret mount (`--mount=type=secret`).
#   Without this directive an older default frontend fails with
#   "Dockerfile parse error: unknown instruction: IMPORTLIB".
#
# =============================================================================
# verl + Megatron/mcore (Megatron-Bridge + Megatron-FSDP) + vLLM rollout
# for Qwen3.5-35B-A3B GRPO on Databricks AI Runtime serverless GPU.
#
# WHY THIS BASE  (an AWS workspace)
#   The target workspace runs on AWS. That matters twice over:
#
#   1. CUDA 13 NATIVELY. The -cu13 base tags are published for AWS ONLY
#      (verified against the Docker Hub tag list: dcs-base-aws-{devel,runtime}-cu13
#      exist; the Azure side has only dcs-base-azure-{devel,runtime}, both CUDA
#      12.9.1). This image is CUDA 13.0.3 with NCCL 2.28.3+cuda13.0, which lines
#      up with our torch 2.11.0+cu130 and the cu130 wheelhouse binaries. Running
#      on df2/Azure instead would mean cu130 wheels on a 12.9 base -- workable,
#      but it rests on a four-step argument about bundled CUDA libs, soname
#      major-versioning, host driver floors and cuda-compat absence. Here none of
#      that is needed: the toolchain matches.
#
#   2. RDMA. Multi-node H100 needs the cloud's RDMA stack, pre-wired by the
#      Databricks base. On AWS that is EFA: /opt/aws-ofi-nccl/install/lib,
#      /opt/amazon/{openmpi,efa}/lib are already on LD_LIBRARY_PATH.
#
#   We use the *runtime* (not devel) variant: ~4.7 GB vs ~11 GB. That headroom is
#   the whole reason this fits under the 20 GB DCS cap, because the Megatron stack
#   (TransformerEngine + apex + flash-attn) is heavy.
#
#   Also verified from the registry config: cuda-compat is NOT on this image's
#   LD_LIBRARY_PATH. Never add one -- a cuda-compat ahead of the driver's libcuda
#   pins userspace BELOW the kernel driver and CUDA fails with Error 803.
#
# WHY NO CUDA DEVEL BASE
#   verl publishes a prebuilt wheelhouse of cu130 / torch-2.11 / cp312 native
#   wheels: https://verl-project.github.io/verl-wheelhouse/simple/
#   TransformerEngine, apex and flash-attn all come from there as binaries, so no
#   heavy CUDA extension compiles at build time -- which is why the runtime base
#   suffices. (The build does compile megatron-core's small pybind11 extension,
#   and step 5c deliberately compiles a CUDA probe with the pip-installed nvcc.)
#   We install them by DIRECT WHEEL URL, each checked against its sha256 in
#   docker/artifacts.lock, rather than via --extra-index-url: PyPI has an
#   unrelated package literally named `apex`, and an index-resolution race would
#   silently install the wrong one.
#
# WHY build-essential STAYS IN THE IMAGE
#   Two reasons, both runtime not build-time:
#     1. megatron-core is a source install (one pybind11 C++ extension, no CUDA).
#     2. Qwen3.5's Gated-DeltaNet path runs *Triton* kernels, and Triton JITs a
#        C launcher stub at runtime -> it needs a host C compiler on PATH.
#   Purging the toolchain saves ~400 MB and then dies hours into the first run.
#
# VERSION SET -- based on verl v0.9.0's docker/Dockerfile.stable.vllm, NOT a copy
#     torch 2.11.0 (cu130) · vllm 0.24.0 · transformers 5.5.3
#     TransformerEngine 2.16.1 · flash-attn 2.8.3 · apex (wheelhouse build, by sha256)
#     megatron-core core_v0.18.0 (ba7b5eb) · megatron-bridge 0.5.2 · mbridge 641a5a0
#     verl v0.9.0 (483b8a0) · ray 2.58.0 (+ 2.48.0 for the judge, step 5d)
#   Every other Python package is pinned by docker/requirements.lock (constraints),
#   every non-index artefact by docker/artifacts.lock (sha256 / commit), the base by
#   digest. A tag's pushed digest is recorded in docker/IMAGE.lock.
#
#   Deliberate deviations from upstream Dockerfile.stable.vllm (v0.9.0):
#     base            nvidia/cuda:13.0.2-devel-ubuntu24.04 -> the AI Runtime base
#                     (runtime variant, EFA/RDMA pre-wired; AI Runtime requires it)
#     transformers    5.3.0 -> 5.5.3: vLLM 0.24.0 itself requires >=5.5.3 and verl
#                     v0.9.0 >=5.5.3,<5.11 -- upstream's pin violates both. (megatron-
#                     bridge 0.5.2 declares >=5.8.1; it is installed --no-deps.)
#     TE/apex/FA      source builds -> verl wheelhouse binaries (no nvcc in the base)
#     torch index     download.pytorch.org/whl/cu130 -> PyPI (the same cu130 build;
#                     the PyTorch index is unreachable from some corporate hosts)
#     mbridge         @main (floating) -> 641a5a0, patched (step 5a1)
#     megatron-bridge git r0.5.0 (a branch) -> the wheelhouse's 0.5.2 wheel
#     NCCL            forced >=2.29.7 -> torch's bundled NCCL (OVERRIDE_NCCL=0: keeps
#                     the aws-ofi-nccl EFA plugin loadable)
#     not installed   vllm-omni, trl, torchcodec (WITH_VIDEO=1), a cuDNN upgrade
#     added           flash-linear-attention (Qwen3.5 Gated-DeltaNet), cupy-cuda13x,
#                     nvidia-modelopt, a CUDA 13.2 toolchain alignment (5a2), opencv
#                     4.12 (FIPS, step 8), mlflow, bubblewrap, judge Ray 2.48 (5d)
#
# BUILD (amd64 — force the platform, Apple Silicon defaults to arm64):
#   make build      (then: make size push register -- or all of it: make image)
# =============================================================================
# Pinned by digest: the tag is mutable. (2026-09-23: still what the tag serves.)
FROM databricksruntime/air:dcs-base-aws-runtime-cu13@sha256:8da6cc766e745239ed46f573344a6cad351deb2c0caf3135b26e98f20e9197c0

# ---- version pins (all overridable via --build-arg) -------------------------
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG VLLM_VERSION=0.24.0
ARG TRANSFORMERS_VERSION=5.5.3
# verl, megatron-core, mbridge and the four native wheels are in docker/artifacts.lock,
# pinned by commit / sha256 -- not ARGs, so nothing can point them at a moving tag.
ARG MODELOPT_VERSION=0.44.0
ARG CUPY_VERSION=14.0.1
ARG OPENCV_VERSION=4.12.0.88
# The CUDA pip toolchain must be internally consistent: nvcc/cicc/ptxas/crt AND the
# cudart headers all at one MAJOR.MINOR (see step 5a2). 13.2.86 is the only fully
# coherent set on this index -- cuobjdump/nvdisasm have no 13.0 wheels here.
ARG CUDA_TOOLCHAIN_VERSION=13.2.86
# Baked so any job can prove which build it is running. A stale-image incident
# (same :v1 tag re-pushed, platform kept serving the old digest) cost a full debug
# cycle; `make smoke` now prints this, making staleness obvious in one line.
ARG IMAGE_TAG=dev
ARG TRANSFER_QUEUE_VERSION=0.1.8

# The package index is NOT a build ARG. The Makefile auto-detects a corporate index
# (scripts/detect_pypi_index.sh) and passes it as the BuildKit secret `pip_index`,
# which docker/uvi.sh hands to uv. An ARG that a RUN uses is recorded in the image
# history, credentials included. No secret -> public PyPI.

# TORCH_INDEX_URL is EMPTY by default -- torch comes from the same index as the rest.
#
# We deliberately do NOT use download.pytorch.org/whl/cu130. PyPI's own
# torch 2.11.0 is ALREADY the CUDA 13 build: its wheel metadata requires
# nvidia-cudnn-cu13 / nvidia-nccl-cu13 / nvidia-cusparselt-cu13 /
# nvidia-nvshmem-cu13 (verified against the index). So the plain PyPI wheel has
# exactly the cu130 ABI the wheelhouse binaries (TE 2.16.1, apex, flash-attn
# 2.8.3) were compiled against, and one fewer host has to be reachable.
#
# That matters on locked-down networks: download.pytorch.org is NOT routable from
# the Databricks corp boxes, while the PyPI proxy is. Set this only if you need a
# non-default CUDA variant.
ARG TORCH_INDEX_URL=

# Corporate-network plumbing. Declaring these as ARG is sufficient: Docker exposes
# build args as environment variables inside RUN, and uv/apt/curl all honour
# them. Deliberately NOT `ENV` -- that would bake a build-time proxy into the
# runtime image, where the training nodes have entirely different egress.
#   make build BUILD_ARGS="--build-arg HTTPS_PROXY=http://proxy:3128 --build-arg NO_PROXY=..."
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=

# Optional: override torch's bundled NCCL to >=2.29.7 for ncclCommSuspend/Resume
# (verl issue #6266). OFF by default, and on AWS that default matters more than
# it would on Azure.
#
# This base ships libnccl2 2.28.3+cuda13.0 and an aws-ofi-nccl plugin built
# against it; torch's cu130 wheel bundles its own NCCL of the same ~2.28 line, so
# out of the box the skew is small. Forcing >=2.29.7 widens it, and on AWS a
# plugin that declines to load is NOT benign: EFA reaches NCCL *through*
# aws-ofi-nccl, so losing the plugin means losing RDMA and silently falling back
# to TCP sockets. (Azure differs -- NCCL speaks IB verbs natively there.)
# Read NCCL_DEBUG=INFO from a real 2-node run (rung 4 sets it) and confirm
# NET/OFI before flipping this to 1.
ARG OVERRIDE_NCCL=0

# Optional: video decoding (torchcodec + ffmpeg, ~300 MB). geo3k is images-only,
# so it is off. Turn on if you move to a video dataset.
ARG WITH_VIDEO=0

# ---- env -------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
# FIPS: air hosts run a FIPS-enabled kernel. Non-FIPS crypto libs abort on
# SSL init ("FATAL FIPS SELFTEST FAILURE" / ssl.SSLError [CRYPTO]). The base
# image already sets OPENSSL_FORCE_FIPS_MODE=0; restated so it survives any
# future base change.
ENV OPENSSL_FORCE_FIPS_MODE=0 \
    OPENSSL_FIPS=0
# UV_NO_CACHE is not optional: uv's download cache is ~11 GB of wheels that are
# redundant with the installed venv, and baking it into a layer pushed a
# comparable image to 31 GB -> blew the 20 GB DCS limit.
ENV UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=1200 \
    UV_CONCURRENT_DOWNLOADS=4

# The index reaches uv as ENVIRONMENT (docker/uvi.sh exports it), not a per-command
# --index-url. WHY: `uv pip install --no-deps "verl @ git+..."` builds a source tree,
# and uv resolves its build-system.requires (setuptools, wheel) in a SEPARATE
# isolated resolution. A per-command --index-url on the outer install does not reach
# that inner resolution, so it fell back to pypi.org and died with
#   Failed to fetch: `https://pypi.org/simple/setuptools/` ... dns error
# even though the outer install was pointed at the internal proxy. The variables live
# only inside each RUN, so the runtime image never carries a build-time index.
# HF_HUB_ENABLE_HF_TRANSFER is deprecated (hf_transfer is no longer used) and
# huggingface_hub warns about it on every import. Xet is the current accelerated
# path; enable its high-performance mode instead. Safe here because in-container
# downloads land in HF_HOME=/tmp (local disk). Do NOT point Xet at a UC Volume --
# it does parallel range writes, which the FUSE mount rejects with EOPNOTSUPP.
ENV HF_XET_HIGH_PERFORMANCE=1
ENV VERL_ON_AIR_IMAGE_TAG=${IMAGE_TAG}
# Megatron-Bridge resolves HF<->mcore mappings lazily; keep tokenizers quiet.
ENV TOKENIZERS_PARALLELISM=false

# ---- retry helper (used by every network step; see docker/retry.sh) ---------
COPY docker/retry.sh /usr/local/bin/retry
# ---- uv with the build's index (see docker/uvi.sh) --------------------------
COPY docker/uvi.sh /usr/local/bin/uvi
RUN chmod +x /usr/local/bin/retry /usr/local/bin/uvi

# ---- the locks ---------------------------------------------------------------
# Every `uvi pip install` below is CONSTRAINED to docker/requirements.lock: a rebuild
# resolves exactly the versions the lock records, or fails. Both locks stay in the
# image (/etc/voa) as its own record; VOA_CONSTRAINTS (read only by docker/uvi.sh) is
# cleared at the end of the build, so the runtime environment is not affected.
COPY docker/requirements.lock docker/artifacts.lock /etc/voa/
ENV VOA_CONSTRAINTS=/etc/voa/requirements.lock

# =============================================================================
# 0. Corporate TLS interception
# =============================================================================
# uv links rustls with BUNDLED webpki roots and does NOT consult the system trust
# store. On a network with TLS inspection that yields, even though the host is
# perfectly happy:
#
#   Caused by: invalid peer certificate: UnknownIssuer
#
# ...for https://github.com/... while an internal index over untouched TLS works.
# Two things are therefore required:
#   1. the intercepting CA must be IN the image trust store  -> `make certs`
#      copies the host bundle into ./certs, COPYed below
#   2. uv must be told to use that store                     -> UV_SYSTEM_CERTS=1
#
# ./certs holds only a .gitkeep by default, so this is a no-op on open networks.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN set -eu; \
    found=0; \
    for f in /usr/local/share/ca-certificates/extra/*.crt \
             /usr/local/share/ca-certificates/extra/*.pem; do \
      [ -f "$f" ] || continue; \
      found=1; \
      cat "$f" >> /etc/ssl/certs/ca-certificates.crt; \
      echo "trusting extra CA bundle: $f ($(grep -c 'BEGIN CERTIFICATE' "$f") certs)"; \
    done; \
    if [ "$found" = 1 ]; then update-ca-certificates || true; \
    else echo "no extra CA bundles in ./certs (fine on an open network)"; fi

# Make uv/git/python use the system store rather than uv's bundled roots.
# UV_SYSTEM_CERTS is the current name; UV_NATIVE_TLS is deprecated and emits a
# warning on every invocation, so we do not set it. SSL_CERT_FILE is the fallback
# that older uv honours regardless.
ENV UV_SYSTEM_CERTS=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# =============================================================================
# 1. Toolchain + harness prerequisites
# =============================================================================
# The AI Runtime job harness itself imports pip and yaml before your command
# ever runs; on a bare base venv it dies in ~1 s with "No module named pip".
# build-essential is needed twice over: megatron-core's pybind11 ext at BUILD
# time, and Triton's launcher stub at RUN time (see the header note).
#
# python3-dev (unversioned) rather than python3.12-dev: this base sets
# UV_PYTHON_INSTALL_DIR=/opt/uv/python (true of the AWS and Azure bases alike),
# so /opt/venv's interpreter may be a uv-managed standalone build carrying its
# OWN headers rather than the system python3. In that case apt's headers are for
# the wrong interpreter and megatron-core fails with a confusing
# "Python.h: No such file". We therefore ASSERT that Python.h is reachable from
# the venv interpreter's own sysconfig, so the build fails here with a
# diagnostic instead of two layers later.
# NOTE the retries. This step used to have none, and it is exactly where a real
# build died: apt succeeded, then `uv pip install pybind11` failed with
#   error sending request for url (https://pypi.org/simple/pybind11/)
#   Caused by: dns error: failed to lookup address information
# One unlucky DNS lookup, no retry, whole build lost. `make doctor` now probes
# container DNS + every index this build needs before you start.
# bubblewrap (`bwrap`) is here for use cases whose tool EXECUTES model-generated code:
# running it inside `bwrap --unshare-all --ro-bind ...` means the snippet cannot read host
# files or reach the network. Neither shipped use case needs it (agentic-search searches,
# math evaluates a safe AST — see usecases/math/tool.py), but a code-execution tool is a
# common next step and the sandbox has to exist in the image to be usable. It needs
# unprivileged USER NAMESPACES permitted by the RUNTIME, which an image cannot grant --
# verify on your workspace before relying on it. Tiny (~200 KB); the `bwrap --version`
# check fails the build if apt silently drops it.
RUN --mount=type=secret,id=pip_index \
    retry 6 apt-get update \
    && retry 3 apt-get install -y --no-install-recommends \
        build-essential python3-dev git curl bubblewrap \
    && rm -rf /var/lib/apt/lists/* \
    && bwrap --version \
    && retry 6 uvi pip install --no-cache \
        pip setuptools wheel pybind11 ninja packaging pyyaml \
    && python3 - <<'PY'
import os, sys, sysconfig
inc = sysconfig.get_paths()["include"]
hdr = os.path.join(inc, "Python.h")
print(f"  interpreter : {sys.executable}")
print(f"  version     : {sys.version.split()[0]}")
print(f"  include dir : {inc}")
if not os.path.exists(hdr):
    sys.exit(
        f"\nPython.h NOT found at {hdr}\n"
        "megatron-core builds a pybind11 C++ extension and needs it.\n"
        "This base sets UV_PYTHON_INSTALL_DIR=/opt/uv/python, so /opt/venv's\n"
        "interpreter is likely a uv standalone build and apt's python3-dev\n"
        "installed headers for the SYSTEM python instead.\n"
        "Fix: install the matching headers for the interpreter above, or export\n"
        "CPPFLAGS=-I<that build's include/pythonX.Y> before the megatron-core step."
    )
print(f"  Python.h    : ok ({hdr})")
PY

# =============================================================================
# 2. torch 2.11.0 / cu130  (the ABI everything below is compiled against)
# =============================================================================
# (No quotes nested in ${...} here: the Dockerfile frontend then misses the heredoc.)
RUN --mount=type=secret,id=pip_index \
    if [ -n "${TORCH_INDEX_URL}" ]; then TORCH_IDX="--index-url ${TORCH_INDEX_URL}"; else TORCH_IDX=""; fi \
    && echo "installing torch from: ${TORCH_INDEX_URL:-the default index}" \
    && retry 6 uvi pip install ${TORCH_IDX} \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
    && python3 - <<'PY'
import sys, torch
cuda = torch.version.cuda or ""
print(f"torch {torch.__version__}  cuda {cuda}  "
      f"cxx11abi {torch._C._GLIBCXX_USE_CXX11_ABI}")
# HARD GATE. Every native wheel below (TransformerEngine 2.16.1, apex,
# flash-attn 2.8.3) is a cu130/torch-2.11 build. A CUDA 12 torch here would
# still import fine and then fail at runtime with
# "undefined symbol: _ZN3c105Error..." on a GPU node, hours later.
if not cuda.startswith("13"):
    sys.exit(
        f"FATAL: torch reports CUDA {cuda!r}, expected 13.x.\n"
        "The verl wheelhouse binaries are cu130-only, so a CUDA 12 torch makes\n"
        "this image unusable. Point TORCH_INDEX_URL at a cu130 index, e.g.\n"
        "  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
    )
PY

# =============================================================================
# 3. vLLM 0.24.0 rollout engine  (the version verl v0.9.0 pins; it requires
#    transformers>=5.5.3)
# =============================================================================
RUN --mount=type=secret,id=pip_index \
    retry 6 uvi pip install "vllm==${VLLM_VERSION}" \
 && python3 -c "import vllm; print('vllm', vllm.__version__)"

# =============================================================================
# 4. verl runtime dependencies
#    verl itself is installed --no-deps: its published metadata pins numpy<2 and
#    vllm<=0.12, both irreconcilable with the vllm 0.24 / Qwen3.5-era stack. We
#    supply the real deps explicitly, which is also what verl's own Dockerfile
#    effectively does.
# =============================================================================
RUN --mount=type=secret,id=pip_index \
    retry 6 uvi pip install \
        "torch==${TORCH_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "accelerate>=0.34,<2" "peft" \
        `# datasets>=4.0 deliberately, not >=3.0: paired with huggingface_hub 1.x` \
        `# a datasets 3.x resolution reproduces the observed` \
        `# "HfFileSystem.find() got multiple values for keyword argument maxdepth"` \
        "datasets>=4.0,<6" \
        "hydra-core" "omegaconf" "einops" "ninja" "codetiming" "dill" \
        "pylatexenc" "torchdata" "pyarrow>=19.0.0" "pandas" \
        `# ray: exactly what verl ran on here (a multi-node judge gets its own, 5d)` \
        "ray[default]==2.58.0" \
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
        "mathruler" "qwen-vl-utils" "cachetools" "nvtx" \
        "huggingface_hub>=0.35" "hf_transfer" \
        `# nvcc + CUDA headers for RUNTIME JIT (see the CUDA_HOME step below).` \
        `# They arrive transitively via vllm today; named explicitly so a future` \
        `# resolution cannot silently drop them and break FlashInfer's GDN JIT.` \
        `#` \
        `# NOTE the UNSUFFIXED names. In the CUDA 13 era NVIDIA unified these, and` \
        `# the -cu13 variants are DEPRECATION TOMBSTONES: the index still serves a` \
        `# page for them (HTTP 200 means nothing), but the only artefact is a 0.0.1` \
        `# sdist whose build backend deliberately fails with` \
        `#   "THIS PROJECT nvidia-cuda-crt-cu13 IS DEPRECATED. Use nvidia-cuda-crt".` \
        `# Left UNPINNED so the resolver keeps whatever vllm/torch require, rather` \
        `# than fighting them; the real guarantee is the nvcc assertion in step 5b.` \
        "nvidia-cuda-nvcc" "nvidia-cuda-runtime" "nvidia-cuda-crt" \
        "fastapi" "uvicorn" "matplotlib" \
        "wandb" "tensorboard" "mlflow>=3.6,<=3.15.0"
# =============================================================================
# 5. Megatron stack — prebuilt binaries from the verl wheelhouse
#    --no-deps throughout: these wheels carry pins (or, for megatron-bridge,
#    metadata verl deliberately discards) that would drag the resolver off the
#    torch 2.11 ABI. Their real deps are already installed above.
# =============================================================================
# The four wheels in docker/artifacts.lock, each checked against its sha256 before
# it is installed. `make vendor` pre-fetches them on the HOST (which trusts an
# intercepting CA) into ./vendor/wheels, verified the same way; if present, the
# local file is used and github.com is never touched. vendor/ is BIND-MOUNTED, not
# COPYed: a COPY would keep ~400 MB of wheels in a layer of the shipped image.
RUN --mount=type=bind,source=vendor,target=/vendor \
    --mount=type=secret,id=pip_index \
    set -eu; \
    mkdir -p /tmp/wheels; \
    grep -E '^wheel[[:space:]]' /etc/voa/artifacts.lock > /tmp/wheels.txt; \
    while read -r _kind name version url sha; do \
      f="/tmp/wheels/$(basename "${url}")"; \
      if [ -f "/vendor/wheels/$(basename "${url}")" ]; then \
        cp "/vendor/wheels/$(basename "${url}")" "${f}"; src="vendor/wheels"; \
      else \
        retry 6 curl -fsSL -o "${f}" "${url}"; src="${url}"; \
      fi; \
      got="$(sha256sum "${f}" | cut -d' ' -f1)"; \
      if [ "${got}" != "${sha}" ]; then \
        echo "FATAL: ${name} ${version} from ${src}: sha256 ${got}, docker/artifacts.lock pins ${sha}"; exit 1; \
      fi; \
      echo "==> ${name} ${version}: sha256 verified (${src})"; \
    done < /tmp/wheels.txt; \
    retry 3 uvi pip install --no-deps /tmp/wheels/*.whl; \
    rm -rf /tmp/wheels /tmp/wheels.txt \
 && retry 6 uvi pip install \
      "nvidia-modelopt==${MODELOPT_VERSION}" \
      "nvidia-mathdx" "onnxscript" "pydantic" \
 && python3 -c "import transformer_engine, flash_attn, apex; print('TE/FA/apex OK')"

# The three source installs in docker/artifacts.lock, at their pinned COMMITS (a tag
# or branch can move): megatron-core (pybind11 C++ ext, no CUDA toolchain); mbridge,
# the legacy HF<->mcore connector still required by MEGATRON_MODE=classic (verl's
# tested Qwen3.5-35B path uses vanilla_mbridge=True); and verl itself, --no-deps: its
# published metadata pins numpy<2 and vllm<=0.12, both irreconcilable with this stack,
# so its real dependencies are the ones installed above. `make vendor` checks the
# three out on the host and records each tree's commit in .voa-commit; a tree at any
# other commit stops the build. (rw: a build writes into its source tree; the writes
# are discarded with the mount.) --no-build-isolation: setuptools + wheel are already
# installed (step 1).
RUN --mount=type=bind,source=vendor,target=/vendor,rw \
    --mount=type=secret,id=pip_index \
    set -eu; \
    grep -E '^git[[:space:]]' /etc/voa/artifacts.lock > /tmp/git.txt; \
    while read -r _kind name version repo commit; do \
      if [ -d "/vendor/src/${name}" ]; then \
        have="$(cat "/vendor/src/${name}/.voa-commit" 2>/dev/null || echo unknown)"; \
        if [ "${have}" != "${commit}" ]; then \
          echo "FATAL: vendor/src/${name} is at ${have}, docker/artifacts.lock pins ${commit} -- re-run make vendor"; exit 1; \
        fi; \
        echo "==> ${name} ${version}: vendored tree at ${commit}"; \
        retry 3 uvi pip install --no-deps --no-build-isolation "/vendor/src/${name}"; \
      else \
        echo "==> ${name} ${version}: ${repo} at ${commit}"; \
        retry 6 uvi pip install --no-deps --no-build-isolation "${name} @ git+${repo}@${commit}"; \
      fi; \
    done < /tmp/git.txt; \
    rm -f /tmp/git.txt \
 && python3 -c "import importlib.metadata as m; print({p: m.version(p) for p in ('verl', 'megatron-core', 'mbridge')})"

# TransferQueue: verl's optional data-transfer backend, pinned as upstream does. A failed
# install fails the build (it used to end in `|| true`), and step 9 imports it.
RUN --mount=type=secret,id=pip_index \
    retry 3 uvi pip install --no-deps "TransferQueue==${TRANSFER_QUEUE_VERSION}"

# =============================================================================
# 5a1. Patch the mbridge <-> megatron-core version skew (classic / legacy path)
# =============================================================================
# mbridge (pinned at 641a5a0, docker/artifacts.lock) hard-codes
# async_tensor_model_parallel_allreduce=True into
# Qwen3_5VLTransformerConfig(TransformerConfig), but megatron-core core_v0.18.0
# DROPPED that field. So the legacy-mbridge path (MEGATRON_MODE=classic,
# vanilla_mbridge=True -- what verl's tested Qwen3.5-35B script uses) dies at
# model init on every worker with:
#   TypeError: Qwen3_5VLTransformerConfig.__init__() got an unexpected keyword
#              argument 'async_tensor_model_parallel_allreduce'
# fsdp (rung 1/2) escaped it: vanilla_mbridge=False routes through the
# megatron.bridge providers, which RE-DECLARE the field as their own. classic
# (rung 3) is the first run down the legacy path. The flag toggled an async TP
# all-reduce this megatron-core no longer exposes as a config knob, so removing
# the pass-through simply restores the current Megatron default. It is the ONLY
# skewed kwarg -- the AST audit in step 9 checks all 46 kwargs both Qwen3.5
# bridges forward against the 287 config fields and flags exactly this one.
#
# Scoped to the mbridge tree, and the regex matches only the `=True,` keyword-arg
# form, so it cannot touch the valid megatron.bridge field defs or verl's own
# `"...": False,` dict entry. The `test -n` guard FAILS the build if the kwarg is
# absent -- so a future mbridge bump that fixes this upstream forces us to
# drop this now-dead patch instead of silently masking a changed surface.
RUN set -eu; \
    MB="$(python3 -c 'import os, mbridge; print(os.path.dirname(mbridge.__file__))')"; \
    echo "mbridge tree: ${MB}"; \
    RE='^[[:space:]]*async_tensor_model_parallel_allreduce=True,[[:space:]]*$'; \
    hits="$(grep -rlE "${RE}" "${MB}" || true)"; \
    test -n "${hits}" || { echo "FATAL: dead kwarg not present in mbridge -- re-audit before shipping (did the mbridge pin in docker/artifacts.lock change?)"; exit 1; }; \
    echo "patching:"; printf '%s\n' "${hits}" | sed 's/^/  /'; \
    printf '%s\n' "${hits}" | xargs sed -i -E "/${RE}/d"; \
    find "${MB}" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true; \
    left="$(grep -rE "${RE}" "${MB}" | wc -l)"; \
    test "${left}" -eq 0 || { echo "FATAL: ${left} occurrence(s) remain after patch"; exit 1; }; \
    echo "mbridge patched: async_tensor_model_parallel_allreduce pass-through removed"

# cupy: NOT optional for this model. Megatron's MoE all-to-all dispatcher uses it
# directly, and ray.util.collective's NCCL backend is cupy-based (every verl
# weight-sync engine imports it). Without it the checkpoint engine never
# registers and rollout weight sync fails at step 1.
RUN --mount=type=secret,id=pip_index \
    retry 6 uvi pip install "cupy-cuda13x==${CUPY_VERSION}"

# flash-linear-attention: Triton kernels for Qwen3.5's Gated-DeltaNet layers
# (30 of its 40 layers are linear attention). Pure-python wheel.
RUN --mount=type=secret,id=pip_index \
    retry 6 uvi pip install "flash-linear-attention==0.5.2"

# =============================================================================
# 5a2. Align nvcc's version with the CUDA *headers* (CCCL enforces this)
# =============================================================================
# With nvcc merely made discoverable, rung 1 got further -- exit 127 became exit 1 --
# and failed inside FlashInfer's GDN build with:
#   cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:
#     error "CUDA compiler and CUDA toolkit headers are incompatible,
#            please check your include paths"
#
# CCCL compares the compiler against the toolkit headers, roughly:
#   CUDA_VERSION  !=  __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10
# so only MAJOR.MINOR has to agree; the patch level is free.
#
# The transitive resolution produced a skewed set (measured, not guessed):
#   nvidia-cuda-nvcc      13.2.86   -> __CUDACC_VER_* => 13020
#   nvidia-cuda-runtime   13.0.96   -> CUDA_VERSION   => 13000   MISMATCH
#   nvidia-cuda-crt       13.3.73
#
# FIRST ATTEMPT, WRONG DIRECTION: pinning nvcc DOWN to 13.0.88 fixed the CCCL
# assert and then broke one layer deeper, in ptxas:
#   ptxas fatal: Unsupported .version 9.2; current version is '9.0'
# PTX ISA 9.2 is CUDA 13.2, 9.0 is CUDA 13.0 -- so nvcc's NVVM frontend (cicc) was
# still 13.2 while ptxas had moved to 13.0. These CUDA pip packages are modular and
# must be aligned as a SET, and there is no separate `nvidia-cuda-nvvm` to pin
# (404 on the index). Worse, this index carries NO 13.0 wheels at all for
# nvidia-cuda-cuobjdump / -nvdisasm, so a fully-13.0 toolchain is not even
# constructible here.
#
# So align UP to 13.2.86 instead -- the version nvcc, crt and nvvm already were.
# Only the HEADERS move (nvidia-cuda-runtime 13.0.96 -> 13.2.86), which is what
# CCCL actually reads, and nothing gets downgraded. CUDA minor-version
# compatibility covers running torch-built-for-13.0 against a 13.2 cudart, and the
# driver (580.126.16) supports all of CUDA 13.x.
#
# --no-deps: this is a deliberate, surgical version alignment. Without it uv would
# try to re-resolve torch, which declares its own cudart expectation.
#
# Placed AFTER every other pip install so nothing can silently re-upgrade nvcc.
RUN --mount=type=secret,id=pip_index \
    retry 6 uvi pip install --no-deps \
        "nvidia-cuda-nvcc==${CUDA_TOOLCHAIN_VERSION}" \
        "nvidia-cuda-crt==${CUDA_TOOLCHAIN_VERSION}" \
        "nvidia-cuda-runtime==${CUDA_TOOLCHAIN_VERSION}" \
        "nvidia-cuda-nvrtc==${CUDA_TOOLCHAIN_VERSION}" \
 && python3 - <<'PY'
import re, sysconfig, os, subprocess
sp = sysconfig.get_paths()["purelib"]
nvcc = os.path.join(sp, "nvidia", "cu13", "bin", "nvcc")
out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
cver = re.search(r"release (\d+)\.(\d+)", out)
assert cver, out
cc = int(cver.group(1)) * 1000 + int(cver.group(2)) * 10
hdr = os.path.join(sp, "nvidia", "cu13", "include", "cuda_runtime_api.h")
src = open(hdr).read() if os.path.exists(hdr) else ""
m = re.search(r"#define\s+CUDA_VERSION\s+(\d+)", src)
tk = int(m.group(1)) if m else None
print(f"nvcc release {cver.group(1)}.{cver.group(2)} -> __CUDACC_VER == {cc}")
print(f"headers CUDA_VERSION == {tk}")
if tk is not None and tk != cc:
    raise SystemExit(f"FATAL: CCCL will reject this: {cc} != {tk}")
print("nvcc and CUDA headers agree")
PY

# =============================================================================
# 5b. Make /usr/local/cuda a COMPLETE toolchain (runtime CUDA JIT)
# =============================================================================
# rung 1 died in vLLM with:
#   ninja ... cached_ops/gdn_prefill_sm90 ... returned non-zero exit status 127
# 127 = command not found. vLLM's FlashInfer JIT-compiles the Qwen3.5
# Gated-DeltaNet prefill kernel AT RUNTIME, and needs nvcc to do it.
#
# The mistake was reasoning "we install only prebuilt wheels, so nothing CUDA
# compiles" -- true at BUILD time, false at RUN time. Same lesson as Triton
# needing a C compiler at runtime, which we did account for.
#
# Measured on a real node (infra/diagnostics/diag_cuda.py):
#   nvcc IS present at  <site-packages>/nvidia/cu13/bin/nvcc  (13.2.86, runs fine)
#   /usr/local/cuda/bin is MISSING; /usr/local/cuda/include holds only nvtx3
#   CUDA_HOME=/usr/local/cuda therefore points at an incomplete tree
#
# So nothing needs installing -- the toolchain just has to be discoverable at the
# conventional location. Symlink the pip tree's bin/ and nvvm/ into
# /usr/local/cuda and graft its headers into the existing include dir. Keeps
# CUDA_HOME=/usr/local/cuda valid (what every tool expects) and keeps the base
# image's lib64 intact.
#
# The header graft must SKIP entries the base already provides. The base ships
# include/nvtx3 as a real DIRECTORY, and the pip tree has one too; a bare
# `ln -sfn "$f" "$INC/"` then dies with
#   ln: /usr/local/cuda/targets/x86_64-linux/include/nvtx3: cannot overwrite directory
# Skipping conflicts is also the safer merge order: the base image's own headers
# win, and we only add what is missing (cuda_runtime.h and friends).
RUN set -eu; \
    CU="$(python3 -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13"))')"; \
    echo "pip CUDA tree: ${CU}"; \
    test -x "${CU}/bin/nvcc" || { echo "FATAL: no nvcc at ${CU}/bin/nvcc"; exit 1; }; \
    ln -sfn "${CU}/bin"  /usr/local/cuda/bin; \
    ln -sfn "${CU}/nvvm" /usr/local/cuda/nvvm; \
    INC=/usr/local/cuda/targets/x86_64-linux/include; \
    mkdir -p "${INC}"; \
    linked=0; kept=0; \
    for f in "${CU}"/include/*; do \
      b="$(basename "$f")"; \
      if [ -e "${INC}/${b}" ] || [ -L "${INC}/${b}" ]; then \
        kept=$((kept+1)); continue; \
      fi; \
      ln -sfn "$f" "${INC}/${b}"; \
      linked=$((linked+1)); \
    done; \
    echo "headers: ${linked} linked, ${kept} already provided by the base image"; \
    /usr/local/cuda/bin/nvcc --version | tail -2; \
    test -f "${INC}/cuda_runtime.h" || { echo "FATAL: cuda_runtime.h not reachable at ${INC}"; exit 1; }; \
    echo "cuda_runtime.h OK at ${INC}"

# =============================================================================
# 5b2. Expose libnvrtc.so on the linker path (runtime CUDA JIT *link* step)
# =============================================================================
# Section 5b symlinks the pip CUDA tree's bin/nvvm/include so nvcc can COMPILE at
# runtime, but not its lib/. FlashInfer JIT-compiles kernels at runtime (e.g. the
# fp8_blockscale_gemm FP8 MoE GEMM for GLM-5.3/GlmMoeDsa) and the final LINK runs:
#   c++ ... -L/usr/local/cuda/lib64 -lcudart -lcuda -lnvrtc ...
# libnvrtc.so.* IS already installed (nvidia-cuda-nvrtc, step above) but lives in
# the pip tree, not on the -L path, so the link dies with:
#   /usr/bin/ld: cannot find -lnvrtc  (air/52c run3, GLM-5.3 TP16 judge serve)
# Fix = symlink the existing lib into lib64 (0 bytes added) + refresh ldconfig so
# the runtime loader also resolves the SONAME. Same class of bug as 5b's nvcc gap.
RUN set -eu; \
    NVDIR="$(python3 -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "nvidia"))')"; \
    nvrtc_real="$(find "${NVDIR}" -name 'libnvrtc.so.*' 2>/dev/null | sort -V | tail -1)"; \
    test -n "${nvrtc_real}" || { echo "FATAL: libnvrtc.so.* not found under ${NVDIR}"; find "${NVDIR}" -name 'libnvrtc*' 2>/dev/null | head; exit 1; }; \
    SRC="$(dirname "${nvrtc_real}")"; LIB64=/usr/local/cuda/lib64; mkdir -p "${LIB64}"; \
    for so in "${SRC}"/libnvrtc.so* "${SRC}"/libnvrtc-builtins.so*; do \
      [ -e "$so" ] && ln -sfn "$so" "${LIB64}/$(basename "$so")"; \
    done; \
    ln -sfn "${nvrtc_real}" "${LIB64}/libnvrtc.so"; \
    { grep -qxF "${LIB64}" /etc/ld.so.conf.d/*.conf 2>/dev/null; } || echo "${LIB64}" > /etc/ld.so.conf.d/zz-cuda-lib64.conf; \
    ldconfig 2>/dev/null || true; \
    echo "nvrtc: linked $(basename "${nvrtc_real}") into ${LIB64}"; ls -la "${LIB64}"/libnvrtc*; \
    test -e "${LIB64}/libnvrtc.so" || { echo "FATAL: ${LIB64}/libnvrtc.so missing after link"; exit 1; }

# =============================================================================
# 5c. Prove nvcc compiles what FlashInfer ACTUALLY compiles
# =============================================================================
# A kernel including only cuda_runtime.h compiles fine even with a skewed
# toolchain -- which is precisely how the nvcc-13.2-vs-headers-13.0 mismatch got
# past the first version of this check and only surfaced on 8xH100. The version
# assert lives in CCCL headers, so the probe has to pull those in, with
# FlashInfer's own include paths and its real -gencode.
#
# NOTE: no backtick-comments in this RUN. `# ...` is only safe in ARGUMENT
# position; in COMMAND position the empty expansion takes the command-name slot
# and the next assignment is parsed as a command:
#   /bin/sh: 1: CCCL=/opt/venv/.../flashinfer/data/cccl: not found
# scripts/lint_dockerfile.py now rejects that pattern.
COPY docker/cccl_probe.cu /tmp/cccl_probe.cu
RUN set -eu; \
    CCCL="$(python3 -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "flashinfer", "data", "cccl"))')"; \
    if [ ! -d "${CCCL}/libcudacxx/include" ]; then \
      echo "FATAL: flashinfer CCCL headers absent at ${CCCL} -- cannot verify the GDN JIT path"; \
      exit 1; \
    fi; \
    if ! /usr/local/cuda/bin/nvcc -c /tmp/cccl_probe.cu -o /tmp/cccl_probe.o \
      -std=c++20 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr \
      -I"${CCCL}/libcudacxx/include" -I"${CCCL}/cub" -I"${CCCL}/thrust" \
      -isystem /usr/local/cuda/include; then \
      echo "=================== TOOLCHAIN DIAGNOSTICS ==================="; \
      echo "-- nvcc:";   /usr/local/cuda/bin/nvcc --version | tail -2 || true; \
      echo "-- ptxas:";  /usr/local/cuda/bin/ptxas --version | tail -2 || true; \
      echo "-- nvvm/cicc:"; ls -la /usr/local/cuda/nvvm/bin/ 2>&1 | head -5 || true; \
      echo "-- installed nvidia-cuda-* versions:"; \
      uv pip list 2>/dev/null | grep -i "^nvidia-cuda" || true; \
      echo "-- CUDA_VERSION in the grafted headers:"; \
      grep -h "define CUDA_VERSION" /usr/local/cuda/include/cuda_runtime_api.h 2>/dev/null || true; \
      echo "============================================================"; \
      echo "FATAL: nvcc cannot compile with CCCL headers -- FlashInfer's GDN JIT will fail at runtime"; \
      exit 1; \
    fi; \
    rm -f /tmp/cccl_probe.o; \
    echo "CCCL probe OK: sm_90a + CCCL + cuda_runtime.h (cicc AND ptxas both ran)"

# nvcc on PATH for every runtime JIT (FlashInfer, and anything else that shells out).
ENV PATH=/usr/local/cuda/bin:${PATH}

# =============================================================================
# 6. The judge's own Ray  (verl itself is installed in step 5, from artifacts.lock)
# =============================================================================
# vLLM 0.24's Ray executor breaks on ray>=2.55 (vllm#45318: ActorHandleNotFoundError,
# "not valid across Ray sessions", at EngineCore init), while verl needs the venv's
# ray 2.58. A MULTI-NODE judge runs ONLY vLLM, so serve_judge.sh puts this tree first
# on PYTHONPATH -- the ray CLI, vLLM and every Ray worker then load 2.48. Baked here in
# place of the `pip install ray==...` the judge used to run at start-up, which mutated
# the environment and needed PyPI from every judge node. (--no-deps: ray's own
# dependencies are the venv's, exactly as that start-up install had them.)
ARG JUDGE_RAY_VERSION=2.48.0
RUN --mount=type=secret,id=pip_index \
    VOA_CONSTRAINTS= retry 6 uvi pip install --no-deps --target /opt/judge-ray "ray==${JUDGE_RAY_VERSION}" \
 && PYTHONPATH=/opt/judge-ray python3 -c "import ray; assert ray.__version__ == '${JUDGE_RAY_VERSION}', ray.__version__; print('judge ray', ray.__version__, ray.__file__)"
ENV JUDGE_RAY_PATH=/opt/judge-ray \
    JUDGE_RAY_VERSION=${JUDGE_RAY_VERSION}

# =============================================================================
# 7. Optional bits
# =============================================================================
# torchcodec is not in docker/requirements.lock (v8 had none), so it resolves
# unconstrained: a WITH_VIDEO=1 image is a different, unlocked build.
RUN --mount=type=secret,id=pip_index \
    if [ "${WITH_VIDEO}" = "1" ]; then \
      retry 6 apt-get update && retry 3 apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/* \
      && if [ -n "${TORCH_INDEX_URL}" ]; then TORCH_IDX="--index-url ${TORCH_INDEX_URL}"; else TORCH_IDX=""; fi \
      && retry 6 uvi pip install ${TORCH_IDX} torchcodec; \
    else echo "WITH_VIDEO=0 -> skipping torchcodec/ffmpeg (geo3k is images-only)"; fi

RUN --mount=type=secret,id=pip_index \
    if [ "${OVERRIDE_NCCL}" = "1" ]; then \
      VOA_CONSTRAINTS= retry 6 uvi pip install --no-deps --upgrade "nvidia-nccl-cu13>=2.29.7,<3.0" \
      && echo "NCCL overridden - confirm EFA still binds (NET/OFI) via NCCL_DEBUG=INFO"; \
    else echo "OVERRIDE_NCCL=0 -> keeping torch's bundled NCCL (EFA-safe default)"; fi

# =============================================================================
# 8. opencv pin — MUST BE THE LAST pip operation
#    vllm pulls opencv-python-headless 5.x, whose *bundled* libcrypto is
#    FIPS-enforcing and hard-aborts the process on `import cv2`
#    ("crypto/fips/fips.c: FATAL FIPS SELFTEST FAILURE"). transformers imports
#    cv2 transitively via mistral_common, so every Ray worker dies. It is a
#    vendored blob, so OPENSSL_* env vars cannot save you. 4.12.0.88 ships
#    FIPS-free bundled libs. Uninstall first to force removal of the 5.x .libs
#    directory, and keep this last so nothing reinstalls 5.x afterwards.
# =============================================================================
RUN --mount=type=secret,id=pip_index \
    uv pip uninstall opencv-python-headless opencv-python 2>/dev/null || true; \
    retry 6 uvi pip install "opencv-python-headless==${OPENCV_VERSION}" \
 && python3 -c "import cv2; print('cv2', cv2.__version__, 'FIPS-clean')"

# =============================================================================
# 9. Build-time verification
# =============================================================================
# The image carries NO repository code: every job ships engine/ + its use case as a
# code_source snapshot, so the image is the dependency set alone -- a code change
# never makes it stale, and what ran is identified by the snapshot's commit (GIT_SHA).
ENV VOA_CONSTRAINTS=

# Its own resolved package set, for the record and for the lock check below.
RUN uv pip freeze > /etc/voa/freeze.txt && wc -l /etc/voa/freeze.txt

# Fail the build here rather than 6 minutes into `air register image` plus a
# 16-GPU job launch. Imports only; no GPU present during docker build.
RUN python3 - <<'PY'
import importlib, sys
mods = ["torch", "vllm", "transformers", "verl", "megatron.core", "megatron.bridge",
        "transformer_engine", "apex", "flash_attn", "fla", "mbridge", "cupy",
        "cv2", "ray", "mlflow", "mathruler", "qwen_vl_utils", "yaml", "pip", "transfer_queue"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        bad.append((m, e)); print(f"  FAIL {m}: {type(e).__name__}: {e}")
# Qwen3.5-35B-A3B is Qwen3_5MoeForConditionalGeneration; if this class is absent
# the transformers pin is wrong and nothing downstream can work.
try:
    from transformers import Qwen3_5MoeForConditionalGeneration  # noqa: F401
    print("  ok   transformers.Qwen3_5MoeForConditionalGeneration")
except Exception as e:
    bad.append(("Qwen3_5MoeForConditionalGeneration", e))
    print(f"  FAIL Qwen3_5Moe...: {e}")
# mbridge<->megatron-core config-skew gate. Assert the Qwen3.5 bridges forward NO
# kwarg that Qwen3_5VLTransformerConfig (a megatron-core TransformerConfig)
# rejects -- catches the async_tensor_model_parallel_allreduce regression (and any
# future skew) HERE, at build time, instead of at model init on a GPU node 175 s
# into a run. Pairs with the surgical sed in step 5a1.
try:
    import ast, inspect, dataclasses
    from mbridge.models.qwen3_5.transformer_config import Qwen3_5VLTransformerConfig as _Cfg
    import mbridge.models.qwen3_5.qwen3_5_vl_bridge as _Br
    _valid = {f.name for f in dataclasses.fields(_Cfg)}
    _consumed = set(inspect.signature(_Br.Qwen3_5VlBridge._build_base_config).parameters) - {"self"}
    _passed = set()
    for _n in ast.walk(ast.parse(inspect.getsource(_Br))):
        if isinstance(_n, ast.Call) and isinstance(getattr(_n, "func", None), ast.Attribute) \
                and _n.func.attr == "_build_base_config":
            _passed |= {k.arg for k in _n.keywords if k.arg}
    _skew = sorted((_passed - _consumed) - _valid)
    if _skew:
        bad.append(("mbridge_config_skew", f"bridge forwards kwargs the config rejects: {_skew}"))
        print(f"  FAIL mbridge config skew: {_skew}")
    else:
        print("  ok   mbridge<->megatron config audit (no skewed kwargs)")
except Exception as e:
    bad.append(("mbridge_config_audit", e))
    print(f"  FAIL mbridge config audit: {type(e).__name__}: {e}")
# The installed set IS the lock: every package is constrained to exactly its locked
# version (docker/requirements.lock), installed from docker/artifacts.lock, or one of
# the hand-aligned exceptions the lock's header names. A package the lock does not
# know means the lock no longer describes this image.
import re
def _norm(n): return re.sub(r"[-_.]+", "-", n).lower()
locked = {}
for ln in open("/etc/voa/requirements.lock"):
    ln = ln.strip()
    if ln and not ln.startswith("#"):
        n, v = ln.split("==", 1); locked[_norm(n)] = v
artefacts = {_norm(ln.split()[1]) for ln in open("/etc/voa/artifacts.lock")
             if ln.split() and ln.split()[0] in ("wheel", "git")}
hand = {"nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc",
        "opencv-python-headless", "torchcodec"}
drift = []
for ln in open("/etc/voa/freeze.txt"):
    ln = ln.strip()
    if not ln:
        continue
    n = _norm(re.split(r"==| @ ", ln)[0])
    if n in artefacts or n in hand:
        continue
    v = ln.split("==", 1)[1] if "==" in ln else ln
    if locked.get(n) != v:
        drift.append(f"{ln} (lock: {locked.get(n, 'absent')})")
if drift:
    bad.append(("lock_drift", drift))
    print("  FAIL the installed set departs from docker/requirements.lock:")
    for d in drift[:40]:
        print(f"         {d}")
else:
    print(f"  ok   installed set == docker/requirements.lock ({len(locked)} pins) + artifacts.lock")
if bad:
    sys.exit(f"image verification failed: {[m for m, _ in bad]}")
print("image verification passed")
PY
