# syntax=docker/dockerfile:1.7
# ^ REQUIRED, and must be the first line: the self-verification step at the end
#   uses a `RUN ... <<'PY'` heredoc, which needs BuildKit frontend >= 1.4.
#   Without this directive an older default frontend fails with
#   "Dockerfile parse error: unknown instruction: IMPORTLIB".
#
# =============================================================================
# verl + Megatron/mcore (Megatron-Bridge + Megatron-FSDP) + vLLM rollout
# for Qwen3.5-35B-A3B GRPO on Databricks AI Runtime serverless GPU.
#
# WHY THIS BASE  (AWS / df1)
#   df1 = dbc-559ffd80-2bfc.cloud.databricks.com -> AWS. That matters twice over:
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
# WHY WE DON'T NEED nvcc
#   verl publishes a prebuilt wheelhouse of cu130 / torch-2.11 / cp312 native
#   wheels: https://verl-project.github.io/verl-wheelhouse/simple/
#   TransformerEngine, apex and flash-attn all come from there as binaries, so
#   nothing CUDA compiles at build time -- which is why the runtime base suffices
#   even though it has no nvcc. We install them by DIRECT WHEEL URL rather than
#   via --extra-index-url on purpose: PyPI has an unrelated package literally
#   named `apex`, and an index-resolution race would silently install the wrong
#   one. Direct URLs are deterministic.
#
# WHY build-essential STAYS IN THE IMAGE
#   Two reasons, both runtime not build-time:
#     1. megatron-core is a source install (one pybind11 C++ extension, no CUDA).
#     2. Qwen3.5's Gated-DeltaNet path runs *Triton* kernels, and Triton JITs a
#        C launcher stub at runtime -> it needs a host C compiler on PATH.
#   Purging the toolchain saves ~400 MB and then dies hours into the first run.
#
# VERSION SET
#   Taken verbatim from verl v0.9.0's own docker/Dockerfile.stable.vllm, which
#   pins a mutually-tested combination. Do not bump one of these alone.
#     torch 2.11.0/cu130 · vllm 0.24.0 · transformers 5.5.3
#     TransformerEngine 2.16.1 · flash-attn 2.8.3 · apex master
#     megatron-core core_v0.18.0 · megatron-bridge 0.5.2 (r0.5.0) · mbridge 641a5a0
#
# BUILD (amd64 — force the platform, Apple Silicon defaults to arm64):
#   make build push register
# or:
#   docker build --platform linux/amd64 -f docker/Dockerfile \
#     -t mshtelma/verl-megatron-air:v1 .
# =============================================================================
FROM databricksruntime/air:dcs-base-aws-runtime-cu13

# ---- version pins (all overridable via --build-arg) -------------------------
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG VLLM_VERSION=0.24.0
ARG TRANSFORMERS_VERSION=5.5.3
ARG VERL_REF=v0.9.0
ARG MCORE_VERSION=core_v0.18.0
ARG MBRIDGE_REV=641a5a0
ARG MODELOPT_VERSION=0.44.0
ARG CUPY_VERSION=14.0.1
ARG OPENCV_VERSION=4.12.0.88
ARG TRANSFER_QUEUE_VERSION=0.1.8

# PIP_INDEX_URL: the Makefile auto-detects a corporate index from the local pip
# config (`pip config get global.index-url`) and passes it through, so a box
# behind e.g. https://pypi-proxy.dev.databricks.com/simple works with no edits.
# Falls back to public PyPI.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=

# TORCH_INDEX_URL is EMPTY by default -- torch comes from PIP_INDEX_URL.
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

# Prebuilt native wheels from the verl wheelhouse, pinned by direct URL.
ARG WH=https://github.com/verl-project/verl-wheelhouse/releases/download
ARG TE_WHL=${WH}/transformer-engine-v2.16.1/transformer_engine-2.16.1-cp312-cp312-linux_x86_64.whl
ARG APEX_WHL=${WH}/apex-master/apex-0.1-cp312-cp312-linux_x86_64.whl
ARG FA_WHL=${WH}/flash-attention-v2.8.3/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
ARG MBRIDGE_NV_WHL=${WH}/megatron-bridge-r0.5.0/megatron_bridge-0.5.2-py3-none-any.whl

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
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Megatron-Bridge resolves HF<->mcore mappings lazily; keep tokenizers quiet.
ENV TOKENIZERS_PARALLELISM=false

# ---- retry helper (used by every network step; see docker/retry.sh) ---------
COPY docker/retry.sh /usr/local/bin/retry
RUN chmod +x /usr/local/bin/retry

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
#   2. uv must be told to use that store                     -> UV_NATIVE_TLS=1
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
ENV UV_NATIVE_TLS=1 \
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
RUN retry 6 apt-get update \
    && retry 3 apt-get install -y --no-install-recommends \
        build-essential python3-dev git curl \
    && rm -rf /var/lib/apt/lists/* \
    && retry 6 uv pip install --no-cache --index-url ${PIP_INDEX_URL} \
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
RUN IDX="${TORCH_INDEX_URL:-${PIP_INDEX_URL}}" \
    && echo "installing torch from: ${IDX}" \
    && retry 6 uv pip install --index-url "${IDX}" \
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
# 3. vLLM 0.24.0 rollout engine  (first vLLM with Qwen3.5 support)
# =============================================================================
RUN retry 6 uv pip install --index-url ${PIP_INDEX_URL} "vllm==${VLLM_VERSION}" \
 && python3 -c "import vllm; print('vllm', vllm.__version__)"

# =============================================================================
# 4. verl runtime dependencies
#    verl itself is installed --no-deps: its published metadata pins numpy<2 and
#    vllm<=0.12, both irreconcilable with the vllm 0.24 / Qwen3.5-era stack. We
#    supply the real deps explicitly, which is also what verl's own Dockerfile
#    effectively does.
# =============================================================================
RUN retry 6 uv pip install --index-url ${PIP_INDEX_URL} \
        "torch==${TORCH_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "accelerate>=0.34,<2" "peft" \
        `# datasets>=4.0 deliberately, not >=3.0: paired with huggingface_hub 1.x` \
        `# a datasets 3.x resolution reproduces the observed` \
        `# "HfFileSystem.find() got multiple values for keyword argument maxdepth"` \
        "datasets>=4.0,<6" \
        "hydra-core" "omegaconf" "einops" "ninja" "codetiming" "dill" \
        "pylatexenc" "torchdata" "pyarrow>=19.0.0" "pandas" \
        "ray[default]>=2.41.0,<3" \
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
        "mathruler" "qwen-vl-utils" "cachetools" "nvtx" \
        "huggingface_hub>=0.35" "hf_transfer" \
        "fastapi" "uvicorn" "matplotlib" \
        "wandb" "tensorboard" "mlflow>=3.6,<=3.15.0"
# =============================================================================
# 5. Megatron stack — prebuilt binaries from the verl wheelhouse
#    --no-deps throughout: these wheels carry pins (or, for megatron-bridge,
#    metadata verl deliberately discards) that would drag the resolver off the
#    torch 2.11 ABI. Their real deps are already installed above.
# =============================================================================
# `make vendor` pre-fetches these four wheels on the HOST (which trusts the
# intercepting CA) into ./vendor/wheels. If present we install the local file and
# never touch github.com -- the bulletproof path when TLS inspection cannot be
# satisfied from inside the container. Empty by default -> installs from URL.
COPY vendor/wheels/ /vendor/wheels/
RUN set -eu; \
    for spec in "transformer_engine=${TE_WHL}" \
                "apex=${APEX_WHL}" \
                "flash_attn=${FA_WHL}" \
                "megatron_bridge=${MBRIDGE_NV_WHL}"; do \
      name="${spec%%=*}"; url="${spec#*=}"; \
      local_whl="$(ls /vendor/wheels/${name}-*.whl 2>/dev/null | head -1 || true)"; \
      if [ -n "${local_whl}" ]; then \
        echo "==> ${name}: vendored ${local_whl}"; \
        retry 3 uv pip install --no-deps "${local_whl}"; \
      else \
        echo "==> ${name}: fetching ${url}"; \
        retry 6 uv pip install --no-deps "${url}"; \
      fi; \
    done \
 && retry 6 uv pip install --index-url ${PIP_INDEX_URL} \
      "nvidia-modelopt==${MODELOPT_VERSION}" \
      "nvidia-mathdx" "onnxscript" "pydantic" \
 && python3 -c "import transformer_engine, flash_attn, apex; print('TE/FA/apex OK')"

# megatron-core: source install (pybind11 C++ ext, no CUDA toolchain).
# mbridge: the legacy HF<->mcore connector, still required by MEGATRON_MODE=classic
#          (verl's tested Qwen3.5-35B path uses vanilla_mbridge=True).
COPY vendor/src/ /vendor/src/
RUN set -eu; \
    if [ -d /vendor/src/megatron-lm ]; then \
      echo "==> megatron-core: vendored /vendor/src/megatron-lm"; \
      retry 3 uv pip install --no-deps --no-build-isolation /vendor/src/megatron-lm; \
    else \
      retry 6 uv pip install --no-deps --no-build-isolation \
        "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_VERSION}"; \
    fi; \
    if [ -d /vendor/src/mbridge ]; then \
      echo "==> mbridge: vendored /vendor/src/mbridge"; \
      retry 3 uv pip install --no-deps /vendor/src/mbridge; \
    else \
      retry 6 uv pip install --no-deps \
        "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_REV}"; \
    fi \
 && retry 3 uv pip install --no-deps "TransferQueue==${TRANSFER_QUEUE_VERSION}" || true

# cupy: NOT optional for this model. Megatron's MoE all-to-all dispatcher uses it
# directly, and ray.util.collective's NCCL backend is cupy-based (every verl
# weight-sync engine imports it). Without it the checkpoint engine never
# registers and rollout weight sync fails at step 1.
RUN retry 6 uv pip install --index-url ${PIP_INDEX_URL} "cupy-cuda13x==${CUPY_VERSION}"

# flash-linear-attention: Triton kernels for Qwen3.5's Gated-DeltaNet layers
# (30 of its 40 layers are linear attention). Pure-python wheel.
RUN retry 6 uv pip install --index-url ${PIP_INDEX_URL} "flash-linear-attention>=0.5.1"

# =============================================================================
# 6. verl itself, --no-deps, pinned
# =============================================================================
RUN retry 6 uv pip install --no-deps "verl @ git+https://github.com/volcengine/verl.git@${VERL_REF}" \
 && python3 -c "import verl; print('verl', verl.__version__)"

# =============================================================================
# 7. Optional bits
# =============================================================================
RUN if [ "${WITH_VIDEO}" = "1" ]; then \
      retry 6 apt-get update && retry 3 apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/* \
      && retry 6 uv pip install --index-url ${TORCH_INDEX_URL} torchcodec; \
    else echo "WITH_VIDEO=0 -> skipping torchcodec/ffmpeg (geo3k is images-only)"; fi

RUN if [ "${OVERRIDE_NCCL}" = "1" ]; then \
      retry 6 uv pip install --no-deps --upgrade "nvidia-nccl-cu13>=2.29.7,<3.0" \
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
RUN uv pip uninstall opencv-python-headless opencv-python 2>/dev/null || true; \
    retry 6 uv pip install --index-url ${PIP_INDEX_URL} "opencv-python-headless==${OPENCV_VERSION}" \
 && python3 -c "import cv2; print('cv2', cv2.__version__, 'FIPS-clean')"

# =============================================================================
# 9. Code + build-time verification
#    WORKDIR is NOT honoured by AI Runtime -> everything under /app, absolute.
# =============================================================================
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh /app/scripts/lib/*.sh 2>/dev/null || true

# Fail the build here rather than 6 minutes into `air register image` plus a
# 16-GPU job launch. Imports only; no GPU present during docker build.
RUN python3 - <<'PY'
import importlib, sys
mods = ["torch", "vllm", "transformers", "verl", "megatron.core", "megatron.bridge",
        "transformer_engine", "apex", "flash_attn", "fla", "mbridge", "cupy",
        "cv2", "ray", "mlflow", "mathruler", "qwen_vl_utils", "yaml", "pip"]
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
if bad:
    sys.exit(f"image verification failed: {[m for m, _ in bad]}")
print("image verification passed")
PY
