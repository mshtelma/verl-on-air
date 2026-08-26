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
# WHY THIS BASE  (AZURE — df2 is an Azure workspace)
#   AI Runtime DCS requires a Docker Hub image <20 GB, and multi-node H100 needs
#   the cloud's RDMA stack, which the Databricks base images ship pre-wired. On
#   Azure that is InfiniBand: the base carries the NCCL RDMA/SHARP plugin at
#   /opt/nccl-rdma-sharp-plugins/lib (already on LD_LIBRARY_PATH) and presets
#   NCCL_ALGO=Tree. We use the *runtime* (not devel) variant: ~4.1 GB vs ~10.3 GB.
#   That headroom is the whole reason this fits, because the Megatron stack
#   (TransformerEngine + apex + flash-attn) is heavy.
#
# CUDA 12.9 BASE + CUDA 13 WHEELS — deliberate, and why it is safe
#   There is NO -cu13 Azure base image (verified against the Docker Hub tag list:
#   only dcs-base-azure-{devel,runtime} exist, both CUDA 12.9.1, whereas AWS also
#   publishes -cu13). Our stack is torch 2.11.0+cu130. This works because:
#     1. The pip cu130 wheels BUNDLE their own CUDA 13 runtime libraries under
#        site-packages/nvidia/* (cudart, cublas, cudnn, nccl). The base image's
#        CUDA *toolkit* version is therefore not what torch links against.
#     2. CUDA library sonames are major-versioned (libcublas.so.12 vs .so.13), so
#        the base's /usr/local/cuda/lib64 cannot shadow the wheels' CUDA 13 libs.
#     3. What torch needs from outside the image is libcuda.so.1 (the driver API),
#        injected by the container runtime. The Azure host driver is 580.105.08,
#        and CUDA 13.0 requires >= 580.65.06. Satisfied.
#     4. CRITICALLY: `cuda-compat` is NOT on this image's LD_LIBRARY_PATH
#        (verified: LD_LIBRARY_PATH=/opt/nccl-rdma-sharp-plugins/lib:/usr/local/
#        cuda/lib64). Had a cuda-compat-12-9 been ahead of the driver's libcuda,
#        it would pin userspace below the kernel driver and CUDA 13 would fail
#        with Error 803. It is not, so we must not add one.
#   The residual risk is NCCL, not CUDA — see the OVERRIDE_NCCL note below.
#
# WHY WE DON'T NEED nvcc
#   verl publishes a prebuilt wheelhouse of cu130 / torch-2.11 / cp312 native
#   wheels: https://verl-project.github.io/verl-wheelhouse/simple/
#   TransformerEngine, apex and flash-attn all come from there as binaries, so
#   nothing CUDA compiles at build time. This is also what lets us ignore the
#   base's 12.9 nvcc entirely. We install them by DIRECT WHEEL URL rather than
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
FROM databricksruntime/air:dcs-base-azure-runtime

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

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

# Prebuilt native wheels from the verl wheelhouse, pinned by direct URL.
ARG WH=https://github.com/verl-project/verl-wheelhouse/releases/download
ARG TE_WHL=${WH}/transformer-engine-v2.16.1/transformer_engine-2.16.1-cp312-cp312-linux_x86_64.whl
ARG APEX_WHL=${WH}/apex-master/apex-0.1-cp312-cp312-linux_x86_64.whl
ARG FA_WHL=${WH}/flash-attention-v2.8.3/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
ARG MBRIDGE_NV_WHL=${WH}/megatron-bridge-r0.5.0/megatron_bridge-0.5.2-py3-none-any.whl

# Optional: override torch's bundled NCCL to >=2.29.7 for ncclCommSuspend/Resume
# (verl issue #6266). OFF by default.
#
# Azure specifics: the base ships libnccl2 2.27.3+cuda12.9 plus an RDMA/SHARP
# NCCL plugin built against it, while torch's cu130 wheel bundles its own NCCL
# ~2.28.x and loads that preferentially. So there is already a version skew
# between torch's NCCL and the plugin, and the plugin may decline to load.
# That is tolerable on Azure in a way it would not be on AWS: NCCL speaks
# InfiniBand verbs NATIVELY, so losing the plugin costs the SHARP in-network
# reduction optimisation, not RDMA itself. Bumping NCCL here only widens that
# skew. Read NCCL_DEBUG=INFO from a real 2-node run (rung 4 sets it) before
# flipping this to 1.
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

# =============================================================================
# 1. Toolchain + harness prerequisites
# =============================================================================
# The AI Runtime job harness itself imports pip and yaml before your command
# ever runs; on a bare base venv it dies in ~1 s with "No module named pip".
# build-essential is needed twice over: megatron-core's pybind11 ext at BUILD
# time, and Triton's launcher stub at RUN time (see the header note).
#
# python3-dev (unversioned) rather than python3.12-dev: this base sets
# UV_PYTHON_INSTALL_DIR=/opt/uv/python, so /opt/venv's interpreter may be a
# uv-managed standalone build carrying its OWN headers, not the system python3.
# In that case apt's headers are for the wrong interpreter and megatron-core
# would fail with a confusing "Python.h: No such file". We therefore ASSERT that
# Python.h is reachable from the venv interpreter's own sysconfig, so the build
# fails here with a diagnostic instead of two layers later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev git curl \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --no-cache pip setuptools wheel pybind11 ninja packaging pyyaml \
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
RUN for i in 1 2 3 4 5 6; do \
      uv pip install --index-url ${TORCH_INDEX_URL} \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" && break || \
      { echo "torch attempt $i failed; retrying"; sleep 15; }; \
    done \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"

# =============================================================================
# 3. vLLM 0.24.0 rollout engine  (first vLLM with Qwen3.5 support)
# =============================================================================
RUN for i in 1 2 3 4 5 6; do \
      uv pip install --index-url ${PIP_INDEX_URL} "vllm==${VLLM_VERSION}" && break || \
      { echo "vllm attempt $i failed; retrying"; sleep 15; }; \
    done

# =============================================================================
# 4. verl runtime dependencies
#    verl itself is installed --no-deps: its published metadata pins numpy<2 and
#    vllm<=0.12, both irreconcilable with the vllm 0.24 / Qwen3.5-era stack. We
#    supply the real deps explicitly, which is also what verl's own Dockerfile
#    effectively does.
# =============================================================================
RUN for i in 1 2 3 4 5 6; do \
      uv pip install --index-url ${PIP_INDEX_URL} \
        "torch==${TORCH_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "accelerate>=0.34,<2" "datasets>=3.0,<6" "peft" \
        "hydra-core" "omegaconf" "einops" "ninja" "codetiming" "dill" \
        "pylatexenc" "torchdata" "pyarrow>=19.0.0" "pandas" \
        "ray[default]>=2.41.0,<3" \
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
        "mathruler" "qwen-vl-utils" "cachetools" "nvtx" \
        "fastapi" "uvicorn" "matplotlib" \
        "wandb" "tensorboard" "mlflow>=3.6,<=3.15.0" \
      && break || { echo "deps attempt $i failed; retrying"; sleep 15; }; \
    done

# =============================================================================
# 5. Megatron stack — prebuilt binaries from the verl wheelhouse
#    --no-deps throughout: these wheels carry pins (or, for megatron-bridge,
#    metadata verl deliberately discards) that would drag the resolver off the
#    torch 2.11 ABI. Their real deps are already installed above.
# =============================================================================
RUN uv pip install --no-deps "${TE_WHL}" \
 && uv pip install --no-deps "${APEX_WHL}" \
 && uv pip install --no-deps "${FA_WHL}" \
 && uv pip install --no-deps "${MBRIDGE_NV_WHL}" \
 && uv pip install --index-url ${PIP_INDEX_URL} \
      "nvidia-modelopt==${MODELOPT_VERSION}" \
      "nvidia-mathdx" "onnxscript" "pydantic" \
 && python3 -c "import transformer_engine, flash_attn, apex; print('TE/FA/apex OK')"

# megatron-core: source install (pybind11 C++ ext, no CUDA toolchain).
# mbridge: the legacy HF<->mcore connector, still required by MEGATRON_MODE=classic
#          (verl's tested Qwen3.5-35B path uses vanilla_mbridge=True).
RUN uv pip install --no-deps --no-build-isolation \
      "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_VERSION}" \
 && uv pip install --no-deps \
      "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_REV}" \
 && uv pip install --no-deps "TransferQueue==${TRANSFER_QUEUE_VERSION}" || true

# cupy: NOT optional for this model. Megatron's MoE all-to-all dispatcher uses it
# directly, and ray.util.collective's NCCL backend is cupy-based (every verl
# weight-sync engine imports it). Without it the checkpoint engine never
# registers and rollout weight sync fails at step 1.
RUN uv pip install --index-url ${PIP_INDEX_URL} "cupy-cuda13x==${CUPY_VERSION}"

# flash-linear-attention: Triton kernels for Qwen3.5's Gated-DeltaNet layers
# (30 of its 40 layers are linear attention). Pure-python wheel.
RUN uv pip install --index-url ${PIP_INDEX_URL} "flash-linear-attention>=0.5.1"

# =============================================================================
# 6. verl itself, --no-deps, pinned
# =============================================================================
RUN uv pip install --no-deps "verl @ git+https://github.com/volcengine/verl.git@${VERL_REF}" \
 && python3 -c "import verl; print('verl', verl.__version__)"

# =============================================================================
# 7. Optional bits
# =============================================================================
RUN if [ "${WITH_VIDEO}" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/* \
      && uv pip install --index-url ${TORCH_INDEX_URL} torchcodec; \
    else echo "WITH_VIDEO=0 -> skipping torchcodec/ffmpeg (geo3k is images-only)"; fi

RUN if [ "${OVERRIDE_NCCL}" = "1" ]; then \
      uv pip install --no-deps --upgrade "nvidia-nccl-cu13>=2.29.7,<3.0" \
      && echo "NCCL overridden - confirm IB still binds via NCCL_DEBUG=INFO"; \
    else echo "OVERRIDE_NCCL=0 -> keeping torch's bundled NCCL (Azure IB-safe default)"; fi

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
    uv pip install --index-url ${PIP_INDEX_URL} "opencv-python-headless==${OPENCV_VERSION}" \
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
