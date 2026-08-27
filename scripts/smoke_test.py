#!/usr/bin/env python3
"""Pre-flight check for the verl-on-air image. Runs on 1xA10 in ~2 minutes.

Purpose: turn every "fails 40 minutes into a 16-GPU job" failure mode into a
cheap, fast, explicit assertion. Checks, in order of how much money they save:

  1. Host facts       - driver, GPU, CPU RAM (the offload=1 constraint), NCCL.
  2. Import graph     - torch / vllm / mcore / Megatron-Bridge / TE / apex / fla.
  3. Architecture     - Qwen3_5MoeForConditionalGeneration exists in this
                        transformers pin; the target config actually loads.
  4. Bridge coverage  - Megatron-Bridge can resolve a mapping for the model.
  5. Reward path      - mathruler grades a synthetic geo3k-style response.

Exit code is non-zero if any REQUIRED check fails.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import traceback

MODEL_ID = os.environ.get("SMOKE_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")

failures: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check(name: str, fn, required: bool = True):
    try:
        result = fn()
        print(f"  ok    {name}" + (f" -> {result}" if result is not None else ""))
        return result
    except Exception as exc:  # noqa: BLE001
        tag = "FAIL " if required else "warn "
        print(f"  {tag} {name}: {type(exc).__name__}: {exc}")
        if os.environ.get("SMOKE_VERBOSE"):
            traceback.print_exc()
        (failures if required else warnings).append(name)
        return None


# ---------------------------------------------------------------------------
section("1. Host")
# ---------------------------------------------------------------------------


def _nvidia_smi() -> str:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not on PATH")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return " | ".join(out.splitlines())


def _cpu_ram() -> str:
    # THE number that decides whether OFFLOAD=1 is viable. Qwen3.5-35B-A3B with
    # optimizer_offload_fraction=1 parks ~390 GB of Adam state in host RAM per
    # node, plus offloaded actor+ref params. Budget ~500 GB/node.
    with open("/proc/meminfo") as fh:
        total_kb = int(next(l for l in fh if l.startswith("MemTotal")).split()[1])
    gib = total_kb / 1024 / 1024
    # IMPORTANT: this reflects THIS node type. The smoke test runs on GPU_1xA10
    # (~62 GiB), whereas OFFLOAD=1 would run on GPU_8xH100 (P5-class, ~2 TiB). So a
    # low number here is NOT a verdict on rung 3 -- the launcher prints `free -g` on
    # the actual node when OFFLOAD=1. Only flag it when we are on an H100 box.
    is_h100 = "H100" in os.environ.get("SMOKE_GPU_NAME", "") or _gpu_is_h100()
    if is_h100:
        verdict = ("enough for OFFLOAD=1" if gib >= 500 else
                   "TOO SMALL for OFFLOAD=1 (~550 GiB needed) -> use 16 GPUs + fsdp + OFFLOAD=0")
        if gib < 500:
            warnings.append("h100_cpu_ram_below_550GiB")
    else:
        verdict = ("informational only: this is not an H100 node, so it says nothing "
                   "about OFFLOAD=1 feasibility")
    return f"{gib:.0f} GiB  [{verdict}]"


def _gpu_is_h100() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return "H100" in out
    except Exception:
        return False


def _driver_supports_cuda13() -> str:
    # df1/AWS uses the -cu13 base (CUDA 13.0.3), matching our torch cu130. CUDA
    # 13.0 still requires a host driver >= 580.65.06, so assert it: the documented
    # AWS node driver is 580.126.16, but node pools get rebuilt and this is the
    # one host fact we cannot control from the image.
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    major, minor = (int(x) for x in out.split(".")[:2])
    if (major, minor) < (580, 65):
        raise RuntimeError(
            f"driver {out} < 580.65.06 required by CUDA 13.0, so torch cu130 "
            "cannot initialise. Options: use a pool with a newer driver, or pin a "
            "CUDA 12 torch -- but note verl's wheelhouse is cu130-only, so that "
            "means source-building TransformerEngine/apex/flash-attn."
        )
    return f"{out} (>= 580.65.06 -> CUDA 13.0 OK)"


def _no_cuda_compat_shadowing() -> str:
    # A cuda-compat ahead of the driver's libcuda pins userspace BELOW the kernel
    # driver -> CUDA Error 803, and NCCL reporting "no GPUs found" while
    # nvidia-smi works fine. The base ships none on LD_LIBRARY_PATH; assert that
    # nothing we installed added one.
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    hits = [p for p in ldp.split(":") if "compat" in p.lower()]
    if hits:
        raise RuntimeError(f"cuda-compat on LD_LIBRARY_PATH: {hits} -> risk of Error 803")
    return f"clean (LD_LIBRARY_PATH={ldp or '<empty>'})"


def _efa() -> str:
    # AWS multi-node RDMA is EFA, and NCCL reaches it *through* aws-ofi-nccl --
    # unlike Azure, where NCCL speaks IB verbs natively. So a missing plugin here
    # means TCP fallback, not merely a lost optimisation.
    # EFA exposes an ibverbs device, so /sys/class/infiniband is the right probe.
    # Absent on 1xA10 (this job): G-family hosts have no EFA hardware. Expected.
    devs = sorted(os.listdir("/sys/class/infiniband")) if os.path.isdir(
        "/sys/class/infiniband") else []
    plugin_dirs = ["/opt/aws-ofi-nccl/install/lib", "/opt/amazon/ofi-nccl/lib"]
    plugin = next((d for d in plugin_dirs if os.path.isdir(d)), None)
    plugin_str = plugin or "MISSING"
    if not devs:
        warnings.append("no_efa_devices_on_this_node")
        return ("no EFA/ibverbs devices (expected on 1xA10 -- G-family has no EFA; "
                f"REQUIRED for multi-node H100). aws-ofi-nccl: {plugin_str}")
    return f"{len(devs)} device(s): {' '.join(devs)}. aws-ofi-nccl: {plugin_str}"


check("nvidia-smi", _nvidia_smi)
check("driver supports CUDA 13", _driver_supports_cuda13)
check("no cuda-compat shadowing", _no_cuda_compat_shadowing)
check("cpu ram", _cpu_ram)
check("nproc", lambda: os.cpu_count())
check("EFA / RDMA (AWS)", _efa, required=False)

# ---------------------------------------------------------------------------
section("2. Imports")
# ---------------------------------------------------------------------------
REQUIRED_MODULES = [
    "torch", "vllm", "transformers", "verl",
    "megatron.core", "megatron.bridge",
    "transformer_engine", "apex", "flash_attn", "fla", "mbridge",
    "cupy", "cv2", "ray", "mlflow", "mathruler", "qwen_vl_utils", "yaml",
]
for mod in REQUIRED_MODULES:
    check(f"import {mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", "ok"))


def _torch_facts() -> str:
    import torch
    return (f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))} "
            f"cxx11abi={torch._C._GLIBCXX_USE_CXX11_ABI} "
            f"gpus={torch.cuda.device_count()}")


def _cuda_actually_works() -> str:
    # Import success proves nothing about the CUDA stack. Allocate on device and
    # run a real kernel: driver/ABI mismatches surface here, not at import.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    checksum = (a @ a).float().sum().item()
    _, total = torch.cuda.mem_get_info()
    return (f"bf16 matmul ok (sum={checksum:.1f}), "
            f"torch.version.cuda={torch.version.cuda}, hbm={total / 2**30:.0f} GiB")


check("torch facts", _torch_facts)
check("CUDA works on device", _cuda_actually_works)

# A CXX11-ABI mismatch between torch and any CUDA extension shows up as
# "ImportError: undefined symbol: _ZN3c105Error..." at runtime, not build time.
check("TE + flash_attn load real symbols", lambda: (
    __import__("transformer_engine.pytorch", fromlist=["x"]) and
    __import__("flash_attn", fromlist=["flash_attn_func"]).flash_attn_func.__name__
))

# Triton JITs a host C launcher stub -> it needs a compiler on PATH at RUNTIME.
# Qwen3.5's Gated-DeltaNet layers are Triton kernels, so this is load-bearing.
check("C compiler on PATH (Triton runtime JIT)",
      lambda: shutil.which("cc") or shutil.which("gcc") or (_ for _ in ()).throw(
          RuntimeError("no cc/gcc — Triton GDN kernels will fail to compile")))

# ---------------------------------------------------------------------------
section("3. Qwen3.5 architecture support")
# ---------------------------------------------------------------------------


def _arch_class():
    from transformers import Qwen3_5MoeForConditionalGeneration
    return Qwen3_5MoeForConditionalGeneration.__name__


def _config_loads():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tc = getattr(cfg, "text_config", cfg)
    n_lin = sum(1 for t in getattr(tc, "layer_types", []) if t == "linear_attention")
    return (f"{cfg.architectures} layers={tc.num_hidden_layers} "
            f"experts={getattr(tc, 'num_experts', '?')}/top{getattr(tc, 'num_experts_per_tok', '?')} "
            f"hidden={tc.hidden_size} vocab={tc.vocab_size} gdn_layers={n_lin}")


check("transformers has Qwen3_5MoeForConditionalGeneration", _arch_class)
check(f"AutoConfig loads {MODEL_ID}", _config_loads)

# ---------------------------------------------------------------------------
section("4. Megatron-Bridge coverage")
# ---------------------------------------------------------------------------


def _bridge_resolves():
    # Megatron-FSDP is only reachable through Megatron-Bridge (verl threads
    # use_megatron_fsdp exclusively through the Bridge provider path), so Bridge
    # MUST know this architecture or MEGATRON_MODE=fsdp cannot work at all.
    from megatron.bridge import AutoBridge
    return type(AutoBridge.from_hf_pretrained(MODEL_ID, trust_remote_code=True)).__name__


check("AutoBridge resolves the model", _bridge_resolves, required=False)

# ---------------------------------------------------------------------------
section("5. Reward path")
# ---------------------------------------------------------------------------


def _geo3k_reward():
    from verl.utils.reward_score import geo3k
    good = r"<think>the triangle is right</think> so the answer is \boxed{42}"
    bad = "42"
    s_good, s_bad = geo3k.compute_score(good, "42"), geo3k.compute_score(bad, "42")
    assert s_good > s_bad, f"reward not discriminating: {s_good} vs {s_bad}"
    # NB: bare+correct is 0.0, NOT 0.9 -- extract_boxed_content() finds nothing in
    # an unboxed response, so the accuracy term is gated on \boxed{} too. Emitting
    # the box is a precondition for ANY reward. See scripts/reward/custom_reward.py.
    return (f"formatted+correct={s_good}  bare+correct={s_bad}  "
            f"(expect 1.0 / 0.0 -- accuracy is gated on \\boxed{{}})")


check("geo3k rule reward", _geo3k_reward)

# ---------------------------------------------------------------------------
section("Summary")
# ---------------------------------------------------------------------------
if warnings:
    print(f"  warnings ({len(warnings)}): {', '.join(warnings)}")
if failures:
    print(f"  FAILURES ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
print("  all required checks passed")
