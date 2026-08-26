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
    verdict = "enough for OFFLOAD=1" if gib >= 500 else (
        "TIGHT for OFFLOAD=1 (need ~500 GB) -> prefer 16 GPUs + MEGATRON_MODE=fsdp + OFFLOAD=0"
    )
    if gib < 500:
        warnings.append("cpu_ram_below_500GB")
    return f"{gib:.0f} GiB  [{verdict}]"


check("nvidia-smi", _nvidia_smi)
check("cpu ram", _cpu_ram)
check("nproc", lambda: os.cpu_count())
check("efa devices", lambda: subprocess.run(
    ["bash", "-lc", "ls /sys/class/infiniband 2>/dev/null | tr '\\n' ' ' || echo none"],
    capture_output=True, text=True).stdout.strip() or "none", required=False)

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


check("torch facts", _torch_facts)

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
    return f"formatted+correct={s_good}  bare+correct={s_bad}  (expect 1.0 / 0.9)"


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
