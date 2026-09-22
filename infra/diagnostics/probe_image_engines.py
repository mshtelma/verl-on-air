#!/usr/bin/env python3
"""Introspect the verl training image: what engines/versions ship, whether they
can serve GLM-5.3-Flash (glm5_next), and how much disk headroom is left under
AIR's 20 GB image cap.

Decides the judge topology under the hard constraints (ONE image, <=20 GB, no
cross-job connectivity): can the EXISTING image serve GLM-5.3-Flash as-is, and if
not, what strong models CAN its vLLM serve (so the judge rides the same image with
no rebuild)?  Runs on the v5 image, 1x GPU_1xA10, no real GPU work.
"""
import importlib
import json
import os
import shutil
import subprocess


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "present(no __version__)")
    except Exception as e:
        return f"ABSENT ({type(e).__name__})"


print("================ image engine probe ================", flush=True)
for m in ["torch", "transformers", "vllm", "sglang", "flashinfer", "flash_attn", "verl", "ray"]:
    print(f"{m:14s}: {ver(m)}", flush=True)

print("\n--- CUDA / torch ---", flush=True)
try:
    import torch

    print("torch.version.cuda :", torch.version.cuda, flush=True)
except Exception as e:
    print("torch probe failed:", e, flush=True)

# Does the image's transformers understand GLM-5.3-Flash (glm5_next)?
GLM_PATH = os.environ.get("JUDGE_MODEL_PATH", "/Volumes/main/mshtelma/verl/models/GLM-5.3-Flash")
print(f"\n--- transformers AutoConfig on {GLM_PATH} (trust_remote_code) ---", flush=True)
try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(GLM_PATH, trust_remote_code=True)
    print("OK model_type =", getattr(cfg, "model_type", "?"),
          " architectures =", getattr(cfg, "architectures", "?"), flush=True)
except Exception as e:
    print("AutoConfig FAILED:", type(e).__name__, str(e)[:400], flush=True)

# Is glm5_next / the GLM arch in vLLM's model registry?
print("\n--- vLLM model registry: GLM / glm5_next support? ---", flush=True)
try:
    from vllm.model_executor.models.registry import ModelRegistry

    archs = sorted(ModelRegistry.get_supported_archs())
    glm = [a for a in archs if "glm" in a.lower() or "Glm" in a]
    print("GLM-ish archs supported:", glm, flush=True)
    print("total supported archs:", len(archs), flush=True)
except Exception as e:
    print("vLLM registry probe failed:", type(e).__name__, str(e)[:300], flush=True)

# Disk footprint (what a rebuild is working against, vs the 20 GB cap).
print("\n--- disk footprint ---", flush=True)
try:
    import site

    sp = site.getsitepackages()[0]
    for label, path in [("site-packages", sp), ("torch", os.path.join(sp, "torch")),
                        ("vllm", os.path.join(sp, "vllm")), ("sglang", os.path.join(sp, "sglang"))]:
        if os.path.exists(path):
            out = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
            print(f"{label:14s}: {out.stdout.strip()}", flush=True)
        else:
            print(f"{label:14s}: <absent>", flush=True)
    total, used, free = shutil.disk_usage("/")
    print(f"root fs: total={total//2**30}G used={used//2**30}G free={free//2**30}G", flush=True)
except Exception as e:
    print("footprint probe failed:", e, flush=True)

print("====================================================", flush=True)
