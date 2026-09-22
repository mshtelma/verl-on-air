#!/usr/bin/env python3
"""Is the nvcc fix actually PRESENT in the image this job is running?

Distinguishes two hypotheses for `/bin/sh: /usr/local/cuda/bin/nvcc: not found`:
  A) the image is STALE (same :v1 tag re-pushed; registration is per-tag, so the
     platform may still serve the previously cached content)
  B) the symlink exists but is dangling / the binary is unusable at runtime
"""
from __future__ import annotations
import os, shutil, subprocess

def sh(cmd: str, t: int = 30) -> str:
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=t)
        return (r.stdout + r.stderr).strip() or "(empty)"
    except Exception as e:
        return f"(failed: {e})"

print("== marker: does the image contain the 5b nvcc fix? ==")
print(f"  /usr/local/cuda/bin exists : {os.path.exists('/usr/local/cuda/bin')}")
print(f"  is a symlink               : {os.path.islink('/usr/local/cuda/bin')}")
if os.path.islink("/usr/local/cuda/bin"):
    tgt = os.readlink("/usr/local/cuda/bin")
    print(f"  -> target                  : {tgt}")
    print(f"  -> target exists           : {os.path.exists(tgt)}   <-- False = DANGLING")
print(f"  ls -la /usr/local/cuda     :\n{sh('ls -la /usr/local/cuda/')}")
print(f"\n  PATH                       : {os.environ.get('PATH')}")
print(f"  CUDA_HOME                  : {os.environ.get('CUDA_HOME')}")
print(f"  which nvcc                 : {shutil.which('nvcc')}")
print(f"  nvcc direct                : {sh('/usr/local/cuda/bin/nvcc --version 2>&1 | tail -2')}")
print(f"  pip nvcc                   : {sh('ls -la /opt/venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc 2>&1')}")
print(f"  HF_XET_HIGH_PERFORMANCE    : {os.environ.get('HF_XET_HIGH_PERFORMANCE')}  <-- set only in the NEW image")
print(f"  HF_HUB_ENABLE_HF_TRANSFER  : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}  <-- set only in the OLD image")
print(f"\n  cuda_runtime.h grafted     : {sh('ls -la /usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h 2>&1')}")
