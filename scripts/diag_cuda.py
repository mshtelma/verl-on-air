#!/usr/bin/env python3
"""Locate the pip-installed CUDA toolchain precisely (bounded searches only)."""
from __future__ import annotations
import os, subprocess

SP = "/opt/venv/lib/python3.12/site-packages"

def sh(cmd: str, timeout: int = 45) -> str:
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip() or "(empty)"
    except Exception as exc:
        return f"(failed: {exc})"

print("== site-packages/nvidia tree ==")
print(sh(f"ls -1 {SP}/nvidia 2>/dev/null"))
print("\n== nvcc binaries under site-packages (bounded) ==")
print(sh(f"find {SP}/nvidia -maxdepth 4 -name nvcc -o -maxdepth 4 -name 'nvcc*' 2>/dev/null | head -10"))
print("\n== bin dirs under nvidia/* ==")
print(sh(f"find {SP}/nvidia -maxdepth 3 -type d -name bin 2>/dev/null | head -10"))
print("\n== contents of each nvidia bin dir ==")
print(sh(f"for d in $(find {SP}/nvidia -maxdepth 3 -type d -name bin 2>/dev/null); do echo \"--- $d\"; ls -1 $d | head -12; done"))
print("\n== nvvm (nvcc needs it) ==")
print(sh(f"find {SP}/nvidia -maxdepth 3 -type d -name nvvm 2>/dev/null | head"))
print("\n== include dirs ==")
print(sh(f"find {SP}/nvidia -maxdepth 3 -type d -name include 2>/dev/null | head"))
print("\n== /usr/local/cuda ==")
print(sh("ls -la /usr/local/cuda/ && echo '--- include:' && ls /usr/local/cuda/include | head"))
print("\n== can nvcc actually run? ==")
print(sh(f"NVCC=$(find {SP}/nvidia -maxdepth 4 -type f -name nvcc 2>/dev/null | head -1); "
         f"echo \"nvcc=$NVCC\"; [ -n \"$NVCC\" ] && $NVCC --version 2>&1 | tail -3"))
print("\n== flashinfer jit env ==")
print(sh("python3 -c \"from flashinfer.jit import env as e; "
         "[print(f'{k}={getattr(e,k)}') for k in dir(e) if k.isupper()]\" 2>&1 | head -20"))
