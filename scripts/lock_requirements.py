#!/usr/bin/env python3
"""Resolve the AIR image lock, using the pinned verl checkout's frozen uv.lock.

Run with the dev environment: .venv/bin/python scripts/lock_requirements.py
The package index is detected as for make build and passed only in the environment.
Review the resulting requirements.lock before building a new image tag.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import tempfile
from pathlib import Path

from compose_check import REPO, ensure_verl_src
from image_lock import HAND_ALIGNED


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    src = ensure_verl_src()
    artifacts = {
        norm(fields[1])
        for line in (REPO / "docker/artifacts.lock").read_text().splitlines()
        if (fields := line.split()) and fields[0] in ("wheel", "git")
    }
    # AIR needs MLflow/pandas<3 and the actual PyPI torch CUDA dependencies.
    # numpy must satisfy mistral-common AND the FIPS-compatible OpenCV pin.
    # cuda-tile's released wheel avoids the prerelease NVIDIA wheel_stub fetch.
    air_resolution = {"pandas", "numpy", "nvidia-cusparselt-cu13", "cuda-tile",
                      "opencv-python-headless"}
    # The Flash judge trial uses released vLLM 0.30 + Transformers 5.16.1.
    # Their dependency pins supersede the older backend pins in verl's uv.lock;
    # torch and all native Megatron wheels retain the exact 2.13 ABI.
    air_resolution |= {
        "vllm", "transformers", "tokenizers", "flashinfer-python", "quack-kernels", "mcp",
        "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    }
    env = dict(os.environ)
    index = env.get("PIP_INDEX_URL") or subprocess.check_output(
        ["bash", str(REPO / "scripts/detect_pypi_index.sh")], text=True
    ).strip()
    if index:
        env["UV_DEFAULT_INDEX"] = index
        env["UV_INDEX_URL"] = index
    env["UV_SYSTEM_CERTS"] = "1"
    with tempfile.TemporaryDirectory(prefix="voa-lock-") as tmp:
        exported = Path(tmp) / "upstream.txt"
        cmd = ["uv", "export", "--project", str(src), "--frozen", "--extra", "vllm",
               "--extra", "megatron", "--no-dev", "--no-hashes", "--no-emit-project",
               "--no-header", "--output-file", str(exported)]
        for name in sorted(artifacts):
            cmd.extend(["--no-emit-package", name])
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        constraints = []
        for line in exported.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name = norm(re.split(r"==| @ ", line)[0].strip())
            if name in air_resolution:
                continue
            constraints.append(line.replace("+cu130", ""))
        preferred = Path(tmp) / "preferred.txt"
        preferred.write_text("\n".join(constraints) + "\n")
        resolved = Path(tmp) / "resolved.txt"
        subprocess.run(
            ["uv", "pip", "compile", str(REPO / "docker/requirements.in"),
             "--constraint", str(preferred), "--python-version", "3.12",
             "--python-platform", "x86_64-manylinux_2_28", "--no-header", "--no-annotate",
             "--output-file", str(resolved)],
            env=env, check=True, stdout=subprocess.DEVNULL,
        )
        pins = []
        for line in resolved.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name = norm(line.split("==", 1)[0])
            if name not in artifacts | HAND_ALIGNED:
                pins.append(line)
    header = (
        "# Exact index-package versions for the isolated verl 0.10.0.dev AIR trial.\n"
        f"# Resolved on {dt.date.today()} for Linux x86_64 / Python 3.12.\n"
        "# Regenerate: .venv/bin/python scripts/lock_requirements.py\n"
        "# Inputs: requirements.in + the pinned verl commit's frozen vLLM/Megatron uv.lock.\n"
        "# AIR adaptations: MLflow/pandas<3; numpy within mistral-common/OpenCV ranges;\n"
        "# PyPI torch's CUDA dependencies; cuda-tile 1.6.0's real PyPI wheel.\n"
        "# Flash judge: vLLM 0.30 / Transformers 5.16.1 and their backend dependency pins.\n"
        "# Native wheels/git trees are in artifacts.lock. Dockerfile aligns the four CUDA\n"
        "# toolchain packages and OpenCV separately; those are excluded here.\n"
        "# The build verifies every installed package against this lock and artifacts.lock.\n"
    )
    target = REPO / "docker/requirements.lock"
    target.write_text(header + "\n".join(pins) + "\n")
    print(f"Wrote {len(pins)} exact pins to {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
