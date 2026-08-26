#!/usr/bin/env python3
"""Stage a HF checkpoint into a UC Volume once, instead of on every run.

Qwen3.5-35B-A3B is ~70 GB. Pulling it from Hugging Face on every training job
costs 10-20 minutes of paid H100 time per run and is the single biggest
avoidable expense in the iteration loop. Stage once, then point
`actor_rollout_ref.model.path` at the Volume.

Env:
    MODEL_ID      HF repo id            (default Qwen/Qwen3.5-35B-A3B)
    MODEL_DIR     destination directory (default <VOL>/models/<basename>)
    HF_TOKEN      only needed for gated repos (Qwen3.5 is Apache-2.0, public)
"""

from __future__ import annotations

import os
import shutil
import sys
import time

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
DEFAULT_DIR = f"/Volumes/mshtelma/rlonair/verl/models/{MODEL_ID.split('/')[-1]}"
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_DIR)

# Weights only. Skipping the duplicate .bin/.pth formats matters: several Qwen
# repos ship both safetensors and consolidated files, and pulling both doubles
# the transfer for no benefit.
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py", "*.jinja"]
IGNORE = ["*.bin", "*.pth", "*.pt", "original/*", "*.gguf"]


def free_gb(path: str) -> float:
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe or "/").free / 1024**3


def main() -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"model    : {MODEL_ID}")
    print(f"dest     : {MODEL_DIR}")
    print(f"hf_xfer  : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
    print(f"free     : {free_gb(MODEL_DIR):.0f} GiB at destination")

    started = time.time()
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        max_workers=int(os.environ.get("HF_MAX_WORKERS", "8")),
        token=os.environ.get("HF_TOKEN") or None,
    )
    elapsed = time.time() - started

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    print(f"\ndone in {elapsed / 60:.1f} min -> {total / 1024**3:.1f} GiB at {path}")

    # A truncated download surfaces much later as a cryptic safetensors error
    # inside a Ray worker, so verify the shard set here where it is obvious.
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        import json

        with open(index) as fh:
            shards = set(json.load(fh)["weight_map"].values())
        missing = [s for s in shards if not os.path.exists(os.path.join(path, s))]
        if missing:
            sys.exit(f"INCOMPLETE: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
        print(f"verified all {len(shards)} safetensors shards present")

    print(f"\nUse this in the air YAML:\n  model_name: {path}")


if __name__ == "__main__":
    main()
