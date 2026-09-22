#!/usr/bin/env python3
"""Stage a HF checkpoint into a UC Volume once, instead of on every run.

Qwen3.5-35B-A3B is ~70 GB. Pulling it from Hugging Face on every training job
costs 10-20 minutes of paid H100 time per run, so we stage once and point
`actor_rollout_ref.model.path` at the Volume.

WHY THIS IS NOT JUST snapshot_download(local_dir=<volume>)
----------------------------------------------------------
UC Volumes are a FUSE mount that supports sequential writes but NOT the
random-access / sparse-file patterns that HF's accelerated downloaders use.
Pointing `snapshot_download` straight at the Volume fails with:

    RuntimeError: Data processing error: CAS service error :
                  IO Error: Operation not supported (os error 95)

(os error 95 = EOPNOTSUPP.) Xet/CAS and hf_transfer both seek and write ranges
in parallel; the mount rejects it.

So we download each file to LOCAL scratch (full POSIX, fast) and then copy it to
the Volume with a plain sequential streaming write. Doing this per-file rather
than per-snapshot bounds local disk use to the largest single shard (~5 GB)
instead of the whole 70 GB, which matters because node scratch is finite.

A useful side effect: it is RESUMABLE. Files already on the Volume with the
correct byte size are skipped, so a timeout or retry picks up where it left off.

Env:
    MODEL_ID      HF repo id            (default Qwen/Qwen3.5-35B-A3B)
    MODEL_DIR     destination directory (default <VOL>/models/<basename>)
    SCRATCH_DIR   local staging dir     (default /local_disk0/hf_stage, else /tmp)
    HF_TOKEN      only for gated repos (Qwen3.5 is Apache-2.0, public)
"""

from __future__ import annotations

import os
import shutil
import sys
import time

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
DEFAULT_DIR = f"/Volumes/main/mshtelma/verl/models/{MODEL_ID.split('/')[-1]}"
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_DIR)

# Weights only. Skipping duplicate .bin/.pth formats matters: several Qwen repos
# ship both safetensors and consolidated files, doubling the transfer for nothing.
ALLOW_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".py", ".jinja")
IGNORE_SUFFIXES = (".bin", ".pth", ".pt", ".gguf", ".onnx")
IGNORE_PREFIXES = ("original/",)

COPY_BUF = 32 * 1024 * 1024   # 32 MiB sequential writes to the FUSE mount


def pick_scratch() -> str:
    explicit = os.environ.get("SCRATCH_DIR")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    for cand in ("/local_disk0", "/tmp"):
        if os.path.isdir(cand):
            path = os.path.join(cand, "hf_stage")
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except OSError:
                continue
    raise RuntimeError("no writable scratch directory found")


def free_gb(path: str) -> float:
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe or "/").free / 1024**3


def wanted(name: str) -> bool:
    if name.startswith(IGNORE_PREFIXES) or name.endswith(IGNORE_SUFFIXES):
        return False
    return name.endswith(ALLOW_SUFFIXES)


def stream_copy(src: str, dst: str) -> None:
    """Sequential copy onto the FUSE mount, via a temp name then rename."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, COPY_BUF)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    scratch = pick_scratch()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"model    : {MODEL_ID}")
    print(f"dest     : {MODEL_DIR}   ({free_gb(MODEL_DIR):.0f} GiB free)")
    print(f"scratch  : {scratch}     ({free_gb(scratch):.0f} GiB free)")
    print(f"hf_xfer  : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}   "
          f"xet_disabled: {os.environ.get('HF_HUB_DISABLE_XET', '0')}")

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.model_info(MODEL_ID, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings if wanted(s.rfilename)]
    total = sum(sz for _, sz in files)
    print(f"\n{len(files)} file(s), {total / 1024**3:.1f} GiB to transfer\n")

    if free_gb(scratch) < 12:
        print(f"WARNING: only {free_gb(scratch):.0f} GiB of scratch; largest shard "
              "may not fit. Set SCRATCH_DIR to a bigger mount if this fails.")

    started = time.time()
    done_bytes = skipped = 0

    for idx, (name, size) in enumerate(sorted(files), 1):
        dst = os.path.join(MODEL_DIR, name)
        # Resume: trust an existing file only if the byte size matches exactly.
        if os.path.exists(dst) and size and os.path.getsize(dst) == size:
            skipped += 1
            done_bytes += size
            print(f"[{idx}/{len(files)}] skip  {name}  ({size / 1024**2:.0f} MiB, already staged)")
            continue

        t0 = time.time()
        local = hf_hub_download(
            repo_id=MODEL_ID,
            filename=name,
            local_dir=scratch,
            token=os.environ.get("HF_TOKEN") or None,
        )
        dl = time.time() - t0

        t1 = time.time()
        stream_copy(local, dst)
        cp = time.time() - t1

        # Free scratch immediately so the peak stays at one shard.
        try:
            os.remove(local)
        except OSError:
            pass

        done_bytes += size
        mib = size / 1024**2
        pct = 100 * done_bytes / total if total else 100
        print(f"[{idx}/{len(files)}] ok    {name}  {mib:.0f} MiB  "
              f"dl {dl:.1f}s ({mib / max(dl, 1e-3):.0f} MiB/s)  "
              f"cp {cp:.1f}s ({mib / max(cp, 1e-3):.0f} MiB/s)  [{pct:.0f}%]")

    elapsed = time.time() - started
    print(f"\ntransferred in {elapsed / 60:.1f} min "
          f"({skipped} skipped, {len(files) - skipped} copied)")

    # A truncated transfer otherwise surfaces much later as a cryptic safetensors
    # error inside a Ray worker, so verify the shard set here where it is obvious.
    index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if os.path.exists(index):
        import json

        with open(index) as fh:
            shards = set(json.load(fh)["weight_map"].values())
        missing = [s for s in sorted(shards)
                   if not os.path.exists(os.path.join(MODEL_DIR, s))]
        if missing:
            sys.exit(f"INCOMPLETE: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
        print(f"verified all {len(shards)} safetensors shards present")
    else:
        print("note: no safetensors index (single-shard model?) — skipping shard check")

    on_disk = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, fs in os.walk(MODEL_DIR) for f in fs
    )
    print(f"\n{on_disk / 1024**3:.1f} GiB staged at {MODEL_DIR}")
    print(f"\nUse this in the air YAML:\n  model_name: {MODEL_DIR}")


if __name__ == "__main__":
    main()
