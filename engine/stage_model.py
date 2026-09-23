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

ONE REVISION, VERIFIED. The requested revision (MODEL_REVISION, default "main") is
resolved ONCE to a commit, and every file is downloaded at that commit -- a repo update
mid-transfer can no longer produce a mixed snapshot. Each file's content is checked
against the Hub's own metadata (the LFS sha256 for weights, the git blob hash for small
files) before it is copied, and the finished directory records what it holds in
STAGED.json. A directory that already holds a DIFFERENT revision is refused: stage that
one into its own directory.

RESUMABLE: a file already on the Volume counts only if its content hash matches the
metadata (recorded in .staging.json as each file lands, so a retry does not re-hash what
this staging verified), so a timeout or retry picks up where it left off. A lock file
keeps two staging jobs from writing the same directory at once.

Env:
    MODEL_ID        HF repo id            (default Qwen/Qwen3.5-35B-A3B)
    MODEL_REVISION  branch, tag or commit (default main; pin a commit for reproducibility)
    MODEL_DIR       destination directory (default <VOL>/models/<basename>)
    SCRATCH_DIR     local staging dir     (default /local_disk0/hf_stage, else /tmp)
    HF_TOKEN        only for gated repos (Qwen3.5 is Apache-2.0, public)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import time

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
DEFAULT_DIR = f"/Volumes/main/mshtelma/verl/models/{MODEL_ID.split('/')[-1]}"
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_DIR)

# Weights only. Skipping duplicate .bin/.pth formats matters: several Qwen repos
# ship both safetensors and consolidated files, doubling the transfer for nothing.
ALLOW_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".py", ".jinja")
IGNORE_SUFFIXES = (".bin", ".pth", ".pt", ".gguf", ".onnx")
IGNORE_PREFIXES = ("original/",)

COPY_BUF = 32 * 1024 * 1024   # 32 MiB sequential writes to the FUSE mount
STAGED, PROGRESS, LOCK = "STAGED.json", ".staging.json", ".staging.lock"
LOCK_STALE_S = float(os.environ.get("STAGE_LOCK_STALE_H", "12")) * 3600


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


def expected_digest(sibling) -> tuple[str, str] | None:
    """The content hash the Hub publishes for a file: LFS sha256, else the git blob sha1."""
    lfs = getattr(sibling, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha:
        return "sha256", sha
    blob = getattr(sibling, "blob_id", None)
    return ("git-blob-sha1", blob) if blob else None


def file_digest(path: str, kind: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(COPY_BUF), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def take_lock(model_dir: str) -> str:
    lock = os.path.join(model_dir, LOCK)
    if os.path.exists(lock):
        age = time.time() - os.path.getmtime(lock)
        if age < LOCK_STALE_S:
            sys.exit(f"FATAL: {lock} is held ({open(lock).read().strip()}, {age / 60:.0f} min old): another "
                     "staging job is writing this directory. Delete the lock only if none is running.")
        print(f"WARNING: taking over a stale lock ({age / 3600:.1f} h old): {lock}")
    with open(lock, "w") as fh:
        fh.write(f"host={socket.gethostname()} pid={os.getpid()} since={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    return lock


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    scratch = pick_scratch()
    os.makedirs(MODEL_DIR, exist_ok=True)

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    commit = info.sha
    print(f"model    : {MODEL_ID} @ {MODEL_REVISION} = commit {commit}")
    print(f"dest     : {MODEL_DIR}   ({free_gb(MODEL_DIR):.0f} GiB free)")
    print(f"scratch  : {scratch}     ({free_gb(scratch):.0f} GiB free)")
    print(f"hf_xfer  : {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}   "
          f"xet_disabled: {os.environ.get('HF_HUB_DISABLE_XET', '0')}")

    # One directory, one revision: never mix a new commit's files into an older snapshot.
    for rec_name in (STAGED, PROGRESS):
        held = read_json(os.path.join(MODEL_DIR, rec_name)).get("revision")
        if held and held != commit:
            sys.exit(f"FATAL: {MODEL_DIR} holds {MODEL_ID}@{held} ({rec_name}), not {commit}. Stage this "
                     f"revision into its own directory, e.g. MODEL_DIR={MODEL_DIR.rstrip('/')}@{commit[:12]}")
    if read_json(os.path.join(MODEL_DIR, STAGED)).get("revision") == commit:
        print(f"already staged: {MODEL_ID}@{commit} ({STAGED}) -- nothing to do")
        return

    lock = take_lock(MODEL_DIR)
    try:
        siblings = sorted((s for s in info.siblings if wanted(s.rfilename)), key=lambda s: s.rfilename)
        total = sum(s.size or 0 for s in siblings)
        print(f"\n{len(siblings)} file(s), {total / 1024**3:.1f} GiB to transfer\n")
        if free_gb(scratch) < 12:
            print(f"WARNING: only {free_gb(scratch):.0f} GiB of scratch; largest shard "
                  "may not fit. Set SCRATCH_DIR to a bigger mount if this fails.")

        progress_path = os.path.join(MODEL_DIR, PROGRESS)
        progress = read_json(progress_path) or {"model_id": MODEL_ID, "revision": commit, "files": {}}
        started = time.time()
        done_bytes = skipped = 0
        for idx, sib in enumerate(siblings, 1):
            name, size = sib.rfilename, sib.size or 0
            digest = expected_digest(sib)
            dst = os.path.join(MODEL_DIR, name)
            rec = {"size": size, "digest": list(digest) if digest else None}
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                # resume: trust a file only if its CONTENT matches the Hub's hash
                if progress["files"].get(name) == rec or (digest and file_digest(dst, digest[0]) == digest[1]):
                    progress["files"][name] = rec
                    write_json(progress_path, progress)
                    skipped += 1
                    done_bytes += size
                    print(f"[{idx}/{len(siblings)}] skip  {name}  ({size / 1024**2:.0f} MiB, verified)")
                    continue

            t0 = time.time()
            local = hf_hub_download(repo_id=MODEL_ID, filename=name, revision=commit, local_dir=scratch,
                                    token=os.environ.get("HF_TOKEN") or None)
            dl = time.time() - t0
            if digest and file_digest(local, digest[0]) != digest[1]:
                sys.exit(f"FATAL: {name} downloaded from {MODEL_ID}@{commit} does not match its {digest[0]}")
            t1 = time.time()
            stream_copy(local, dst)
            cp = time.time() - t1
            if os.path.getsize(dst) != size:
                sys.exit(f"FATAL: {dst} is {os.path.getsize(dst)} bytes after the copy, expected {size}")
            try:  # free scratch immediately so the peak stays at one shard
                os.remove(local)
            except OSError:
                pass
            progress["files"][name] = rec
            write_json(progress_path, progress)
            done_bytes += size
            mib = size / 1024**2
            pct = 100 * done_bytes / total if total else 100
            print(f"[{idx}/{len(siblings)}] ok    {name}  {mib:.0f} MiB  "
                  f"dl {dl:.1f}s ({mib / max(dl, 1e-3):.0f} MiB/s)  "
                  f"cp {cp:.1f}s ({mib / max(cp, 1e-3):.0f} MiB/s)  [{pct:.0f}%]")

        elapsed = time.time() - started
        print(f"\ntransferred in {elapsed / 60:.1f} min ({skipped} verified in place, "
              f"{len(siblings) - skipped} copied)")

        # A truncated transfer otherwise surfaces much later as a cryptic safetensors
        # error inside a Ray worker, so verify the shard set here where it is obvious.
        index = os.path.join(MODEL_DIR, "model.safetensors.index.json")
        if os.path.exists(index):
            shards = set(read_json(index).get("weight_map", {}).values())
            missing = [s for s in sorted(shards) if not os.path.exists(os.path.join(MODEL_DIR, s))]
            if missing:
                sys.exit(f"INCOMPLETE: {len(missing)} shard(s) missing, e.g. {missing[:3]}")
            print(f"verified all {len(shards)} safetensors shards present")
        else:
            print("note: no safetensors index (single-shard model?) — skipping shard check")

        write_json(os.path.join(MODEL_DIR, STAGED), {
            "model_id": MODEL_ID, "requested_revision": MODEL_REVISION, "revision": commit,
            "files": progress["files"], "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        os.remove(progress_path)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass

    on_disk = sum(os.path.getsize(os.path.join(root, f)) for root, _, fs in os.walk(MODEL_DIR) for f in fs)
    print(f"\n{on_disk / 1024**3:.1f} GiB staged at {MODEL_DIR}  ({MODEL_ID}@{commit}, see {STAGED})")
    print(f"\nUse this in the air YAML:\n  model_name: {MODEL_DIR}")


if __name__ == "__main__":
    main()
