#!/usr/bin/env python3
"""Stage the authoritative OfficeQA dataset (HF ``databricks/officeqa``) into the
UC Volume, ON Databricks -- so the data never round-trips through the local CLI /
Files API (which is rate-limited). Mirrors ``stage_model.py``: download each file
to LOCAL scratch (full POSIX), then SEQUENTIALLY stream-copy onto the UC Volumes
FUSE mount, which rejects the parallel range-writes that hf_transfer / Xet use
(``os error 95`` EOPNOTSUPP). HF_TOKEN is injected from df1's ``msh`` secret scope
via the air ``secrets:`` block.

Writes to ``OFFICEQA_VOL_DIR``:
    officeqa_pro.csv, officeqa_full.csv                    (always -- eval/train Qs;
                                                            Pro=133 hard, Full=246)
    treasury_bulletins_parsed/jsons/*.json                 (only if FETCH_JSONS=1 --
                                                            authoritative source for
                                                            corpus regen + synthesis)

Idempotent/resumable: a file already present with non-zero size is skipped, so a
retry or a later FETCH_JSONS=1 run continues where it left off.

Env: OFFICEQA_REPO (databricks/officeqa), OFFICEQA_VOL_DIR, SCRATCH_DIR,
     FETCH_JSONS (0/1), HF_TOKEN.
"""

from __future__ import annotations

import csv
import os
import shutil
import time

REPO = os.environ.get("OFFICEQA_REPO", "databricks/officeqa")
VOL = os.environ.get("OFFICEQA_VOL_DIR", "/Volumes/main/mshtelma/verl/data/officeqa")
FETCH_JSONS = os.environ.get("FETCH_JSONS", "0") == "1"
COPY_BUF = 32 * 1024 * 1024   # 32 MiB sequential writes to the FUSE mount


def pick_scratch() -> str:
    explicit = os.environ.get("SCRATCH_DIR")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    for cand in ("/local_disk0", "/tmp"):
        if os.path.isdir(cand):
            path = os.path.join(cand, "hf_officeqa")
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except OSError:
                continue
    raise RuntimeError("no writable scratch directory found")


def stream_copy(src: str, dst: str) -> None:
    """Sequential copy onto the FUSE mount, via a temp name then atomic rename."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, COPY_BUF)
        fdst.flush()
        os.fsync(fdst.fileno())
    os.replace(tmp, dst)


def fetch_one(hf_hub_download, name: str, token, scratch: str) -> tuple[str, int]:
    local = hf_hub_download(
        repo_id=REPO, filename=name, repo_type="dataset",
        local_dir=scratch, token=token,
    )
    dst = os.path.join(VOL, name)
    stream_copy(local, dst)
    try:
        os.remove(local)   # free scratch immediately
    except OSError:
        pass
    return dst, os.path.getsize(dst)


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    scratch = pick_scratch()
    os.makedirs(VOL, exist_ok=True)
    print(f"repo={REPO}\ndest={VOL}\nscratch={scratch}\nfetch_jsons={FETCH_JSONS}  "
          f"hf_xfer={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER','0')}  "
          f"xet_disabled={os.environ.get('HF_HUB_DISABLE_XET','0')}  "
          f"token={'set' if token else 'none'}", flush=True)

    # 1) CSVs first -- critical + tiny, so the eval is unblocked even if (2) is slow.
    for f in ("officeqa_pro.csv", "officeqa_full.csv"):
        dst, sz = fetch_one(hf_hub_download, f, token, scratch)
        rows = list(csv.DictReader(open(dst, newline="")))
        import collections
        diff = collections.Counter((r.get("difficulty") or "").strip().lower() for r in rows)
        print(f"[csv] {f} -> {dst}  ({sz} bytes, {len(rows)} records, difficulty={dict(diff)})", flush=True)

    # 2) parsed JSONs (bulk; authoritative source for clean-corpus regen + synthesis atoms).
    if FETCH_JSONS:
        api = HfApi(token=token)
        info = api.dataset_info(REPO, files_metadata=True)
        jsons = sorted(
            s.rfilename for s in info.siblings
            if s.rfilename.startswith("treasury_bulletins_parsed/jsons/")
            and s.rfilename.endswith(".json")
        )
        print(f"[json] {len(jsons)} json file(s) to stage", flush=True)
        t0 = time.time()
        copied = skipped = 0
        for i, name in enumerate(jsons, 1):
            dst = os.path.join(VOL, name)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                skipped += 1
            else:
                fetch_one(hf_hub_download, name, token, scratch)
                copied += 1
            if i % 100 == 0 or i == len(jsons):
                print(f"[json] {i}/{len(jsons)}  copied={copied} skipped={skipped}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[json] done: {copied} copied, {skipped} skipped -> {VOL}/treasury_bulletins_parsed/jsons/", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
