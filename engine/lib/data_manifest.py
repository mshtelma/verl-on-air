#!/usr/bin/env python3
"""Pinned dataset sources, and the manifest every data-prep job writes beside its outputs.

Every Hub dataset this repo reads is a `Source`: repo id, config, and the COMMIT it is read at.
A floating `main` lets the same job build a different dataset next month; a commit cannot.

Each prep job records what it built -- sources and revisions, row counts before and after every
filter, the sampling policy, and the sha256 and row count of every file written -- in
DATA_MANIFEST.json beside its outputs, or <stem>.manifest.json for a single artifact such as the
search corpus. `provenance(file)` reads that record back for an eval artifact or a run manifest
and says whether the file is still the one the manifest describes.

A mirror can publish different data under the same name. A fallback source is therefore tried
only with ALLOW_FALLBACK_SOURCE=1, and accepted only if its content equals the pinned source's
(`load_verified`): an unavailable primary never silently redefines the experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "voa.data_manifest/v1"
DIR_MANIFEST = "DATA_MANIFEST.json"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class SourceError(Exception):
    """A dataset could not be read as pinned, or a candidate does not hold the pinned data."""


@dataclass(frozen=True)
class Source:
    """One Hub dataset at one commit."""
    hf_id: str
    revision: str               # a full 40-hex commit -- never a branch or a tag
    config: str | None = None

    def __post_init__(self):
        if not _COMMIT.fullmatch(self.revision or ""):
            raise SourceError(f"{self.hf_id}: pin a 40-hex commit, not {self.revision!r}")

    @property
    def label(self) -> str:
        return f"{self.hf_id}{':' + self.config if self.config else ''}@{self.revision[:12]}"

    def load(self, split: str | None = None, **kw: Any):
        import datasets
        return datasets.load_dataset(self.hf_id, self.config, split=split, revision=self.revision, **kw)

    def record(self, **extra: Any) -> dict[str, Any]:
        return {"hf_id": self.hf_id, "config": self.config, "revision": self.revision, **extra}


def content_digest(rows: Iterable[dict], fields: Sequence[str]) -> str:
    """sha256 over `fields` of every row, independent of row order: equal digests = the same rows."""
    per_row = sorted(hashlib.sha256(json.dumps([r[f] for f in fields], default=str).encode()).hexdigest()
                     for r in rows)
    return hashlib.sha256("".join(per_row).encode()).hexdigest()


def load_verified(candidates: Sequence[tuple[Source, Callable[[Source], Any]]],
                  verify: Callable[[Any], str | None]) -> tuple[Any, dict[str, Any]]:
    """Load candidates[0] -- the pinned source -- with its loader and require `verify(data)` to
    return None (else it returns why this is not the pinned content). Later candidates are mirrors:
    tried only with ALLOW_FALLBACK_SOURCE=1, and taken only if they pass the same check.
    Returns (data, the source record for the manifest)."""
    allow = os.environ.get("ALLOW_FALLBACK_SOURCE", "0") == "1"
    tried: list[str] = []
    for i, (src, loader) in enumerate(candidates):
        if i and not allow:
            break
        try:
            data = loader(src)
            why = verify(data)
        except Exception as e:  # noqa: BLE001 - a candidate that cannot load is a reason, not a crash
            why = f"{type(e).__name__}: {e}"
        if why is None:
            if i:
                print(f"[data] {candidates[0][0].label} unavailable; using the mirror {src.label}, "
                      f"verified to hold the same content", flush=True)
            return data, src.record(fallback=bool(i), rejected=tried)
        tried.append(f"{src.label}: {why}")
    mirrors = len(candidates) - 1
    hint = (f" -- {mirrors} mirror(s) are listed; ALLOW_FALLBACK_SOURCE=1 tries them, and each is "
            f"accepted only if its content matches" if mirrors and not allow else "")
    raise SourceError("no source holds the pinned data: " + "; ".join(tried) + hint)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# (size, sha256) of each file write_parquet produced, computed from the LOCAL copy before it went to
# its destination: reading back a file just written -- or overwritten -- on a Volume FUSE mount can
# fail with EIO (acceptance run A3: `[Errno 5] Input/output error` hashing a fresh train.parquet).
_WRITTEN: dict[str, tuple[int, str]] = {}


def write_parquet(path: str | Path, data) -> Path:
    """Write `data` (a list of row dicts, or a datasets.Dataset) to local disk, hash it there, then
    copy it next to `path` and rename: `path` is never a half-written file, and its manifest
    record never needs to read it back."""
    import shutil
    import tempfile
    p = Path(path)
    if isinstance(data, list):
        import datasets
        data = datasets.Dataset.from_list(data)
    with tempfile.TemporaryDirectory(prefix="voa_parquet_") as d:
        local = Path(d) / p.name
        data.to_parquet(str(local))
        digest = (local.stat().st_size, sha256_file(local))
        tmp = p.with_name(f".{p.name}.partial")
        shutil.copyfile(local, tmp)
        os.replace(tmp, p)
    _WRITTEN[str(p.resolve())] = digest
    return p


def output_record(path: str | Path, rows: int, **extra: Any) -> dict[str, Any]:
    p = Path(path)
    size, sha = _WRITTEN.get(str(p.resolve())) or (p.stat().st_size, sha256_file(p))
    return {"name": p.name, "rows": rows, "bytes": size, "sha256": sha, **extra}


def _versions() -> dict[str, str | None]:
    import importlib.metadata as md
    out: dict[str, str | None] = {}
    for pkg in ("datasets", "huggingface_hub", "pyarrow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def write_manifest(path: str | Path, *, tool: str, sources: list[dict], outputs: list[dict],
                   **fields: Any) -> dict[str, Any]:
    """Atomically write the manifest; `outputs` are output_record()s of files already written."""
    doc = {"schema": SCHEMA, "tool": tool,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run_id": os.environ.get("RUN_ID"), "git_sha": os.environ.get("GIT_SHA"),
           "versions": _versions(), "sources": sources, **fields, "outputs": outputs}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str))
    os.replace(tmp, p)
    print(f"[data] manifest -> {p}", flush=True)
    return doc


def find_manifest(path: str | Path) -> tuple[Path, dict, dict] | None:
    """The manifest that lists `path` -- <stem>.manifest.json, else DATA_MANIFEST.json beside it --
    as (manifest path, manifest, the file's output record); None if no manifest lists it."""
    p = Path(path)
    for cand in (p.with_name(f"{p.stem}.manifest.json"), p.with_name(DIR_MANIFEST)):
        if cand.is_file():
            doc = json.loads(cand.read_text())
            entry = next((o for o in doc.get("outputs") or [] if o.get("name") == p.name), None)
            if entry is not None:
                return cand, doc, entry
    return None


def provenance(path: str | Path) -> dict[str, Any]:
    """The file's own sha256, plus what its manifest records about how it was built and whether it
    is still that file. `manifest` is None for data built before manifests existed."""
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    found = find_manifest(p)
    if found is None:
        return {**out, "manifest": None}
    m, doc, entry = found
    return {**out, "manifest": str(m), "tool": doc.get("tool"), "created_utc": doc.get("created_utc"),
            "run_id": doc.get("run_id"), "git_sha": doc.get("git_sha"), "sources": doc.get("sources"),
            "complete": doc.get("complete", True), "rows": entry.get("rows"),
            "file_matches_manifest": entry.get("sha256") == out["sha256"]}
