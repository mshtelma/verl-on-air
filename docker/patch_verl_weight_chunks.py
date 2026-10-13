#!/usr/bin/env python3
"""Guarded repair for strided NVIDIA Bridge exports in pinned verl 0.10.

The byte serializer needs logical row-major values. Its metadata relay needs no
tensor buffer. Refuse any source other than the exact pinned implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PATCH_ID = "verl010-strided-weight-chunks-v1"
BEFORE_SHA256 = "97b51131a5770892007b7fb07264a2657fe83255b085927acd08c952a5682f64"
BEFORE = "        buffer = weight.view(-1).view(torch.uint8)\n"
AFTER = ("        # Bridge can export strided tensors; transmit logical row-major bytes.\n"
         "        # Metadata-only relays must not allocate a copy of the weights.\n"
         "        buffer = None if meta_only else weight.contiguous().view(-1).view(torch.uint8)\n")


def source_path() -> Path:
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("verl is not installed")
    return Path(spec.origin).parent / "checkpoint_engine/base.py"


def patched_source(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != BEFORE_SHA256:
        raise RuntimeError("verl checkpoint serializer differs from the pinned source; refusing to patch")
    if source.count(BEFORE) != 1:
        raise RuntimeError("expected exactly one weight serializer")
    return source.replace(BEFORE, AFTER, 1)


def apply(path: Path) -> dict:
    source = path.read_text()
    if AFTER in source:
        original = source.replace(AFTER, BEFORE, 1)
        expected = patched_source(original)
        if source != expected:
            raise RuntimeError("serializer patch exists alongside unexpected source changes")
        changed = False
    else:
        expected = patched_source(source)
        temp = path.with_suffix(".voa.tmp")
        temp.write_text(expected)
        temp.chmod(path.stat().st_mode)
        temp.replace(path)
        changed = True
    return {"id": PATCH_ID, "path": str(path), "before_sha256": BEFORE_SHA256,
            "after_sha256": hashlib.sha256(expected.encode()).hexdigest(), "changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = apply(args.source or source_path())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print("VERL_SOURCE_PATCH " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
