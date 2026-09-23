#!/usr/bin/env python3
"""Verify that a model directory is complete and servable, and print its identity.

    verify_checkpoint.py <path> [--train-checkpoint] [--json-out FILE] [--print-hf-dir]
    verify_checkpoint.py --list-complete <run_dir>

<path> is any of:
  * a training step dir      .../<run>/global_step_N         (verl fully-async / megatron layout)
  * its actor dir            .../<run>/global_step_N/actor
  * an HF model dir          .../actor/model/huggingface, or a staged base model

A training checkpoint counts only if ``actor/ckpt_contents.json`` exists: verl's Megatron
checkpoint manager writes it LAST, atomically, from rank 0, once every piece of the save is on
disk. Directories prove nothing -- verl's own path helper creates ``model/huggingface/`` as a side
effect, and an interrupted save leaves ``global_step_N/actor/{model,optimizer,extra}`` behind with
no weights. For any HF dir this checks config.json, the tokenizer files, and that every shard the
safetensors index names exists and is non-empty.

Exit 0 and print an identity JSON (config/index/manifest hashes, shard sizes, step, run dir) when
the model is usable; exit 1 with the reason otherwise. The identity is a cheap content address
(metadata + sizes, not a hash of ~70 GB of weights): enough to tell two checkpoints apart and to
tie an eval result to exactly what was served.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

MANIFEST = "ckpt_contents.json"
_STEP_RE = re.compile(r"^global_step_(\d+)$")


class CheckpointError(Exception):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise CheckpointError(f"{what} missing: {path}")
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CheckpointError(f"{what} is not valid JSON ({e}): {path}") from e


def verify_hf_dir(hf: Path) -> dict[str, Any]:
    """Weights + config + tokenizer of a HuggingFace model dir. Raises CheckpointError."""
    if not hf.is_dir():
        raise CheckpointError(f"not a directory: {hf}")
    cfg = _load_json(hf / "config.json", "config.json")
    if not isinstance(cfg, dict):
        raise CheckpointError(f"config.json is not an object: {hf}")
    if not (hf / "tokenizer_config.json").is_file() or not (
            (hf / "tokenizer.json").is_file() or (hf / "tokenizer.model").is_file()
            or ((hf / "vocab.json").is_file() and (hf / "merges.txt").is_file())):
        raise CheckpointError(f"tokenizer files missing (need tokenizer_config.json + tokenizer.json/"
                              f"tokenizer.model/vocab.json+merges.txt): {hf}")

    index_path = hf / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path, "model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"safetensors index has no weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
    elif (hf / "model.safetensors").is_file():
        index, shard_names = None, ["model.safetensors"]
    else:
        raise CheckpointError(f"no safetensors weights (no index, no model.safetensors): {hf}")

    shards: dict[str, int] = {}
    for name in shard_names:
        p = hf / name
        if not p.is_file():
            raise CheckpointError(f"shard listed in the index is missing: {p}")
        size = p.stat().st_size
        if size == 0:
            raise CheckpointError(f"shard is empty: {p}")
        shards[name] = size
    total = sum(shards.values())
    declared = (index or {}).get("metadata", {}).get("total_size") if index else None
    if isinstance(declared, int) and total < declared:
        raise CheckpointError(f"shards hold {total} bytes but the index declares {declared} "
                              f"of tensor data -- truncated copy? {hf}")
    return {
        "hf_dir": str(hf),
        "architectures": cfg.get("architectures"),
        "config_sha256": _sha(hf / "config.json"),
        "index_sha256": _sha(index_path) if index is not None else None,
        "shards": shards,
        "total_bytes": total,
    }


def _resolve(path: Path) -> tuple[Path, Path | None]:
    """-> (hf_dir, actor_dir or None). Anything inside a verl training checkpoint -- the step dir,
    actor/, actor/model/ or actor/model/huggingface/ -- resolves to its actor/ dir, so the
    completion manifest is always checked; anything else is a plain HF model dir."""
    if path.name == "huggingface" and path.parent.name == "model" and path.parent.parent.name == "actor":
        return path, path.parent.parent
    if path.name == "model" and path.parent.name == "actor":
        return path / "huggingface", path.parent
    if (path / "actor").is_dir():
        path = path / "actor"
    if path.name == "actor" or (path / MANIFEST).exists():
        return path / "model" / "huggingface", path
    return path, None


def verify(path: str | Path, *, require_train_checkpoint: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no such path: {path}")
    hf, actor = _resolve(path)
    ident: dict[str, Any] = {"input": str(path), "kind": "hf_model", "step": None, "run_dir": None,
                             "ckpt_contents_sha256": None}
    if actor is not None:
        manifest = _load_json(actor / MANIFEST, f"{MANIFEST} (written last by verl on a COMPLETE save)")
        step_dir = actor.parent
        m = _STEP_RE.match(step_dir.name)
        step = manifest.get("global_step") if isinstance(manifest, dict) else None
        if m and step is not None and int(m.group(1)) != int(step):
            raise CheckpointError(f"{MANIFEST} says global_step={step} but the dir is {step_dir.name}")
        model = (manifest.get("contents") or {}).get("model") or {}
        if model.get("format") != "huggingface":
            raise CheckpointError(f"checkpoint has no HuggingFace export (model format "
                                  f"{model.get('format')!r}); it cannot be served: {actor}")
        hf = actor / str(model.get("path") or "model/huggingface")
        ident.update(kind="train_checkpoint", step=int(step) if step is not None else None,
                     run_dir=str(step_dir.parent), ckpt_contents_sha256=_sha(actor / MANIFEST))
    elif require_train_checkpoint:
        raise CheckpointError(f"not a training checkpoint (no actor/{MANIFEST}): {path}")
    ident.update(verify_hf_dir(hf))
    digest = hashlib.sha256()
    for part in (ident["config_sha256"], ident["index_sha256"], ident["ckpt_contents_sha256"],
                 json.dumps(sorted(ident["shards"].items()))):
        digest.update(str(part).encode())
    ident["identity"] = digest.hexdigest()[:16]
    return ident


def list_complete(root: str | Path) -> list[tuple[int, Path]]:
    """Every global_step_N that verifies, in root or one run dir below it, oldest first."""
    out = []
    root = Path(root)
    for d in [*root.glob("global_step_*"), *root.glob("*/global_step_*")]:
        m = _STEP_RE.match(d.name)
        if not m:
            continue
        try:
            verify(d, require_train_checkpoint=True)
        except CheckpointError:
            continue
        out.append((int(m.group(1)), d))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--train-checkpoint", action="store_true",
                    help="require a verl training checkpoint (actor/ckpt_contents.json)")
    ap.add_argument("--json-out", help="also write the identity JSON here")
    ap.add_argument("--print-hf-dir", action="store_true", help="print only the servable HF dir")
    ap.add_argument("--list-complete", metavar="RUN_DIR", help="list complete global_step_N dirs")
    args = ap.parse_args(argv)

    if args.list_complete:
        steps = list_complete(args.list_complete)
        for step, d in steps:
            print(f"{step}\t{d}")
        return 0 if steps else 1
    if not args.path:
        ap.error("path is required")
    try:
        ident = verify(args.path, require_train_checkpoint=args.train_checkpoint)
    except CheckpointError as e:
        print(f"verify_checkpoint: FAIL: {e}", file=sys.stderr)
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(ident, indent=2))
    print(ident["hf_dir"] if args.print_hf_dir else json.dumps(ident, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
