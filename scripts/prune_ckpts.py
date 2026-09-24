#!/usr/bin/env python3
"""Free Volume space in one training run's checkpoint dir: keep what matters, delete the rest.

    make prune-ckpts CKPT=<output_dir>/<RUN_ID> [KEEP=20,40] [CONFIRM=1]
    python3 scripts/prune_ckpts.py <run_dir> [--keep 20,40] [--confirm] [--profile df1]

A 35B checkpoint (Megatron state + optimizer + HF export) is ~0.5 TB; MAX_CKPT_TO_KEEP bounds a
run while it trains, this trims it afterwards. KEPT, always: the FINAL complete step (the one the
run's certificate verified), every step in --keep (e.g. the one you evaluated and chose), every
INCOMPLETE step (no actor/ckpt_contents.json -- nothing to reclaim safely without looking), and every
file that is not a global_step_N dir (run_manifest.json, run_result.json, the tracker). Without
--confirm it only prints the plan. Works on a UC Volume through the `databricks fs` CLI.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

_STEP = re.compile(r"^global_step_(\d+)$")


def fs(profile: str, *args: str) -> str:
    r = subprocess.run(["databricks", "fs", *args, "-p", profile], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"databricks fs {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout


def ls(profile: str, path: str) -> list[str]:
    out = fs(profile, "ls", f"dbfs:{path}", "--output", "json")
    entries = json.loads(out) if out.strip() else []
    if isinstance(entries, dict):
        entries = entries.get("files", [])
    return [e.get("name") or e["path"].rstrip("/").rsplit("/", 1)[-1] for e in entries]


def plan(profile: str, run_dir: str, keep: set[int]) -> tuple[list[int], list[int], list[int]]:
    """-> (delete, keep, incomplete) step numbers."""
    steps = sorted(int(m.group(1)) for n in ls(profile, run_dir) if (m := _STEP.match(n)))
    complete = [s for s in steps if "ckpt_contents.json" in ls(profile, f"{run_dir}/global_step_{s}/actor")]
    incomplete = [s for s in steps if s not in complete]
    if not complete:
        raise SystemExit(f"{run_dir}: no complete checkpoint (actor/ckpt_contents.json) -- nothing is pruned")
    unknown = sorted(keep - set(complete))
    if unknown:
        raise SystemExit(f"--keep names step(s) {unknown} with no complete checkpoint in {run_dir}")
    kept = sorted(keep | {complete[-1]})
    return [s for s in complete if s not in kept], kept, incomplete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="<output_dir>/<RUN_ID> on a UC Volume (/Volumes/...)")
    ap.add_argument("--keep", default="", help="comma list of steps to keep besides the final one")
    ap.add_argument("--confirm", action="store_true", help="delete (default: print the plan only)")
    ap.add_argument("--profile", default="df1")
    args = ap.parse_args(argv)
    run_dir = args.run_dir.rstrip("/")
    if not run_dir.startswith("/Volumes/") or _STEP.match(run_dir.rsplit("/", 1)[-1]):
        raise SystemExit(f"give the RUN dir (/Volumes/.../<RUN_ID>), not {args.run_dir!r}")
    keep = {int(x) for x in args.keep.split(",") if x.strip()}
    delete, kept, incomplete = plan(args.profile, run_dir, keep)
    print(f"{run_dir}\n  keep      {kept} (final = {kept[-1] if kept else '-'})"
          f"\n  delete    {delete or 'nothing'}"
          + (f"\n  untouched {incomplete} (incomplete: look before deleting)" if incomplete else ""))
    if not delete:
        return 0
    if not args.confirm:
        print("dry run -- re-run with CONFIRM=1 (--confirm) to delete")
        return 0
    for s in delete:
        fs(args.profile, "rm", "-r", f"dbfs:{run_dir}/global_step_{s}")
        print(f"  deleted global_step_{s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
