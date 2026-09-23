#!/usr/bin/env python3
"""Decide whether a fully-async training run COMPLETED -- for every exit code, including 0.

    run_certificate.py snapshot <ckpt_dir>        # tracker state BEFORE the run (JSON on stdout)
    run_certificate.py check --ckpt-dir D --expected-final N --pre '<snapshot>' --raw-rc RC \\
        [--log LOG] [--abort-file F] [--json-out FILE ...] [--settle-s S]
    # prints the verdict; exit status = the code the launcher should exit with

Why the exit code means nothing on its own (verl v0.9.0 fully_async_policy):
  * a crash can exit 0 -- the Rollouter gathers its tasks with return_exceptions=True and then
    sends the ordinary stop signal, so a rollout/reward failure ends as a NORMAL stop: the Trainer
    force-saves whatever version it reached and both components report success;
  * a finished run can exit non-zero -- the component that finishes cancels the other, which
    surfaces as `RuntimeError: cancelled`; and "[ASYNC MAIN] Training completed or interrupted" is
    printed from a `finally:` block, i.e. after failures too.

A run is certified complete only if ALL of:
  1. <ckpt_dir>/latest_checkpointed_iteration.txt was (re)written during this run -- verl writes it
     only after the actor AND rollouter saves have returned;
  2. its value is the planned final parameter version = total_rollout_steps / samples-per-sync
     (the launcher refuses budgets that do not divide exactly, so this is exact);
  3. global_step_<N> verifies: actor/ckpt_contents.json + a complete HF export;
  4. nobody raised the run's abort channel (run_control.py);
  5. when overriding a NON-ZERO exit, the log shows no hard-failure signature (defence in depth).
Otherwise the run FAILED: exit with the raw code if it was non-zero, else 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_control  # noqa: E402
import verify_checkpoint as vc  # noqa: E402

TRACKER = "latest_checkpointed_iteration.txt"
# Signatures of a real crash. NCCL/DistBackend forms are listed explicitly: a grad-norm all_reduce
# OOM surfaces as "DistBackendError: NCCL error ... Cuda failure 2 'out of memory'".
HARD_ERROR_RE = re.compile(
    r"OutOfMemoryError|out of memory|No available memory for the cache blocks|not found in safetensors"
    r"|AssertionError|Error executing job.*(assert|shape|size mismatch)|Engine core initialization failed"
    r"|died unexpectedly|Cuda error.*invalid argument|NCCL error|Cuda failure|RayTaskError\(DistBackendError\)",
    re.IGNORECASE)


def snapshot(ckpt_dir: str | Path) -> dict[str, Any]:
    t = Path(ckpt_dir) / TRACKER
    if not t.is_file():
        return {"exists": False}
    st = t.stat()
    return {"exists": True, "value": t.read_text().strip(), "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def hard_errors(log: str | Path | None) -> list[str]:
    if not log or not Path(log).is_file():
        return []
    hits = []
    with open(log, errors="replace") as fh:
        for line in fh:
            if HARD_ERROR_RE.search(line):
                hits.append(line.strip()[:300])
                if len(hits) >= 5:
                    break
    return hits


def evaluate(ckpt_dir: str | Path, expected_final: int, pre: dict[str, Any], raw_rc: int, *,
             log: str | Path | None = None, abort_file: str | Path | None = None) -> dict[str, Any]:
    ckpt_dir = Path(ckpt_dir)
    now = snapshot(ckpt_dir)
    problems: list[str] = []
    ident = None
    if not now["exists"]:
        problems.append(f"no {TRACKER}: this run completed no checkpoint")
    elif pre.get("exists") and all(now.get(k) == pre.get(k) for k in ("value", "mtime_ns", "size")):
        problems.append(f"{TRACKER} is unchanged since before the run (version {now['value']}): "
                        "nothing was checkpointed by THIS run")
    elif now["value"] != str(expected_final):
        problems.append(f"the last completed checkpoint is version {now['value']}, but the plan "
                        f"reaches {expected_final}: the run stopped early")
    else:
        try:
            ident = vc.verify(ckpt_dir / f"global_step_{expected_final}", require_train_checkpoint=True)
        except vc.CheckpointError as e:
            problems.append(f"the final checkpoint is not complete: {e}")
    abort = run_control.read_abort(abort_file) if abort_file else None
    if abort:
        problems.append(f"abort requested by {abort.get('source')}: {abort.get('reason')}")
    hard = hard_errors(log)
    if not problems and raw_rc != 0 and hard:
        problems.append(f"non-zero exit with a hard-failure signature in the log: {hard[0]}")
    certified = not problems
    return {
        "run_id": os.environ.get("RUN_ID"),
        "git_sha": os.environ.get("GIT_SHA"),
        "certified": certified,
        "final_rc": 0 if certified else (raw_rc if raw_rc != 0 else 1),
        "raw_rc": raw_rc,
        "expected_final_version": expected_final,
        "observed_version": now.get("value"),
        "problems": problems,
        "hard_error_matches": hard,
        "abort": abort,
        "checkpoint": ident,
        "ckpt_dir": str(ckpt_dir),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("ckpt_dir")
    c = sub.add_parser("check")
    c.add_argument("--ckpt-dir", required=True)
    c.add_argument("--expected-final", type=int, required=True)
    c.add_argument("--pre", required=True, help="the `snapshot` JSON taken before the run")
    c.add_argument("--raw-rc", type=int, required=True)
    c.add_argument("--log")
    c.add_argument("--abort-file")
    c.add_argument("--json-out", action="append", default=[])
    c.add_argument("--settle-s", type=float, default=120.0,
                   help="re-check this long while the tracker is not yet final (Volume FUSE lag "
                        "between the node that saved and this one)")
    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        print(json.dumps(snapshot(args.ckpt_dir)))
        return 0

    pre = json.loads(args.pre)
    deadline = time.monotonic() + args.settle_s
    while True:
        verdict = evaluate(args.ckpt_dir, args.expected_final, pre, args.raw_rc,
                           log=args.log, abort_file=args.abort_file)
        if verdict["certified"] or verdict["abort"] or time.monotonic() >= deadline:
            break
        time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))
    for out in args.json_out:
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(verdict, indent=2, default=str))
        except OSError as e:  # the verdict itself must still reach the log and the exit code
            print(f"[certificate] could not write {out}: {e}", file=sys.stderr)
    print(json.dumps(verdict, indent=2, default=str))
    if verdict["certified"]:
        print(f"[certificate] CERTIFIED: version {verdict['observed_version']} of "
              f"{verdict['expected_final_version']} saved and verified (raw exit {args.raw_rc} -> 0)")
    else:
        print(f"[certificate] NOT CERTIFIED -> exit {verdict['final_rc']}:", file=sys.stderr)
        for p in verdict["problems"]:
            print(f"  - {p}", file=sys.stderr)
    return int(verdict["final_rc"])


if __name__ == "__main__":
    sys.exit(main())
