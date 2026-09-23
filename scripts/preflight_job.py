#!/usr/bin/env python3
"""The plan of one job file, validated on this machine before any GPU is allocated.

    make preflight F=usecases/agentic-search/air/4_train.yaml      (prints JSON; exit 1 if invalid)

A training job is rendered exactly as rank 0 of the job would run it (scripts/compose_check.py:
the job's own command, DRY_RUN=1), so both launcher preflight phases run -- every knob typed and
applicable to its mode, the geometry checked against the model (engine/lib/preflight.py) -- and
their plan is reported: roles, parallelism, budget, checkpoints. Any job gets what only its file
knows: the air profile, the image, the GPUs and the timeout. GPUs x timeout x (1 + max_retries) is
the hard upper bound on the GPU-hours the job can bill, since air stops it at its timeout.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_check as cc  # noqa: E402


def profile() -> str | None:
    m = re.search(r"^AIR_PROFILE=(\S+)", (REPO / "config.env").read_text(), re.M)
    return m.group(1) if m else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: preflight_job.py <job.yaml>", file=sys.stderr)
        return 2
    job = Path(argv[1]).resolve()
    spec = yaml.safe_load(job.read_text())
    gpus = int(spec["compute"]["num_accelerators"])
    timeout = spec.get("timeout_minutes")
    retries = int(spec.get("max_retries") or 0)
    out = {"job": str(job.relative_to(REPO)), "profile": profile(),
           "image": ((spec.get("environment") or {}).get("docker_image") or {}).get("url"),
           "gpus": gpus, "accelerator": spec["compute"].get("accelerator_type"),
           "timeout_minutes": timeout, "max_retries": retries,
           "gpu_hours_upper_bound": round(gpus * timeout / 60 * (1 + retries), 1) if timeout else None}
    if job in {p.resolve() for p in cc.training_jobs()}:
        with tempfile.TemporaryDirectory(prefix="preflight-") as tmp:
            r = cc.render(job, Path(tmp), 29900, {"RUN_ID": os.environ.get("RUN_ID") or cc.RUN_ID})
        lines = [ln for ln in r["stdout"].splitlines() if ln.startswith("PREFLIGHT_PLAN ")]
        if r["returncode"] != 0 or not lines:
            fatal = [ln for ln in (r["stderr"] + r["stdout"]).splitlines() if "FATAL" in ln]
            print("\n".join(fatal) or (r["stderr"] + r["stdout"])[-2000:], file=sys.stderr)
            print(f"preflight: {out['job']} is NOT runnable as written", file=sys.stderr)
            return 1
        out["plan"] = json.loads(lines[-1].split(" ", 1)[1])
        out["verl_overrides"] = len(r["overrides"])
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
