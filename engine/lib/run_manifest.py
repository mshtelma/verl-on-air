#!/usr/bin/env python3
"""Record, at the START of a training run, exactly what is about to run: <run_dir>/run_manifest.json.

    run_manifest.py <out.json> [--kv KEY=VALUE ...] -- <verl overrides...>

It holds the run's identity (RUN_ID, the commit it was submitted from, the image), the resolved
verl overrides the launcher is about to pass, the air `parameters:` block, the engine/use-case
knobs from the environment (anything credential-like is left out), and for the train and val
files their sha256 plus what their data manifest says about how they were built
(engine/lib/data_manifest.py). run_certificate.py writes the
matching run_result.json at the end. A resumed run (RESUME != never) keeps the first manifest and
adds run_manifest.resume-<time>.json next to it.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_manifest  # noqa: E402

KNOB = re.compile(r"^(TRAIN_MODE|TRAINING_NODES|ROLLOUT_\w+|TP|PP|CP|EP|ETP|GEN_TP|OFFLOAD\w*|MEGATRON_MODE|"
                  r"TRIGGER_SYNC_STEP|REQUIRE_BATCHES|STALENESS|PARTIAL_ROLLOUT|MULTI_TURN|MAX_TURNS|TOOL_FORMAT|"
                  r"AGENT_\w+|MAX_TOOL_RESPONSE_LEN|FUNCTION_TOOL_PATH|TOOL_CONFIG_PATH|CUSTOM_REWARD_\w+|"
                  r"REWARD_\w+|QA_\w+|JUDGE_\w+|SAVE_FREQ|TEST_FREQ|RESUME|MAX_CKPT_TO_KEEP|NORM_ADV_\w+|"
                  r"PROJECT_NAME|EXPERIMENT_NAME|RUN_ID|GIT_SHA|VOA_IMAGE|NUM_NODES|LOCAL_WORLD_SIZE|DATA_\w+|"
                  r"PRE_TRAIN_CHECK|MAX_MODEL_LEN|LR_DECAY_STEPS)$")
SECRETISH = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL", re.I)


def data_provenance(overrides: list[str]) -> dict:
    out = {}
    for o in overrides:
        key, _, val = o.lstrip("+").partition("=")
        val = val.strip("'\"")
        if key in ("data.train_files", "data.val_files"):
            try:
                out[key] = data_manifest.provenance(val)
            except OSError as e:   # a list, or a path this node cannot read: say so, don't guess
                out[key] = {"path": val, "error": f"{type(e).__name__}: {e}"}
    return out


def build(overrides: list[str], kv: dict[str, str]) -> dict:
    params = None
    hp = os.environ.get("HYPERPARAMETERS_PATH")
    if hp and Path(hp).is_file():
        try:
            import yaml
            params = yaml.safe_load(Path(hp).read_text())
        except Exception:  # noqa: BLE001 - record the raw text rather than nothing
            params = Path(hp).read_text()
    return {
        "run_id": os.environ.get("RUN_ID"),
        "git_sha": os.environ.get("GIT_SHA"),
        "image": os.environ.get("VOA_IMAGE"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        **kv,
        "parameters": params,
        "knobs": {k: v for k, v in sorted(os.environ.items()) if KNOB.match(k) and not SECRETISH.search(k)},
        "data": data_provenance(overrides),
        "verl_overrides": overrides,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    out = Path(argv[1])
    rest = argv[2:]
    split = rest.index("--") if "--" in rest else len(rest)
    kv = dict(a.split("=", 1) for a in rest[:split] if "=" in a and a != "--kv")
    overrides = rest[split + 1:]
    if out.exists():
        out = out.with_name(f"run_manifest.resume-{time.strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(build(overrides, kv), indent=2, default=str))
    os.replace(tmp, out)
    print(f"[manifest] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
