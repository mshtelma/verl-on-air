"""One machine-readable verdict per probe: a `PROBE_VERDICT {json}` log line (+ PROBE_VERDICT_OUT).

A GATE (smoke_test, probe_tool_format, probe_cross_node_http, probe_vs_access) exits non-zero
unless its claim holds, and says why in the verdict; an INFORMATIONAL diagnostic prints what it
found and is never read as a pass. Both kinds print the same line, so a reader -- or a script
grepping a job's log -- need not parse prose:

    PROBE_VERDICT {"schema": "voa.probe_verdict/v1", "probe": "...", "gate": true,
                   "ok": false, "status": "FAIL", "reasons": [...], ...}

PROBE_VERDICT_OUT=<path> also writes the JSON there (atomically). Stdlib only: the probes run in
images and on hosts that may have nothing else.
"""
from __future__ import annotations

import json
import os
from typing import Any

SCHEMA = "voa.probe_verdict/v1"


def verdict(probe: str, ok: bool, *, gate: bool = True, status: str | None = None,
            reasons: list[str] | None = None, **detail: Any) -> dict[str, Any]:
    return {"schema": SCHEMA, "probe": probe, "gate": gate, "ok": bool(ok),
            "status": status or ("PASS" if ok else "FAIL"), "reasons": list(reasons or []), **detail}


def emit(probe: str, ok: bool, **kw: Any) -> int:
    """Print (and optionally write) the verdict; return the exit code a gate must use."""
    v = verdict(probe, ok, **kw)
    line = json.dumps(v, sort_keys=True, default=str)
    print(f"PROBE_VERDICT {line}", flush=True)
    out = os.environ.get("PROBE_VERDICT_OUT")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        tmp = f"{out}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(line + "\n")
        os.replace(tmp, out)
    return 0 if v["ok"] else 1


def parse(log_text: str) -> dict[str, Any] | None:
    """The last verdict in a log (None if the probe never reached one -- itself a failure)."""
    found = None
    for ln in log_text.splitlines():
        if ln.startswith("PROBE_VERDICT "):
            found = json.loads(ln[len("PROBE_VERDICT "):])
    return found
