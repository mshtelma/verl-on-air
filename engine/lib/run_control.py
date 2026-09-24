#!/usr/bin/env python3
"""The run's abort channel: one file any process can raise, one watchdog that acts on it.

verl swallows exceptions on the paths that matter here. The fully-async rollouter gathers its tasks
with ``return_exceptions=True`` and then sends a normal stop signal; the ``rate_limited`` reward
manager turns any exception into a 0.0 reward. So a component that detects a condition the run must
not continue under -- a judge failure budget exhausted, reward provenance missing -- cannot rely on
raising. It calls ``request_abort()``: an atomic JSON write to ``<rendezvous>/ABORT.json`` on the
shared Volume. The training launcher's watchdog (rank 0) polls for that file and stops the run, and
the exit guard (run_certificate.py) treats its presence as a veto on success.

Location: ``VOA_RDV_DIR`` if set (the dispatcher exports it), else rebuilt from container-level
variables that reach every process on every node -- including Ray actors, which do not reliably
inherit the driver's exports: ``RENDEZVOUS_ROOT`` + ``RUN_ID`` (make sets one per submission), else
``RENDEZVOUS_ROOT`` + ``MASTER_ADDR``_``MASTER_PORT`` -- the same rule as dispatch_agentic.sh.

    python3 run_control.py abort <source> <reason...>   # raise it from a shell component
    python3 run_control.py show                         # print the request, exit 1 if none
    python3 run_control.py path                         # where it lives, exit 1 if unconfigured
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

ABORT_NAME = "ABORT.json"


def rendezvous_dir() -> Path | None:
    explicit = os.environ.get("VOA_RDV_DIR")
    if explicit:
        return Path(explicit)
    root, addr, port = (os.environ.get(k) for k in ("RENDEZVOUS_ROOT", "MASTER_ADDR", "MASTER_PORT"))
    if root and os.environ.get("RUN_ID"):
        return Path(root) / os.environ["RUN_ID"]
    if root and addr and port:
        return Path(root) / f"{addr}_{port}"
    return None


def abort_path() -> Path | None:
    d = rendezvous_dir()
    return d / ABORT_NAME if d else None


def request_abort(reason: str, source: str, **details: Any) -> bool:
    """Ask the run to stop. The first request wins; later ones leave the file untouched. Returns
    False if no rendezvous is configured -- the caller must then fail as loudly as it can alone."""
    p = abort_path()
    if p is None:
        print(f"[abort] NO RENDEZVOUS configured, cannot signal the run: {source}: {reason}",
              file=sys.stderr, flush=True)
        return False
    if p.exists():
        return True
    payload = {"reason": reason, "source": source, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "host": socket.gethostname(), "pid": os.getpid(), "details": details}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{ABORT_NAME}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, p)  # closed + renamed: visible to the other nodes as a whole file
    print(f"[abort] requested by {source}: {reason} -> {p}", file=sys.stderr, flush=True)
    return True


def read_abort(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path else abort_path()
    if p is None or not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"reason": "unreadable abort request", "source": str(p)}
    return data if isinstance(data, dict) else {"reason": str(data), "source": str(p)}


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "abort":
        return 0 if request_abort(" ".join(argv[3:]), argv[2]) else 1
    if len(argv) == 2 and argv[1] == "show":
        req = read_abort()
        if req is None:
            return 1
        print(json.dumps(req, indent=2))
        return 0
    if len(argv) == 2 and argv[1] == "path":
        p = abort_path()
        if p is None:
            return 1
        print(p)
        return 0
    print("usage: run_control.py abort <source> <reason...> | show | path", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
