#!/usr/bin/env python3
"""Does the published judge answer? Its served name is listed, and one real completion comes back.

    judge_ping.py <base_url> <served_name>        exit 0 = answering; 1 = not (the reason is printed)

Run by dispatch_agentic.sh on training rank 0 once the endpoint is published, before any training
step: it goes over the same cross-node HTTP path the reward will use, so a judge that is up but
unreachable from training, or listed but unable to generate, stops the job here instead of turning
every reward into a fallback. Retries transient failures for up to PING_TIMEOUT_S (default 300).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _call(url: str, body: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ping(base: str, name: str) -> str | None:
    """None when the judge answers, else why not."""
    base = base.rstrip("/")
    try:
        listed = [m.get("id") for m in _call(f"{base}/models").get("data") or []]
        if name not in listed:
            return f"served models are {listed}, not {name!r}"
        out = _call(f"{base}/chat/completions", {"model": name, "max_tokens": 8, "temperature": 0,
                                                  "messages": [{"role": "user", "content": "Reply with OK."}]})
        choice = (out.get("choices") or [{}])[0]
        if not isinstance(choice.get("message"), dict):
            return f"the completion has no message: {str(out)[:200]}"
        return None
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    base, name = argv[1], argv[2]
    deadline = time.monotonic() + float(os.environ.get("PING_TIMEOUT_S", "300"))
    while True:
        why = ping(base, name)
        if why is None:
            print(f"[judge-ping] {base} answers as {name!r}")
            return 0
        if time.monotonic() >= deadline:
            print(f"[judge-ping] FAIL: {base}: {why}", file=sys.stderr)
            return 1
        print(f"[judge-ping] not yet: {why}; retrying", flush=True)
        time.sleep(float(os.environ.get("PING_RETRY_S", "10")))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
