#!/usr/bin/env python3
"""GATE: can this environment query the Vector Search index the tools use? Read-only.

    QA_VS_INDEX=main.<schema>.<index> python3 usecases/agentic-search/probe_vs_access.py

It resolves workspace auth the way tool.py does (DATABRICKS_HOST/DATABRICKS_TOKEN if set, else
ambient auth -- inside a job in the workspace, no token is needed), then runs one ANN and one
HYBRID query over the index's REST API (the SDK's generic api_client.do(), which is how tool.py
avoids the databricks-vectorsearch client and its protobuf pin). Exit 0 only if both queries
return rows with the id/title/text columns; any auth, query or shape failure exits 1. The last
line is a machine-readable `PROBE_VERDICT {...}`.

It installs NOTHING and changes nothing: an earlier version pip-installed databricks-sdk and
databricks-vectorsearch into the active environment to watch protobuf move, which could break
the very environment it was run from (a vectorsearch install downgrades protobuf below vLLM's
gencode). It now only REPORTS the protobuf and client versions that are present; try a
dependency change in a disposable venv or job, never from a connectivity check.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

QUERY = "who directed the film Inception"
COLUMNS = ["id", "title", "text"]


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _default_client():
    from databricks.sdk import WorkspaceClient
    host, token = os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
    return WorkspaceClient(host=host, token=token) if host and token else WorkspaceClient()


def probe(index: str, client_factory: Callable[[], Any] = _default_client) -> dict[str, Any]:
    """-> the verdict fields: ok, reasons, and what was observed."""
    facts: dict[str, Any] = {"index": index, "versions": {p: _version(p) for p in
                             ("protobuf", "databricks-sdk", "databricks-vectorsearch")}}
    if not index:
        return {"ok": False, "reasons": ["QA_VS_INDEX is not set"], **facts}
    try:
        client = client_factory()
        facts["host"] = getattr(getattr(client, "config", None), "host", None)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reasons": [f"auth: {type(e).__name__}: {e}"], **facts}
    reasons = []
    path = f"/api/2.0/vector-search/indexes/{index}/query"
    for qtype in ("ANN", "HYBRID"):
        body = {"num_results": 3, "columns": COLUMNS, "query_text": QUERY, "query_type": qtype}
        try:
            resp = client.api_client.do("POST", path, body=body) or {}
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{qtype} query: {type(e).__name__}: {str(e)[:300]}")
            continue
        rows = (resp.get("result") or {}).get("data_array") or []
        cols = [c.get("name") for c in (resp.get("manifest") or {}).get("columns") or []]
        facts[qtype] = {"rows": len(rows), "columns": cols,
                        "first_row": json.dumps(rows[0])[:240] if rows else None}
        if not rows:
            reasons.append(f"{qtype} query returned no rows")
        missing = [c for c in COLUMNS if c not in cols]
        if missing:
            reasons.append(f"{qtype} query lacks columns {missing}")
    return {"ok": not reasons, "reasons": reasons, **facts}


def _emit(probe_name: str, ok: bool, **kw: Any) -> int:
    """infra/diagnostics/probe_verdict.emit when it is there (a checkout); the same line otherwise
    (a job snapshot of this use case ships no infra/)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra" / "diagnostics"))
    try:
        import probe_verdict
    except ImportError:
        v = {"schema": "voa.probe_verdict/v1", "probe": probe_name, "gate": True, "ok": bool(ok),
             "status": "PASS" if ok else "FAIL", **kw}
        print(f"PROBE_VERDICT {json.dumps(v, sort_keys=True, default=str)}", flush=True)
        return 0 if ok else 1
    return probe_verdict.emit(probe_name, ok, **kw)


def main(argv: list[str] | None = None, client_factory: Callable[[], Any] = _default_client) -> int:
    index = os.environ.get("QA_VS_INDEX", "")
    v = probe(index, client_factory)
    for k in ("index", "host", "versions", "ANN", "HYBRID"):
        if k in v:
            print(f"PROBE: {k} = {v[k]}", flush=True)
    for r in v["reasons"]:
        print(f"PROBE: FAIL {r}", flush=True)
    ok = v.pop("ok")
    return _emit("probe_vs_access", ok, **v)


if __name__ == "__main__":
    raise SystemExit(main())
