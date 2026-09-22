#!/usr/bin/env python3
"""Probe how to query Vector Search from inside a df1 job WITHOUT breaking the pinned image.

air/134 died because `pip install databricks-vectorsearch` downgraded protobuf (5.29.6) below what
the image's vLLM gencode (6.33.5) needs, so vLLM wouldn't start. This probe gathers the facts to fix
it in one shot:
  1. baseline protobuf version in the image (before any install);
  2. whether databricks-sdk / databricks-vectorsearch are preinstalled;
  3. whether ambient auth resolves a host (+token) in a df1 job;
  4. whether a Vector Search QUERY works over the REST API via the SDK's generic api_client.do()
     -- which avoids the databricks-vectorsearch client and its protobuf-pinning deps entirely;
  5. whether installing ONLY databricks-sdk perturbs protobuf (it should not);
  6. (last, so it can't taint the above) whether databricks-vectorsearch is what downgrades protobuf.

Prints a PROBE: line per fact. Run on an A10 via air/137.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _pb_version() -> str:
    try:
        import google.protobuf
        return google.protobuf.__version__
    except Exception as e:  # noqa: BLE001
        return f"<none: {e}>"


def _pip_show(pkg: str) -> str:
    r = subprocess.run([sys.executable, "-m", "pip", "show", pkg], capture_output=True, text=True)
    if r.returncode != 0:
        return "NOT INSTALLED"
    for line in r.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return "?"


def main() -> int:
    endpoint = os.environ.get("QA_VS_ENDPOINT", "wiki-qa-vs")
    index = os.environ.get("QA_VS_INDEX", "main.mshtelma.wiki_qa_corpus_index")

    print(f"PROBE: baseline protobuf = {_pb_version()}")
    print(f"PROBE: preinstalled databricks-sdk = {_pip_show('databricks-sdk')}")
    print(f"PROBE: preinstalled databricks-vectorsearch = {_pip_show('databricks-vectorsearch')}")
    for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID",
              "DATABRICKS_CLIENT_SECRET", "DATABRICKS_CONFIG_PROFILE"):
        print(f"PROBE: env {k} = {'<set>' if os.environ.get(k) else '<unset>'}")

    # Ensure databricks-sdk is importable (install if missing) and check protobuf AFTER.
    try:
        import databricks.sdk  # noqa: F401
        print("PROBE: databricks-sdk import = OK (preinstalled)")
    except Exception:  # noqa: BLE001
        print("PROBE: databricks-sdk not importable; pip installing it ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "databricks-sdk"], check=False)
        print(f"PROBE: protobuf AFTER installing databricks-sdk = {_pb_version()}  (want UNCHANGED)")

    # Ambient auth + REST query via the SDK's generic api client (no vectorsearch client).
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        print(f"PROBE: ambient host = {w.config.host!r}  token_present = {bool(w.config.token)}")
        body = {"num_results": 3, "columns": ["id", "title", "text"],
                "query_text": "who directed the film Inception", "query_type": "ANN"}
        path = f"/api/2.0/vector-search/indexes/{index}/query"
        try:
            resp = w.api_client.do("POST", path, body=body)
            da = (resp or {}).get("result", {}).get("data_array", []) or []
            cols = [c.get("name") for c in (resp or {}).get("manifest", {}).get("columns", [])]
            print(f"PROBE: REST ANN query OK -- rows={len(da)} cols={cols}")
            if da:
                row = da[0]
                print(f"PROBE: first row (truncated) = {json.dumps(row)[:240]}")
            # HYBRID (keyword) path too
            body["query_type"] = "HYBRID"
            resp2 = w.api_client.do("POST", path, body=body)
            da2 = (resp2 or {}).get("result", {}).get("data_array", []) or []
            print(f"PROBE: REST HYBRID query OK -- rows={len(da2)}")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE: REST query FAILED: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"PROBE: WorkspaceClient/auth FAILED: {type(e).__name__}: {e}")

    # LAST: confirm databricks-vectorsearch is the protobuf-downgrader (so we know to avoid it).
    print("PROBE: installing databricks-vectorsearch to observe its effect on protobuf ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "databricks-vectorsearch"], check=False)
    print(f"PROBE: protobuf AFTER installing databricks-vectorsearch = {_pb_version()}  (if < baseline, it is the culprit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
