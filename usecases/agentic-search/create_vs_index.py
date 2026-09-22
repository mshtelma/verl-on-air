#!/usr/bin/env python3
"""Load corpus.parquet into a UC Delta table and build the Vector Search Delta-Sync index.

Run AFTER build_corpus.py wrote corpus.parquet to a Volume. Designed to run IN-WORKSPACE (air/133)
so it uses ambient auth and can auto-discover a SQL warehouse; can also run from a CLI profile.

  1. CREATE TABLE <catalog.schema.table> (id, title, text) with Change Data Feed ON (required by VS
     Delta-Sync), then COPY INTO it from the Volume parquet -- via the SQL warehouse (no Spark).
  2. Create the Vector Search endpoint (if missing) and a DELTA_SYNC index with MANAGED embeddings
     (databricks-gte-large-en) on `text`, keyed by `id`, and wait until it is ONLINE.

Uses the `databricks-vectorsearch` client for endpoint/index (same client the runtime tool queries
with) and the `databricks-sdk` Statement Execution API for the table load.

NOTE: a Vector Search endpoint is a billable, persistent resource. Idempotent: re-running reuses the
endpoint/index and re-syncs.

CLI (all also QA_VS_* env):
    --catalog --schema --table   UC location for the source Delta table
    --endpoint                   VS endpoint name (STANDARD)
    --index                      full index name (default <catalog.schema.table>_index)
    --corpus                     Volume path to corpus.parquet
    --warehouse-id               SQL warehouse (auto-discovered in-workspace if omitted)
    --profile                    Databricks CLI profile (omit in-workspace -> ambient auth)
    --embedding-model            managed embedding endpoint (default databricks-gte-large-en)
    --skip-load                  only (re)build the index; skip table create/COPY INTO
"""
from __future__ import annotations

import argparse
import os
import time


# --- SQL side: databricks-sdk WorkspaceClient + Statement Execution -----------------------------
def _wsclient(profile: str | None):
    from databricks.sdk import WorkspaceClient

    if profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    return WorkspaceClient()


def _find_warehouse(w) -> str:
    """Pick a SQL warehouse when --warehouse-id is not given (prefer RUNNING, else first)."""
    whs = list(w.warehouses.list())
    if not whs:
        raise SystemExit("no SQL warehouse found; create one or pass --warehouse-id")
    running = [x for x in whs if str(getattr(x.state, "value", x.state)) == "RUNNING"]
    pick = (running or whs)[0]
    print(f"  auto-selected warehouse {pick.name!r} (id={pick.id}, state={pick.state})")
    return pick.id


def _sql(w, warehouse_id: str, statement: str) -> None:
    from databricks.sdk.service.sql import StatementState

    print(f"  SQL> {statement.splitlines()[0][:90]} ...")
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
    )
    sid = resp.statement_id
    state = resp.status.state if resp.status else None
    while state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(sid)
        state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        err = resp.status.error if resp.status else None
        raise SystemExit(f"SQL failed ({state}): {err}")


def _create_table(w, warehouse_id: str, table: str, corpus: str) -> None:
    _sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {table.rsplit('.', 1)[0]}")
    _sql(w, warehouse_id,
         f"CREATE TABLE IF NOT EXISTS {table} (id STRING, title STRING, text STRING) "
         f"TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    _sql(w, warehouse_id, f"TRUNCATE TABLE {table}")   # idempotent reload
    _sql(w, warehouse_id,
         f"COPY INTO {table} FROM '{corpus}' FILEFORMAT = PARQUET "
         f"COPY_OPTIONS ('force' = 'true', 'mergeSchema' = 'true')")
    print(f"  loaded {table} from {corpus}")


# --- Vector Search side: databricks-vectorsearch client -----------------------------------------
def _vsclient():
    from databricks.vector_search.client import VectorSearchClient

    host = os.environ.get("DATABRICKS_HOST") or None
    token = os.environ.get("DATABRICKS_TOKEN") or None
    if host and token:
        return VectorSearchClient(workspace_url=host, personal_access_token=token, disable_notice=True)
    return VectorSearchClient(disable_notice=True)  # ambient workspace auth


def _ensure_endpoint(vsc, endpoint: str) -> None:
    try:
        eps = (vsc.list_endpoints() or {}).get("endpoints", [])
        if any(e.get("name") == endpoint for e in eps):
            print(f"  endpoint {endpoint!r} exists")
            return
    except Exception:  # noqa: BLE001 - fall through to create
        pass
    print(f"  creating endpoint {endpoint!r} (STANDARD) ...")
    vsc.create_endpoint_and_wait(name=endpoint, endpoint_type="STANDARD")


def _ensure_index(vsc, endpoint: str, index: str, table: str, embedding_model: str, wait: bool = True) -> None:
    # create-only (wait=False) via create_delta_sync_index: for a big corpus the initial snapshot far
    # exceeds a single job window, so kick provisioning off here and poll to ONLINE in a --wait-only job.
    create = vsc.create_delta_sync_index_and_wait if wait else vsc.create_delta_sync_index
    try:
        print(f"  creating DELTA_SYNC index {index!r} on {table}.text ({embedding_model}) wait={wait} ...")
        kw = dict(
            endpoint_name=endpoint,
            index_name=index,
            primary_key="id",
            source_table_name=table,
            pipeline_type="TRIGGERED",
            embedding_source_column="text",
            embedding_model_endpoint_name=embedding_model,
        )
        if wait:
            kw["verbose"] = True
        create(**kw)
        print("  index created + ONLINE" if wait else "  index create kicked off (provisioning; poll with --wait-only)")
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "already exists" in msg or "resource_already_exists" in msg:
            print(f"  index {index!r} exists -> sync")
            idx = vsc.get_index(endpoint_name=endpoint, index_name=index)
            idx.sync()
        else:
            raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=os.environ.get("QA_VS_CATALOG", "main"))
    p.add_argument("--schema", default=os.environ.get("QA_VS_SCHEMA", "mshtelma"))
    p.add_argument("--table", default=os.environ.get("QA_VS_TABLE", "wiki_qa_corpus"))
    p.add_argument("--endpoint", default=os.environ.get("QA_VS_ENDPOINT", "wiki-qa-vs"))
    p.add_argument("--index", default=os.environ.get("QA_VS_INDEX", ""))
    p.add_argument("--corpus", default=os.environ.get("QA_CORPUS_OUT",
                   "/Volumes/main/mshtelma/verl/data/qa_search/corpus.parquet"))
    p.add_argument("--warehouse-id", default=os.environ.get("QA_VS_WAREHOUSE_ID", ""))
    p.add_argument("--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE", ""))
    p.add_argument("--embedding-model", default=os.environ.get("QA_VS_EMBED_MODEL", "databricks-gte-large-en"))
    p.add_argument("--skip-load", action="store_true")
    p.add_argument("--status-only", action="store_true", help="print the index state (detailed_state, "
                   "indexed_row_count) and exit -- a cheap progress check for a provisioning index")
    p.add_argument("--wait-only", action="store_true", help="only wait for the existing index to become "
                   "ONLINE, then exit (no table load, no create, no sync)")
    p.add_argument("--no-wait", action="store_true", help="create the index and return immediately "
                   "(do NOT block until ONLINE); poll separately with --wait-only. Use for big corpora.")
    args = p.parse_args()

    table = f"{args.catalog}.{args.schema}.{args.table}"
    index = args.index or f"{table}_index"
    print(f"table={table} endpoint={args.endpoint} index={index}")

    if args.status_only:
        import json as _json
        idx = _vsclient().get_index(endpoint_name=args.endpoint, index_name=index)
        st = (idx.describe() or {}).get("status", {}) or {}
        print(_json.dumps({k: st.get(k) for k in ("detailed_state", "ready", "indexed_row_count", "message")}, indent=2))
        return 0
    if args.wait_only:
        idx = _vsclient().get_index(endpoint_name=args.endpoint, index_name=index)
        idx.wait_until_ready(verbose=True)
        print(f"ONLINE: {index}")
        return 0

    if not args.skip_load:
        print("[1/3] load Delta table")
        w = _wsclient(args.profile or None)
        warehouse_id = args.warehouse_id or _find_warehouse(w)
        _create_table(w, warehouse_id, table, args.corpus)

    vsc = _vsclient()
    print("[2/3] ensure endpoint")
    _ensure_endpoint(vsc, args.endpoint)
    print(f"[3/3] ensure index (wait={not args.no_wait})")
    _ensure_index(vsc, args.endpoint, index, table, args.embedding_model, wait=not args.no_wait)
    print(f"\nDONE. For training/eval set:\n  QA_VS_ENDPOINT={args.endpoint}\n  QA_VS_INDEX={index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
