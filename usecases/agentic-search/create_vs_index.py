#!/usr/bin/env python3
"""Load a built corpus into a versioned UC Delta table and build its Vector Search index.

Run AFTER build_corpus.py. Every name is derived from the corpus CONTENT:

    table  <catalog>.<schema>.<table>_v<h8>      h8 = the manifest's content_sha256[:8]
    index  <that table>_index

so a rebuilt corpus gets a NEW table and a NEW index, and nothing a running job reads is ever
modified: there is no TRUNCATE, no reload and no re-sync of an existing table. A new table is loaded
under a staging name, verified, and only then renamed into place, so the versioned name either
holds the complete corpus or does not exist. An existing table or index is reused only if it
verifies -- columns, Change Data Feed, the corpus hash in its properties and the row count; the
index's endpoint, source table, key, embedding column and model -- and is otherwise left untouched
and reported. Nothing is ever dropped: old versions are removed deliberately (`make cleanup-vs`).

  1. table     CREATE <staging>, COPY INTO, verify, ALTER TABLE ... RENAME TO <table>
  2. endpoint  must exist. A new one is billable and persistent, so it is created only with
               --create-endpoint: a typo in the name must not quietly start paying for one.
  3. index     DELTA_SYNC, managed embeddings on `text`, keyed by `id`. Ready means ONLINE with
               indexed_row_count == the corpus's passage count, i.e. THIS snapshot is indexed.

One authenticated client serves every step and every mode: --profile, else ambient auth (inside a
job). SQL runs on the warehouse you name (--warehouse-id; none is guessed) against a deadline, and a
statement that outlives it is cancelled. The Vector Search calls use the REST API through the same
databricks-sdk client the search tool queries with, so nothing is pip-installed at run time.

    create_vs_index.py --warehouse-id <id> [--no-wait]    # build (the 2_build_index.yaml job)
    create_vs_index.py --status-only [--index <name>]      # print the index state
    create_vs_index.py --wait-only [--index <name>]        # block until ready, up to --wait-timeout-s

Exit codes: 0 done / ready; 1 refused or failed; 3 not ready yet (provisioning continues server-side).

CLI (also QA_VS_* env): --catalog --schema --table (base name, before _v<h8>) --endpoint --corpus
--warehouse-id --profile --embedding-model; --index names an index explicitly for --status-only /
--wait-only (e.g. one built before versioning); --allow-partial indexes a corpus whose manifest
marks it incomplete.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

COLUMNS = [("id", "string"), ("title", "string"), ("text", "string")]
NOT_READY = 3
POLL_S = 30
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENDPOINT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_VOLUME_PATH = re.compile(r"/Volumes/[A-Za-z0-9_./-]+\.parquet")


# --- one client for everything ---------------------------------------------------------------------
def connect(profile: str | None):
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient(profile=profile) if profile else WorkspaceClient()


def _state(resp) -> str:
    s = resp.status.state if resp.status else None
    return str(getattr(s, "value", s))


class Workspace:
    """SQL (Statement Execution, on a named warehouse, with a deadline) and REST on one client."""

    def __init__(self, client, warehouse_id: str | None, sql_timeout_s: int):
        self.client, self.warehouse_id, self.sql_timeout_s = client, warehouse_id, sql_timeout_s

    def sql(self, statement: str) -> list[list[Any]]:
        if not self.warehouse_id:
            raise SystemExit("--warehouse-id (QA_VS_WAREHOUSE_ID) is required: name the SQL warehouse "
                             "that loads the table (`databricks warehouses list -p <profile>`)")
        print(f"  SQL> {statement.splitlines()[0][:110]}", flush=True)
        se = self.client.statement_execution
        deadline = time.monotonic() + self.sql_timeout_s
        resp = se.execute_statement(warehouse_id=self.warehouse_id, statement=statement, wait_timeout="30s")
        while _state(resp) in ("PENDING", "RUNNING"):
            if time.monotonic() >= deadline:
                se.cancel_execution(resp.statement_id)
                raise SystemExit(f"SQL still running after {self.sql_timeout_s}s -- cancelled: {statement[:110]}")
            time.sleep(5)
            resp = se.get_statement(resp.statement_id)
        if _state(resp) != "SUCCEEDED":
            err = resp.status.error if resp.status else None
            raise SystemExit(f"SQL {_state(resp)}: {getattr(err, 'message', err)} -- {statement[:110]}")
        return list(getattr(resp.result, "data_array", None) or [])

    def get(self, path: str) -> dict | None:
        try:
            return self.client.api_client.do("GET", path)
        except Exception as e:  # noqa: BLE001 - only "does not exist" is an answer; the rest propagates
            if type(e).__name__ in ("NotFound", "ResourceDoesNotExist") or \
                    getattr(e, "error_code", None) in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
                return None
            raise

    def post(self, path: str, body: dict) -> dict:
        return self.client.api_client.do("POST", path, body=body)


def _q(name: str) -> str:
    return ".".join(f"`{part}`" for part in name.split("."))


# --- 0. the corpus: exactly the file its manifest describes ----------------------------------------
def corpus_identity(corpus: str, allow_partial: bool) -> dict:
    if not _VOLUME_PATH.fullmatch(corpus):
        raise SystemExit(f"--corpus must be a /Volumes/.../<name>.parquet path of plain characters, got {corpus!r}")
    found = dm.find_manifest(corpus)
    if found is None:
        raise SystemExit(f"no manifest lists {corpus}: build it with build_corpus.py, which records its "
                         f"sources and content hash (the index version is named from it)")
    mpath, doc, entry = found
    if dm.sha256_file(corpus) != entry.get("sha256"):
        raise SystemExit(f"{corpus} is not the file {mpath} describes (sha256 differs) -- rebuild the corpus")
    if not doc.get("complete", False) and not allow_partial:
        failed = [f.get("name") for f in doc.get("failed_sources") or []]
        raise SystemExit(f"{mpath} marks the corpus INCOMPLETE (failed sources: {failed}). Rebuild it, or "
                         f"pass --allow-partial to index it knowingly.")
    return {"content_sha256": entry["content_sha256"], "rows": int(entry["rows"]), "manifest": str(mpath),
            "complete": bool(doc.get("complete", False))}


# --- 1. the table ------------------------------------------------------------------------------------
def verify_table(ws: Workspace, table: str, ident: dict) -> list[str]:
    """Why `table` does not hold exactly this corpus ([] = it does)."""
    problems = []
    cols = []
    for r in ws.sql(f"DESCRIBE TABLE {_q(table)}"):
        if not r or not r[0] or str(r[0]).startswith("#"):
            break
        cols.append((r[0], str(r[1] or "").lower()))
    if cols != COLUMNS:
        problems.append(f"columns {cols} != {COLUMNS}")
    props = {r[0]: r[1] for r in ws.sql(f"SHOW TBLPROPERTIES {_q(table)}")}
    if str(props.get("delta.enableChangeDataFeed")).lower() != "true":
        problems.append("Change Data Feed is off (Delta-Sync needs it)")
    if props.get("voa.corpus_sha256") != ident["content_sha256"]:
        problems.append(f"it was built from corpus {props.get('voa.corpus_sha256')!r}, "
                        f"not {ident['content_sha256']!r}")
    n, distinct = (int(x) for x in ws.sql(f"SELECT count(*), count(DISTINCT id) FROM {_q(table)}")[0])
    if n != ident["rows"] or distinct != n:
        problems.append(f"{n} rows / {distinct} distinct ids, expected {ident['rows']} unique")
    return problems


def ensure_table(ws: Workspace, table: str, corpus: str, ident: dict, run_id: str) -> str:
    catalog, schema, name = table.split(".")
    exists = ws.sql(f"SELECT table_name FROM {_q(catalog)}.information_schema.tables "
                    f"WHERE table_schema = '{schema}' AND table_name = '{name.lower()}'")
    if exists:
        problems = verify_table(ws, table, ident)
        if problems:
            raise SystemExit(f"{table} exists but does not hold this corpus: {'; '.join(problems)}. It was "
                             f"left untouched; drop it deliberately (if voa.created_by is verl-on-air) "
                             f"and re-run.")
        print(f"  {table} exists and verifies -> reused, not reloaded")
        return "reused"
    staging = f"{table}__staging_{re.sub(r'[^A-Za-z0-9_]', '_', run_id)}"
    props = {"delta.enableChangeDataFeed": "true", "voa.created_by": "verl-on-air",
             "voa.corpus_sha256": ident["content_sha256"], "voa.corpus_manifest": ident["manifest"],
             "voa.run_id": run_id}
    if any("'" in v for v in props.values()):
        raise SystemExit(f"a table property contains a quote: {props}")
    # The staging table is this run's own scratch (it carries RUN_ID and never has an index), so
    # replacing a leftover from an interrupted attempt of the same run is safe.
    ws.sql(f"CREATE OR REPLACE TABLE {_q(staging)} (id STRING NOT NULL, title STRING, text STRING) "
           f"TBLPROPERTIES ({', '.join(f'{k!r} = {v!r}' for k, v in props.items())})")
    ws.sql(f"COPY INTO {_q(staging)} FROM (SELECT id, title, text FROM '{corpus}') FILEFORMAT = PARQUET")
    problems = verify_table(ws, staging, ident)
    if problems:
        raise SystemExit(f"the load into {staging} is wrong: {'; '.join(problems)}. {table} was not "
                         f"created; the staging table is left for inspection.")
    ws.sql(f"ALTER TABLE {_q(staging)} RENAME TO {_q(table)}")
    print(f"  loaded {ident['rows']} passages -> {table}")
    return "created"


# --- 2. the endpoint ---------------------------------------------------------------------------------
def ensure_endpoint(ws: Workspace, endpoint: str, create: bool, timeout_s: int) -> bool:
    """True once the endpoint is ONLINE; False if it is still provisioning at the deadline."""
    path = f"/api/2.0/vector-search/endpoints/{endpoint}"
    ep = ws.get(path)
    if ep is None:
        if not create:
            raise SystemExit(f"Vector Search endpoint {endpoint!r} does not exist. A new endpoint is billable "
                             f"and persistent: fix the name, or create it with --create-endpoint "
                             f"(QA_VS_CREATE_ENDPOINT=1).")
        print(f"  creating endpoint {endpoint!r} (STANDARD) ...")
        ws.post("/api/2.0/vector-search/endpoints", {"name": endpoint, "endpoint_type": "STANDARD"})
    deadline = time.monotonic() + timeout_s
    while True:
        state = str(((ws.get(path) or {}).get("endpoint_status") or {}).get("state"))
        print(f"  endpoint {endpoint!r}: {state}")
        if state == "ONLINE":
            return True
        if state not in ("PROVISIONING", "None"):
            raise SystemExit(f"endpoint {endpoint!r} is {state}")
        if time.monotonic() + POLL_S > deadline:
            return False
        time.sleep(POLL_S)


# --- 3. the index ------------------------------------------------------------------------------------
def index_mismatch(cur: dict, endpoint: str, table: str, model: str) -> list[str]:
    spec = cur.get("delta_sync_index_spec") or {}
    want = {"endpoint_name": endpoint, "index_type": "DELTA_SYNC", "primary_key": "id", "source_table": table,
            "embedding": [["text", model]]}
    got = {"endpoint_name": cur.get("endpoint_name"), "index_type": cur.get("index_type"),
           "primary_key": cur.get("primary_key"), "source_table": spec.get("source_table"),
           "embedding": [[c.get("name"), c.get("embedding_model_endpoint_name")]
                         for c in spec.get("embedding_source_columns") or []]}
    return [f"{k} {got[k]!r} != {want[k]!r}" for k in want if got[k] != want[k]]


def ensure_index(ws: Workspace, index: str, endpoint: str, table: str, model: str) -> str:
    cur = ws.get(f"/api/2.0/vector-search/indexes/{index}")
    if cur is not None:
        problems = index_mismatch(cur, endpoint, table, model)
        if problems:
            raise SystemExit(f"index {index} exists but is not this index: {'; '.join(problems)}. "
                             f"Left untouched.")
        print(f"  index {index} exists and matches -> reused")
        return "reused"
    print(f"  creating DELTA_SYNC index {index} on {table}.text ({model}) ...")
    ws.post("/api/2.0/vector-search/indexes", {
        "name": index, "endpoint_name": endpoint, "primary_key": "id", "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": table, "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [{"name": "text", "embedding_model_endpoint_name": model}]}})
    return "created"


def index_state(cur: dict, expected_rows: int | None) -> tuple[str, dict]:
    """-> ("ready" | "pending" | "failed", a status summary). Ready = ONLINE AND every passage indexed:
    serving an older or partial snapshot is not "this corpus is searchable"."""
    st = cur.get("status") or {}
    detail = str(st.get("detailed_state") or "")
    summary = {"index": cur.get("name"), "detailed_state": detail, "ready": st.get("ready"),
               "indexed_row_count": st.get("indexed_row_count"), "expected_rows": expected_rows,
               "message": st.get("message")}
    if "FAILED" in detail:
        return "failed", summary
    online = detail.startswith("ONLINE") if detail else True   # typed SDK views drop detailed_state
    if st.get("ready") and online and \
            (expected_rows is None or st.get("indexed_row_count") == expected_rows):
        return "ready", summary
    return "pending", summary


def wait_index(ws: Workspace, index: str, expected_rows: int | None, timeout_s: int) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        cur = ws.get(f"/api/2.0/vector-search/indexes/{index}")
        if cur is None:
            raise SystemExit(f"index {index} does not exist")
        state, summary = index_state(cur, expected_rows)
        print(f"  {json.dumps(summary)}", flush=True)
        if state == "ready":
            return 0
        if state == "failed":
            raise SystemExit(f"index {index} FAILED: {summary['message']}")
        if time.monotonic() + POLL_S > deadline:
            print(f"  not ready after {timeout_s}s -- it keeps provisioning server-side; check again with "
                  f"--status-only or --wait-only")
            return NOT_READY
        time.sleep(POLL_S)


def main(argv: list[str] | None = None, client=None) -> int:
    env = os.environ.get
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=env("QA_VS_CATALOG", "main"))
    p.add_argument("--schema", default=env("QA_VS_SCHEMA", "mshtelma"))
    p.add_argument("--table", default=env("QA_VS_TABLE", "wiki_qa_big_corpus"), help="base name; _v<h8> is appended")
    p.add_argument("--endpoint", default=env("QA_VS_ENDPOINT", "wiki-qa-vs"))
    p.add_argument("--corpus", default=env("QA_CORPUS_OUT", "/Volumes/main/mshtelma/verl/data/qa_musique/corpus_big.parquet"))
    p.add_argument("--warehouse-id", default=env("QA_VS_WAREHOUSE_ID", ""))
    p.add_argument("--profile", default=env("DATABRICKS_CONFIG_PROFILE", ""))
    p.add_argument("--embedding-model", default=env("QA_VS_EMBED_MODEL", "databricks-gte-large-en"))
    p.add_argument("--create-endpoint", action="store_true", default=env("QA_VS_CREATE_ENDPOINT", "0") == "1",
                   help="create the endpoint if it does not exist (billable, persistent)")
    p.add_argument("--allow-partial", action="store_true", help="index a corpus its manifest marks incomplete")
    p.add_argument("--index", default="", help="with --status-only/--wait-only: an explicit index name")
    p.add_argument("--status-only", action="store_true", help="print the index state; exit 0 ready, 3 not yet")
    p.add_argument("--wait-only", action="store_true", help="wait for the index to be ready; no other change")
    p.add_argument("--no-wait", action="store_true", help="create the index and return; poll with --wait-only")
    p.add_argument("--sql-timeout-s", type=int, default=int(env("QA_VS_SQL_TIMEOUT_S", "1800")))
    p.add_argument("--wait-timeout-s", type=int, default=int(env("QA_VS_WAIT_TIMEOUT_S", "3000")))
    args = p.parse_args(argv)

    for what, part in (("--catalog", args.catalog), ("--schema", args.schema), ("--table", args.table)):
        if not _IDENT.fullmatch(part or ""):
            raise SystemExit(f"{what} {part!r} is not a plain identifier ([A-Za-z_][A-Za-z0-9_]*)")
    if not _ENDPOINT.fullmatch(args.endpoint or ""):
        raise SystemExit(f"--endpoint {args.endpoint!r} is not a valid endpoint name")
    if args.index and not (args.status_only or args.wait_only):
        raise SystemExit("--index is for --status-only / --wait-only; a build names its table and index "
                         "from the corpus content")
    if args.index and not all(_IDENT.fullmatch(x) for x in args.index.split(".")) or \
            args.index.count(".") not in (0, 2):
        raise SystemExit(f"--index {args.index!r} is not <catalog>.<schema>.<name>")

    if args.index:
        index, ident = args.index, None
    else:
        ident = corpus_identity(args.corpus, args.allow_partial)
        table = f"{args.catalog}.{args.schema}.{args.table}_v{ident['content_sha256'][:8]}"
        index = f"{table}_index"
    ws = Workspace(client if client is not None else connect(args.profile or None),
                   args.warehouse_id or None, args.sql_timeout_s)
    expected = ident["rows"] if ident else None

    if args.status_only:
        cur = ws.get(f"/api/2.0/vector-search/indexes/{index}")
        if cur is None:
            print(f"index {index} does not exist")
            return 1
        state, summary = index_state(cur, expected)
        print(json.dumps(summary, indent=2))
        return {"ready": 0, "pending": NOT_READY, "failed": 1}[state]
    if args.wait_only:
        return wait_index(ws, index, expected, args.wait_timeout_s)

    run_id = env("RUN_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"corpus {args.corpus}: {ident['rows']} passages, content {ident['content_sha256'][:12]}"
          f"{'' if ident['complete'] else ' (INCOMPLETE, --allow-partial)'}\n"
          f"table={table}\nindex={index}\nendpoint={args.endpoint}", flush=True)
    print("[1/3] table")
    ensure_table(ws, table, args.corpus, ident, run_id)
    print("[2/3] endpoint")
    if not ensure_endpoint(ws, args.endpoint, args.create_endpoint, args.wait_timeout_s):
        print(f"  endpoint {args.endpoint!r} is still provisioning; re-run this build once it is ONLINE")
        return NOT_READY
    print("[3/3] index")
    ensure_index(ws, index, args.endpoint, table, args.embedding_model)
    print(f"\nFor training/eval set:\n  QA_VS_ENDPOINT={args.endpoint}\n  QA_VS_INDEX={index}", flush=True)
    if args.no_wait:
        print(f"Returning before the index is ready (--no-wait). It is ready when ONLINE with "
              f"indexed_row_count == {ident['rows']}: re-run with --status-only or --wait-only.")
        return 0
    return wait_index(ws, index, expected, args.wait_timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
