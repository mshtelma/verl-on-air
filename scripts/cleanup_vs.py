#!/usr/bin/env python3
"""Delete the Vector Search indexes and Delta tables THIS template created, and nothing else.

    make cleanup-vs WAREHOUSE_ID=<id> [KEEP=<index>,...] [CONFIRM=1]
    python3 scripts/cleanup_vs.py --endpoint E --warehouse-id W [--keep ...] [--confirm] [--profile df1]

Ownership is read, not guessed: create_vs_index.py sets the table property
voa.created_by = verl-on-air on every table it loads, so an index is a candidate only when its source
table carries it, and a staging table (`<table>__staging_<run>`, never indexed) only when it does
too. KEPT, always: the index config.env's VS_INDEX names (what the jobs query), every --keep, and
anything whose table lacks the property -- including the index the published runs used, which
predates the property. Vector Search endpoints are never deleted (other indexes may live on them).
Without --confirm it prints the plan. Uses the `databricks` CLI (REST + SQL statements).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2}$")
OWNER = ("voa.created_by", "verl-on-air")


def api(profile: str, method: str, path: str, body: dict | None = None) -> dict:
    cmd = ["databricks", "api", method, path, "-p", profile]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"databricks api {method} {path}: {(r.stderr or r.stdout).strip()[:300]}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def sql(profile: str, warehouse: str, statement: str) -> list[list]:
    resp = api(profile, "post", "/api/2.0/sql/statements",
               {"warehouse_id": warehouse, "statement": statement, "wait_timeout": "50s"})
    state = (resp.get("status") or {}).get("state")
    if state != "SUCCEEDED":
        raise SystemExit(f"SQL {state}: {(resp.get('status') or {}).get('error')} -- {statement[:110]}")
    return (resp.get("result") or {}).get("data_array") or []


def _q(name: str) -> str:
    return ".".join(f"`{p}`" for p in name.split("."))


def owned(profile: str, warehouse: str, table: str) -> bool:
    try:
        props = {r[0]: r[1] for r in sql(profile, warehouse, f"SHOW TBLPROPERTIES {_q(table)}")}
    except SystemExit:
        return False                      # unreadable or gone: not provably ours
    return props.get(OWNER[0]) == OWNER[1]


def config_index() -> str | None:
    m = re.search(r"^VS_INDEX=(\S+)", (REPO / "config.env").read_text(), re.M)
    return m.group(1) if m else None


def plan(profile: str, endpoint: str, warehouse: str, keep: set[str], schema: str | None):
    """-> (indexes to delete as (index, table), tables to drop, kept with reasons)."""
    kept: list[tuple[str, str]] = []
    del_idx: list[tuple[str, str]] = []
    listing = api(profile, "get", f"/api/2.0/vector-search/indexes?endpoint_name={endpoint}")
    for ix in listing.get("vector_indexes") or []:
        name = ix["name"]
        spec = api(profile, "get", f"/api/2.0/vector-search/indexes/{name}")
        table = ((spec.get("delta_sync_index_spec") or {}).get("source_table")) or ""
        if name in keep:
            kept.append((name, "kept (config.env VS_INDEX / --keep)"))
        elif not table or not _IDENT.fullmatch(table) or not owned(profile, warehouse, table):
            kept.append((name, f"not created by verl-on-air (source table {table or '?'} lacks {OWNER[0]})"))
        else:
            del_idx.append((name, table))
    drop = [t for _, t in del_idx]
    if schema:
        cat, sch = schema.split(".")
        rows = sql(profile, warehouse, f"SELECT table_name FROM `{cat}`.information_schema.tables "
                                       f"WHERE table_schema = '{sch}' AND table_name LIKE '%\\\\_\\\\_staging\\\\_%'")
        for (tname,) in rows:
            full = f"{cat}.{sch}.{tname}"
            if full not in drop and owned(profile, warehouse, full):
                drop.append(full)
    return del_idx, drop, kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--schema", help="<catalog>.<schema> to also sweep for leftover staging tables")
    ap.add_argument("--keep", default="", help="comma list of index names to keep")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--profile", default="df1")
    args = ap.parse_args(argv)
    keep = {k.strip() for k in args.keep.split(",") if k.strip()} | ({config_index()} - {None})
    del_idx, drop, kept = plan(args.profile, args.endpoint, args.warehouse_id, keep, args.schema)
    for name, why in kept:
        print(f"  keep    {name}: {why}")
    for name, table in del_idx:
        print(f"  delete  index {name} (table {table})")
    for t in drop:
        print(f"  drop    table {t}")
    if not (del_idx or drop):
        print("nothing of ours to delete")
        return 0
    if not args.confirm:
        print("dry run -- re-run with CONFIRM=1 (--confirm) to delete")
        return 0
    for name, _ in del_idx:
        api(args.profile, "delete", f"/api/2.0/vector-search/indexes/{name}")
        print(f"  deleted index {name}")
    for t in drop:
        sql(args.profile, args.warehouse_id, f"DROP TABLE IF EXISTS {_q(t)}")
        print(f"  dropped table {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
