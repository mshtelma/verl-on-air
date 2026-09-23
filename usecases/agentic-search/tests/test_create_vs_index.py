"""create_vs_index.py: versioned, verified, non-destructive; one client in every mode (R19).

Before: a reload TRUNCATEd the live table before COPY INTO (a failed load left it empty under
running jobs), a reused table was never checked, an existing index was re-synced without waiting,
--profile was ignored by --status-only/--wait-only/--skip-load, the warehouse was whichever came
first, and SQL polling had no deadline.

The fake below is the two SDK surfaces the script uses -- statement_execution and api_client.do --
over an in-memory catalog. It raises on any statement it does not model, so a TRUNCATE, DROP or
DELETE fails the test by construction.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from support import load_usecase

ROWS = [{"id": f"p_{i}", "title": f"T{i}", "text": f"passage number {i} text"} for i in range(3)]


class NotFound(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class FakeWorkspace:
    def __init__(self, copy_rows: int = len(ROWS)):
        self.tables: dict[str, dict] = {}
        self.endpoints = {"wiki-qa-vs": {"endpoint_status": {"state": "ONLINE"}}}
        self.indexes: dict[str, dict] = {}
        self.copy_rows = copy_rows            # how many rows COPY INTO loads
        self.sql_log: list[str] = []
        self.api_log: list[tuple[str, str]] = []
        self.hang = False
        self.cancelled: list[str] = []
        self.statement_execution = self
        self.api_client = self

    # --- statement_execution ---------------------------------------------------------------
    def execute_statement(self, warehouse_id, statement, wait_timeout):
        assert warehouse_id == "wh1"
        self.sql_log.append(statement)
        rows = None if self.hang else self._run(statement)
        return SimpleNamespace(statement_id="s1", result=SimpleNamespace(data_array=rows),
                               status=SimpleNamespace(state="RUNNING" if self.hang else "SUCCEEDED", error=None))

    def get_statement(self, sid):
        raise AssertionError("the fake answers synchronously unless hung")

    def cancel_execution(self, sid):
        self.cancelled.append(sid)

    def _run(self, s: str):
        names = [".".join(g) for g in re.findall(r"`([^`]+)`\.`([^`]+)`\.`([^`]+)`", s)]
        if s.startswith("SELECT table_name FROM"):
            cat = re.search(r"FROM `(\w+)`\.information_schema", s).group(1)
            schema, name = re.search(r"table_schema = '(\w+)' AND table_name = '(\w+)'", s).groups()
            return [[name]] if f"{cat}.{schema}.{name}" in self.tables else []
        if s.startswith("CREATE OR REPLACE TABLE"):
            self.tables[names[0]] = {"cols": [("id", "string"), ("title", "string"), ("text", "string")],
                                     "props": dict(re.findall(r"'([^']+)' = '([^']*)'", s)), "n": 0}
            return []
        if s.startswith("COPY INTO"):
            self.tables[names[0]]["n"] = self.copy_rows
            return []
        if s.startswith("DESCRIBE TABLE"):
            return [[c, t, None] for c, t in self.tables[names[0]]["cols"]]
        if s.startswith("SHOW TBLPROPERTIES"):
            return [[k, v] for k, v in self.tables[names[0]]["props"].items()]
        if s.startswith("SELECT count(*), count(DISTINCT id)"):
            n = self.tables[names[0]]["n"]
            return [[str(n), str(n)]]
        if s.startswith("ALTER TABLE") and " RENAME TO " in s:
            self.tables[names[1]] = self.tables.pop(names[0])
            return []
        raise AssertionError(f"unexpected SQL: {s}")

    # --- api_client --------------------------------------------------------------------------
    def do(self, method, path, body=None):
        self.api_log.append((method, path))
        kind, _, name = path.removeprefix("/api/2.0/vector-search/").partition("/")
        store = self.endpoints if kind == "endpoints" else self.indexes
        if method == "GET":
            if name not in store:
                raise NotFound(name)
            got = store[name]
            return got.pop(0) if isinstance(got, list) else got
        if kind == "endpoints":
            store[body["name"]] = {"endpoint_status": {"state": "ONLINE"}}
        else:
            store[body["name"]] = {**body, "status": {"detailed_state": "PROVISIONING_INITIAL_SNAPSHOT",
                                                      "ready": False, "indexed_row_count": 0}}
        return {}

    def writes(self) -> list[str]:
        return [s for s in self.sql_log if not s.startswith(("SELECT", "DESCRIBE", "SHOW"))]


@pytest.fixture
def cvi(monkeypatch):
    m = load_usecase("agentic-search", "create_vs_index")
    monkeypatch.setattr(m, "_VOLUME_PATH", re.compile(r"/[A-Za-z0-9_./-]+\.parquet"))   # tmp, not /Volumes
    monkeypatch.setattr(m, "POLL_S", 0)
    monkeypatch.setenv("RUN_ID", "t1")
    return m


@pytest.fixture
def corpus(cvi, tmp_path: Path):
    dm = cvi.dm
    path = dm.write_parquet(tmp_path / "corpus_big.parquet", ROWS)

    def manifest(complete: bool = True):
        dm.write_manifest(tmp_path / "corpus_big.manifest.json", tool="build_corpus", sources=[],
                          complete=complete, failed_sources=[] if complete else [{"name": "hotpotqa"}],
                          outputs=[dm.output_record(path, len(ROWS),
                                                    content_sha256=dm.content_digest(ROWS, ["id", "title", "text"]))])
    manifest()
    sha = dm.content_digest(ROWS, ["id", "title", "text"])
    return SimpleNamespace(path=path, sha=sha, manifest=manifest,
                           table=f"main.mshtelma.wiki_qa_big_corpus_v{sha[:8]}")


def build(cvi, ws, corpus, *extra: str) -> int:
    return cvi.main(["--corpus", str(corpus.path), "--warehouse-id", "wh1", "--no-wait", *extra], client=ws)


def test_a_first_build_loads_a_versioned_table_through_staging(cvi, corpus):
    ws = FakeWorkspace()
    assert build(cvi, ws, corpus) == 0
    staging = f"{corpus.table}__staging_t1"
    assert set(ws.tables) == {corpus.table}
    assert ws.tables[corpus.table]["props"]["voa.corpus_sha256"] == corpus.sha
    assert ws.tables[corpus.table]["props"]["delta.enableChangeDataFeed"] == "true"
    order = [s.split(" (")[0] for s in ws.writes()]
    assert order == [f"CREATE OR REPLACE TABLE {cvi._q(staging)}", f"COPY INTO {cvi._q(staging)} FROM",
                     f"ALTER TABLE {cvi._q(staging)} RENAME TO {cvi._q(corpus.table)}"]
    rename = next(i for i, s in enumerate(ws.sql_log) if s.startswith("ALTER TABLE"))
    count = max(i for i, s in enumerate(ws.sql_log) if s.startswith("SELECT count(*)"))
    assert count < rename                                   # verified BEFORE it gets the real name
    idx = ws.indexes[f"{corpus.table}_index"]
    assert idx["delta_sync_index_spec"]["source_table"] == corpus.table and idx["primary_key"] == "id"


def test_rebuilding_the_same_corpus_reuses_the_table_and_index_untouched(cvi, corpus):
    ws = FakeWorkspace()
    assert build(cvi, ws, corpus) == 0
    ws.sql_log.clear()
    ws.api_log.clear()
    assert build(cvi, ws, corpus) == 0
    assert ws.writes() == [] and [c for c in ws.api_log if c[0] == "POST"] == []


def test_an_existing_table_holding_other_data_is_left_untouched(cvi, corpus):
    ws = FakeWorkspace()
    ws.tables[corpus.table] = {"cols": [("id", "string"), ("title", "string"), ("text", "string")],
                               "props": {"delta.enableChangeDataFeed": "true", "voa.corpus_sha256": "other"},
                               "n": 3}
    with pytest.raises(SystemExit, match="does not hold this corpus.*left untouched"):
        build(cvi, ws, corpus)
    assert ws.writes() == [] and ws.tables[corpus.table]["props"]["voa.corpus_sha256"] == "other"


def test_a_short_load_never_gets_the_versioned_name(cvi, corpus):
    ws = FakeWorkspace(copy_rows=2)
    with pytest.raises(SystemExit, match=r"load into .*__staging_t1 is wrong: 2 rows"):
        build(cvi, ws, corpus)
    assert corpus.table not in ws.tables and f"{corpus.table}__staging_t1" in ws.tables
    assert ws.indexes == {}


def test_a_missing_endpoint_is_created_only_when_asked(cvi, corpus):
    ws = FakeWorkspace()
    ws.endpoints.clear()
    with pytest.raises(SystemExit, match="does not exist.*billable.*--create-endpoint"):
        build(cvi, ws, corpus)
    assert ("POST", "/api/2.0/vector-search/endpoints") not in ws.api_log and ws.indexes == {}
    assert build(cvi, ws, corpus, "--create-endpoint") == 0
    assert "wiki-qa-vs" in ws.endpoints and f"{corpus.table}_index" in ws.indexes


def test_an_existing_index_that_is_not_this_index_is_refused(cvi, corpus):
    ws = FakeWorkspace()
    assert build(cvi, ws, corpus) == 0
    idx = ws.indexes[f"{corpus.table}_index"]
    idx["delta_sync_index_spec"]["embedding_source_columns"][0]["embedding_model_endpoint_name"] = "other-model"
    with pytest.raises(SystemExit, match="is not this index.*other-model.*Left untouched"):
        build(cvi, ws, corpus)


def test_the_corpus_must_be_the_complete_file_its_manifest_describes(cvi, corpus, tmp_path: Path):
    ws = FakeWorkspace()
    corpus.manifest(complete=False)
    with pytest.raises(SystemExit, match=r"INCOMPLETE \(failed sources: \['hotpotqa'\]\)"):
        build(cvi, ws, corpus)
    assert build(cvi, ws, corpus, "--allow-partial") == 0            # knowingly

    ws = FakeWorkspace()
    corpus.manifest()
    cvi.dm.write_parquet(corpus.path, ROWS[:2])                       # edited after the manifest
    with pytest.raises(SystemExit, match="sha256 differs"):
        build(cvi, ws, corpus)
    (tmp_path / "corpus_big.manifest.json").unlink()
    with pytest.raises(SystemExit, match="no manifest lists"):
        build(cvi, ws, corpus)
    assert ws.sql_log == [] and ws.api_log == []


def test_a_build_needs_a_named_warehouse(cvi, corpus):
    ws = FakeWorkspace()
    with pytest.raises(SystemExit, match="--warehouse-id .* is required"):
        cvi.main(["--corpus", str(corpus.path), "--no-wait"], client=ws)
    assert ws.sql_log == []


@pytest.mark.parametrize("bad", [["--table", "x; DROP TABLE y"], ["--catalog", "a.b"], ["--schema", "s`"],
                                 ["--endpoint", "bad name"], ["--corpus", "/tmp/it's.parquet"]])
def test_identifiers_and_paths_are_validated_before_any_call(cvi, corpus, bad):
    ws = FakeWorkspace()
    with pytest.raises(SystemExit):
        cvi.main(["--corpus", str(corpus.path), "--warehouse-id", "wh1", *bad], client=ws)
    assert ws.sql_log == [] and ws.api_log == []


def test_sql_that_outlives_its_deadline_is_cancelled(cvi):
    ws = FakeWorkspace()
    ws.hang = True
    with pytest.raises(SystemExit, match="cancelled"):
        cvi.Workspace(ws, "wh1", sql_timeout_s=0).sql("SELECT 1")
    assert ws.cancelled == ["s1"]


def status(detail: str, ready: bool, rows: int) -> dict:
    return {"name": "i", "status": {"detailed_state": detail, "ready": ready, "indexed_row_count": rows}}


def test_ready_means_this_snapshot_is_indexed_not_just_online(cvi, corpus):
    ws = FakeWorkspace()
    index = f"{corpus.table}_index"
    for st, want in [(status("ONLINE_NO_PENDING_UPDATE", True, 2), 3),    # serving, but 2 of 3 rows
                     (status("PROVISIONING_INITIAL_SNAPSHOT", False, 0), 3),
                     (status("ONLINE_PIPELINE_FAILED", True, 3), 1),
                     (status("ONLINE_NO_PENDING_UPDATE", True, 3), 0)]:
        ws.indexes[index] = st
        assert cvi.main(["--corpus", str(corpus.path), "--status-only"], client=ws) == want
    ws.indexes[index] = [status("PROVISIONING_INITIAL_SNAPSHOT", False, 0), status("ONLINE", True, 3)]
    assert cvi.main(["--corpus", str(corpus.path), "--wait-only"], client=ws) == 0
    ws.indexes[index] = status("PROVISIONING_INITIAL_SNAPSHOT", False, 0)
    assert cvi.main(["--corpus", str(corpus.path), "--wait-only", "--wait-timeout-s", "0"], client=ws) == 3


def test_an_explicitly_named_index_can_be_inspected(cvi):
    ws = FakeWorkspace()
    ws.indexes["main.mshtelma.wiki_qa_big_corpus_index"] = status("ONLINE_NO_PENDING_UPDATE", True, 603607)
    assert cvi.main(["--index", "main.mshtelma.wiki_qa_big_corpus_index", "--status-only"], client=ws) == 0
    with pytest.raises(SystemExit, match="--index is for --status-only"):
        cvi.main(["--index", "main.mshtelma.wiki_qa_big_corpus_index"], client=ws)


@pytest.mark.parametrize("mode", [["--no-wait"], ["--status-only"], ["--wait-only"]])
def test_every_mode_authenticates_with_the_selected_profile(cvi, corpus, monkeypatch, mode):
    ws = FakeWorkspace()
    ws.endpoints["wiki-qa-vs"] = {"endpoint_status": {"state": "ONLINE"}}
    ws.indexes[f"{corpus.table}_index"] = status("ONLINE_NO_PENDING_UPDATE", True, 3)
    profiles = []
    monkeypatch.setattr(cvi, "connect", lambda profile: profiles.append(profile) or ws)
    if mode == ["--no-wait"]:          # a build: the index exists already, so it must match
        ws.indexes[f"{corpus.table}_index"].update(
            endpoint_name="wiki-qa-vs", index_type="DELTA_SYNC", primary_key="id",
            delta_sync_index_spec={"source_table": corpus.table, "embedding_source_columns": [
                {"name": "text", "embedding_model_endpoint_name": "databricks-gte-large-en"}]})
    rc = cvi.main(["--corpus", str(corpus.path), "--warehouse-id", "wh1", "--profile", "df1", *mode])
    assert rc == 0 and profiles == ["df1"]


def test_connect_hands_the_profile_to_the_sdk(cvi, monkeypatch):
    got = []
    sdk = types.ModuleType("databricks.sdk")
    sdk.WorkspaceClient = lambda **kw: got.append(kw) or "client"
    monkeypatch.setitem(sys.modules, "databricks", types.ModuleType("databricks"))
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    assert cvi.connect("df1") == "client" and cvi.connect(None) == "client"
    assert got == [{"profile": "df1"}, {}]
