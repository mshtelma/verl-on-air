"""W2.8: `make prune-ckpts` and `make cleanup-vs` delete only what is provably safe to delete.

Both drive the `databricks` CLI; here it is a stub that serves a fake Volume / workspace and records
every call, so a test can assert what would have been deleted -- and that nothing is without CONFIRM.
"""
from __future__ import annotations

import os
from pathlib import Path

from support import REPO, StubBin, run

FS_STUB = r'''
root="$FAKE_ROOT"
if [ "$1" = fs ] && [ "$2" = ls ]; then
  p="${3#dbfs:}"; d="$root$p"
  [ -d "$d" ] || { echo "no such directory: $p" >&2; exit 1; }
  python3 -c 'import json,os,sys; print(json.dumps([{"name": n} for n in sorted(os.listdir(sys.argv[1]))]))' "$d"
  exit 0
fi
if [ "$1" = fs ] && [ "$2" = rm ]; then p="${4#dbfs:}"; rm -rf "$root$p"; exit 0; fi
exit 3
'''


def _run_dir(root: Path, steps: dict[int, bool]) -> str:
    run_dir = "/Volumes/c/s/v/ckpt/job/RUN1"
    d = root / run_dir.lstrip("/")
    for step, complete in steps.items():
        actor = d / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        if complete:
            (actor / "ckpt_contents.json").write_text("{}")
    (d / "run_manifest.json").write_text("{}")
    (d / "latest_checkpointed_iteration.txt").write_text("40")
    return run_dir


def _prune(stub_bin: StubBin, root: Path, *args: str):
    stub_bin.add("databricks", FS_STUB)
    return run(["python3", "scripts/prune_ckpts.py", *args], env=stub_bin.env(FAKE_ROOT=str(root)))


def test_prune_keeps_final_chosen_incomplete_and_metadata(stub_bin: StubBin, tmp_path: Path):
    run_dir = _run_dir(tmp_path, {10: True, 20: True, 30: True, 35: False, 40: True})
    d = tmp_path / run_dir.lstrip("/")
    r = _prune(stub_bin, tmp_path, run_dir, "--keep", "20")
    assert r.returncode == 0 and "dry run" in r.stdout, r.stdout
    assert not [c for c in stub_bin.calls("databricks") if " rm " in c], "a dry run deleted something"
    r = _prune(stub_bin, tmp_path, run_dir, "--keep", "20", "--confirm")
    assert r.returncode == 0, r.stdout
    assert sorted(p.name for p in d.iterdir()) == ["global_step_20", "global_step_35", "global_step_40",
                                                    "latest_checkpointed_iteration.txt", "run_manifest.json"]


def test_prune_refuses_a_step_dir_an_unknown_keep_and_a_run_with_nothing_complete(stub_bin, tmp_path):
    run_dir = _run_dir(tmp_path, {10: True, 20: False})
    assert _prune(stub_bin, tmp_path, run_dir + "/global_step_10").returncode != 0
    r = _prune(stub_bin, tmp_path, run_dir, "--keep", "20")
    assert r.returncode != 0 and "no complete checkpoint" in r.stdout
    other = _run_dir(tmp_path / "b", {5: False})
    assert _prune(stub_bin, tmp_path / "b", other, "--confirm").returncode != 0
    assert (tmp_path / "b" / other.lstrip("/") / "global_step_5").is_dir()


VS_STUB = r'''
if [ "$1" = api ]; then
  printf '%s %s\n' "$2" "$3" >> "$FAKE_LOG"
  python3 "$FAKE_WS" "$@"; exit $?
fi
exit 3
'''
WS = r'''
import json, sys
args = sys.argv[1:]
method, path = args[1], args[2]
body = json.loads(args[args.index("--json") + 1]) if "--json" in args else {}
INDEXES = {"c.s.ours_v1_index": "c.s.ours_v1", "c.s.theirs_index": "c.s.theirs",
           "c.s.current_index": "c.s.current"}
OWNED = {"c.s.ours_v1", "c.s.current", "c.s.ours_v2__staging_r1"}
if method == "get" and path.startswith("/api/2.0/vector-search/indexes?"):
    print(json.dumps({"vector_indexes": [{"name": n} for n in INDEXES]}))
elif method == "get":
    n = path.rsplit("/", 1)[1]
    print(json.dumps({"name": n, "delta_sync_index_spec": {"source_table": INDEXES[n]}}))
elif method == "post":
    st = body["statement"]
    if st.startswith("SHOW TBLPROPERTIES"):
        t = st.split(" ", 2)[2].replace("`", "")
        rows = [["voa.created_by", "verl-on-air"]] if t in OWNED else [["delta.enableChangeDataFeed", "true"]]
    elif "information_schema" in st:
        rows = [["ours_v2__staging_r1"], ["theirs__staging_x"]]
    else:
        rows = []
    print(json.dumps({"status": {"state": "SUCCEEDED"}, "result": {"data_array": rows}}))
elif method == "delete":
    print("{}")
'''


def _cleanup(stub_bin: StubBin, tmp_path: Path, *extra: str):
    ws = tmp_path / "ws.py"
    ws.write_text(WS)
    log = tmp_path / "api.log"
    log.touch()
    stub_bin.add("databricks", VS_STUB)
    r = run(["python3", "scripts/cleanup_vs.py", "--endpoint", "ep", "--warehouse-id", "w", "--schema", "c.s",
             "--keep", "c.s.current_index", *extra],
            env=stub_bin.env(FAKE_WS=str(ws), FAKE_LOG=str(log)))
    return r, log.read_text()


def test_cleanup_vs_deletes_only_owned_resources_and_only_when_confirmed(stub_bin, tmp_path):
    r, log = _cleanup(stub_bin, tmp_path)
    assert r.returncode == 0 and "dry run" in r.stdout, r.stdout
    assert "delete " not in log.replace("delete  ", "")          # no DELETE call
    assert "delete  index c.s.ours_v1_index" in r.stdout
    assert "drop    table c.s.ours_v2__staging_r1" in r.stdout and "theirs__staging_x" not in r.stdout.split("keep")[0]
    assert "keep    c.s.theirs_index: not created by verl-on-air" in r.stdout
    assert "keep    c.s.current_index" in r.stdout

    r, log = _cleanup(stub_bin, tmp_path, "--confirm")
    assert r.returncode == 0, r.stdout
    deletes = [ln for ln in log.splitlines() if ln.startswith("delete ")]
    assert deletes == ["delete /api/2.0/vector-search/indexes/c.s.ours_v1_index"], deletes
    assert "dropped table c.s.ours_v1" in r.stdout and "dropped table c.s.ours_v2__staging_r1" in r.stdout
    assert "c.s.theirs" not in r.stdout.split("dropped", 1)[1]


def test_budget_gate_blocks_training_targets_without_budget_ok():
    # PATH only: no BUDGET_OK can leak in from the caller's environment and submit a real job
    r = run(["make", "--no-print-directory", "search-train"], env={"PATH": os.environ["PATH"]})
    assert r.returncode != 0 and "can bill up to 160 GPU-hours" in r.stdout and "BUDGET_OK=1" in r.stdout
