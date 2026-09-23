"""engine/lib/data_manifest.py: pinned sources, mirrors only by content, provenance (R17).

Before: every dataset was read from a moving `main`, and a list of mirrors was tried in order --
the first that loaded, whatever it held, became "the dataset"."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import ENGINE, load_module

dm = load_module(ENGINE / "lib" / "data_manifest.py")
SHA = "a" * 40
PRIMARY, MIRROR = dm.Source("org/ds", SHA), dm.Source("mirror/ds", "b" * 40)
GOOD = [{"q": "1"}]


def verify(rows):
    return None if rows == GOOD else f"got {rows}"


def boom(_src):
    raise ConnectionError("hub down")


@pytest.fixture(autouse=True)
def no_fallback(monkeypatch):
    monkeypatch.delenv("ALLOW_FALLBACK_SOURCE", raising=False)


def test_a_source_is_pinned_to_a_commit_not_a_branch():
    with pytest.raises(dm.SourceError, match="40-hex commit"):
        dm.Source("org/ds", "main")
    assert dm.Source("org/ds", SHA, "cfg").label == "org/ds:cfg@aaaaaaaaaaaa"


def test_loading_passes_the_pinned_revision(monkeypatch):
    import datasets
    seen = {}
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: seen.update(args=a, kw=k) or "ds")
    assert dm.Source("org/ds", SHA, "cfg").load(split="test", cache_dir="/c") == "ds"
    assert seen["args"] == ("org/ds", "cfg")
    assert seen["kw"] == {"split": "test", "revision": SHA, "cache_dir": "/c"}


def test_content_digest_ignores_order_but_not_content():
    a = [{"q": "1", "a": "x"}, {"q": "2", "a": "y"}]
    assert dm.content_digest(a, ["q", "a"]) == dm.content_digest(a[::-1], ["q", "a"])
    assert dm.content_digest(a, ["q", "a"]) != dm.content_digest([a[0], {"q": "2", "a": "z"}], ["q", "a"])
    with pytest.raises(KeyError):
        dm.content_digest([{"q": "1"}], ["q", "a"])   # a missing field never compares equal


def test_the_pinned_source_is_used_and_recorded():
    data, rec = dm.load_verified([(PRIMARY, lambda s: GOOD), (MIRROR, boom)], verify)
    assert data == GOOD and (rec["hf_id"], rec["revision"], rec["fallback"]) == ("org/ds", SHA, False)


def test_a_mirror_is_not_even_tried_without_permission():
    tried = []
    with pytest.raises(dm.SourceError, match="hub down.*ALLOW_FALLBACK_SOURCE=1"):
        dm.load_verified([(PRIMARY, boom), (MIRROR, lambda s: tried.append(s) or GOOD)], verify)
    assert tried == []


def test_a_mirror_holding_other_data_is_refused_even_when_allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_FALLBACK_SOURCE", "1")
    with pytest.raises(dm.SourceError, match=r"mirror/ds@b{12}: got"):
        dm.load_verified([(PRIMARY, boom), (MIRROR, lambda s: [{"q": "other"}])], verify)


def test_an_equivalent_mirror_is_taken_and_marked_as_a_fallback(monkeypatch):
    monkeypatch.setenv("ALLOW_FALLBACK_SOURCE", "1")
    data, rec = dm.load_verified([(PRIMARY, boom), (MIRROR, lambda s: GOOD)], verify)
    assert data == GOOD and rec["hf_id"] == "mirror/ds" and rec["fallback"] is True
    assert "hub down" in rec["rejected"][0]


def test_the_pinned_source_itself_must_hold_the_expected_content():
    with pytest.raises(dm.SourceError, match="got"):
        dm.load_verified([(PRIMARY, lambda s: [{"q": "drifted"}])], verify)


def test_a_manifest_describes_its_files_and_provenance_checks_them(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUN_ID", "r1")
    f = dm.write_parquet(tmp_path / "train.parquet", [{"a": 1}, {"a": 2}])
    assert not list(tmp_path.glob(".*partial"))
    dm.write_manifest(tmp_path / dm.DIR_MANIFEST, tool="t", sources=[PRIMARY.record()],
                      outputs=[dm.output_record(f, 2)])
    prov = dm.provenance(f)
    assert prov["file_matches_manifest"] is True and prov["rows"] == 2 and prov["run_id"] == "r1"
    assert prov["sources"][0]["revision"] == SHA and prov["sha256"] == dm.sha256_file(f)

    dm.write_parquet(f, [{"a": 3}])                        # changed after its manifest was written
    assert dm.provenance(f)["file_matches_manifest"] is False


def test_data_without_a_manifest_still_gets_its_hash(tmp_path: Path):
    f = tmp_path / "old.parquet"
    f.write_bytes(b"built before manifests")
    prov = dm.provenance(f)
    assert prov["manifest"] is None and prov["sha256"] == dm.sha256_file(f)


def test_the_manifest_that_lists_the_file_is_the_one_used(tmp_path: Path):
    corpus = dm.write_parquet(tmp_path / "corpus.parquet", [{"a": 1}])
    dm.write_manifest(tmp_path / dm.DIR_MANIFEST, tool="prep", sources=[], outputs=[])
    assert dm.find_manifest(corpus) is None                # a directory manifest that does not list it
    dm.write_manifest(tmp_path / "corpus.manifest.json", tool="corpus", sources=[],
                      outputs=[dm.output_record(corpus, 1)])
    path, doc, entry = dm.find_manifest(corpus)
    assert path.name == "corpus.manifest.json" and doc["tool"] == "corpus" and entry["rows"] == 1
    assert json.loads(path.read_text())["schema"] == dm.SCHEMA


def test_the_output_record_never_reads_the_written_file_back(tmp_path: Path):
    """Acceptance run A3: hashing a train.parquet just overwritten on a Volume FUSE mount raised
    `[Errno 5] Input/output error`. write_parquet hashes its local copy; output_record reuses it."""
    import hashlib
    out = dm.write_parquet(tmp_path / "train.parquet", [{"a": 1}, {"a": 2}])
    want = hashlib.sha256(out.read_bytes()).hexdigest()
    out.chmod(0)                       # unreadable, like the EIO'ing FUSE file
    try:
        rec = dm.output_record(out, 2)
    finally:
        out.chmod(0o644)
    assert rec["sha256"] == want and rec["bytes"] == out.stat().st_size and rec["rows"] == 2
    assert not list(tmp_path.glob(".*.partial"))
