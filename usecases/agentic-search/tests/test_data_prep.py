"""prep_data.py + build_corpus.py: pinned sources, stable ids, no silently smaller corpus (R17, R18).

Before: every loader read a moving `main`; a corpus source that failed -- even midway -- was
logged as SKIPPED and the build succeeded with whatever had been collected, including the
failed source's partial passages."""
from __future__ import annotations

import json
from pathlib import Path

import datasets
import pytest

from support import load_usecase


def musique_row(qid: str) -> dict:
    return {"id": qid, "question": f"Who founded {qid}?", "answer": "A", "answer_aliases": ["a"],
            "answerable": True, "question_decomposition": [{"id": 1}, {"id": 2}],
            "paragraphs": [
                {"idx": 0, "title": f"T{qid}", "paragraph_text": f"the gold passage for {qid}",
                 "is_supporting": True},
                {"idx": 1, "title": "Distractor", "paragraph_text": "a distractor paragraph text",
                 "is_supporting": False}]}


def hotpot_row(qid: str) -> dict:
    return {"id": qid, "question": "Which?", "answer": "B", "type": "bridge", "level": "hard",
            "supporting_facts": {"title": [f"H{qid}"], "sent_id": [0]},
            "context": {"title": [f"H{qid}", "Other"],
                        "sentences": [[f"hotpot gold sentence {qid}."], ["another sentence here."]]}}


MUSIQUE = {"train": [musique_row("2hop__1_2"), musique_row("2hop__3_4")], "validation": [musique_row("2hop__5_6")]}
HOTPOT = {"train": [hotpot_row("h1")], "validation": [hotpot_row("h2")]}


class Flaky:
    """A split that yields one row, then the connection drops."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        yield self.rows[0]
        raise ConnectionError("stream reset")


@pytest.fixture
def hub(monkeypatch):
    """datasets.load_dataset over in-memory repos; records (repo, config, revision) per call."""
    class Hub:
        calls: list = []
        data = {"dgslibisey/MuSiQue": dict(MUSIQUE), "hotpotqa/hotpot_qa": dict(HOTPOT)}

    def load(path, name=None, split=None, revision=None, **kw):
        Hub.calls.append((path, name, revision))
        got = Hub.data[path]
        if isinstance(got, Exception):
            raise got
        if all(isinstance(v, list) for v in got.values()):
            return datasets.DatasetDict({k: datasets.Dataset.from_list(v) for k, v in got.items()})
        return got
    Hub.calls = []
    monkeypatch.setattr(datasets, "load_dataset", load)
    return Hub


def read(path: Path) -> list[dict]:
    return datasets.Dataset.from_parquet(str(path)).to_list()


# ---------------------------------------------------------------- questions (prep_data.py) ----
def test_questions_come_from_the_pinned_revision_and_keep_their_own_ids(hub, tmp_path: Path):
    prep = load_usecase("agentic-search", "prep_data")
    assert prep.main(["--out-dir", str(tmp_path)]) == 0
    assert hub.calls == [("dgslibisey/MuSiQue", None, prep.SOURCES["musique"].revision)]
    train = read(tmp_path / "train.parquet")
    assert [r["extra_info"]["source_id"] for r in train] == ["2hop__1_2", "2hop__3_4"]
    assert train[0]["reward_model"]["ground_truth"] == ["A"] and train[0]["extra_info"]["hop_type"] == "2hop"
    m = json.loads((tmp_path / "DATA_MANIFEST.json").read_text())
    assert m["sources"][0]["revision"] == prep.SOURCES["musique"].revision
    assert {o["name"]: o["rows"] for o in m["outputs"]} == {"train.parquet": 2, "test.parquet": 1}
    assert m["sources"][0]["splits"]["val"] == {"hf_split": "validation", "source_rows": 1, "selected": 1,
                                               "kept": 1, "dropped": {}}


def test_a_question_in_both_splits_fails_the_prep(hub, tmp_path: Path):
    hub.data["dgslibisey/MuSiQue"] = {"train": [musique_row("2hop__1_2")], "validation": [musique_row("2hop__1_2")]}
    prep = load_usecase("agentic-search", "prep_data")
    with pytest.raises(SystemExit, match="in both train and val"):
        prep.main(["--out-dir", str(tmp_path)])
    assert not (tmp_path / "train.parquet").exists()


@pytest.mark.parametrize("name", ["2wiki", "nq"])
def test_only_sources_that_can_be_pinned_and_served_are_offered(name, tmp_path: Path):
    # 2Wiki's mirror is a loader script (datasets>=4 cannot run it); NQ has no corpus behind it
    with pytest.raises(SystemExit, match="unknown dataset"):
        load_usecase("agentic-search", "prep_data").main(["--datasets", name, "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="unknown corpus source"):
        load_usecase("agentic-search", "build_corpus").main(["--datasets", name, "--out", str(tmp_path / "c.parquet")])


# ----------------------------------------------------------------- corpus (build_corpus.py) ----
def test_the_corpus_is_the_union_with_its_provenance_and_coverage(hub, tmp_path: Path):
    bc = load_usecase("agentic-search", "build_corpus")
    out = tmp_path / "corpus_big.parquet"
    assert bc.main(["--out", str(out)]) == 0
    rows = read(out)
    assert len(rows) == 7                  # 3 MuSiQue golds + 1 shared distractor + 2 HotpotQA golds + "Other"
    assert [c[:2] for c in hub.calls] == [("dgslibisey/MuSiQue", None), ("hotpotqa/hotpot_qa", "distractor")]
    m = json.loads((tmp_path / "corpus_big.manifest.json").read_text())
    assert m["complete"] is True and m["failed_sources"] == [] and "transductive" in m["setting"]
    assert [(s["name"], s["revision"]) for s in m["sources"]] == [
        (n, bc.SOURCES[n].revision) for n in ("musique", "hotpotqa")]
    val = m["sources"][0]["splits"]["validation"]
    assert val["supporting"] == val["supporting_kept"] == val["questions_fully_supported"] == 1
    assert m["outputs"][0]["rows"] == 7
    assert m["outputs"][0]["content_sha256"] == bc.dm.content_digest(rows, ["id", "title", "text"])


def test_a_failed_source_fails_the_build_and_the_last_corpus_is_kept(hub, tmp_path: Path):
    bc = load_usecase("agentic-search", "build_corpus")
    out, manifest = tmp_path / "corpus_big.parquet", tmp_path / "corpus_big.manifest.json"
    assert bc.main(["--out", str(out)]) == 0
    before = (out.read_bytes(), manifest.read_text())
    hub.data["hotpotqa/hotpot_qa"] = ConnectionError("hub down")
    with pytest.raises(SystemExit, match=r"1 corpus source\(s\) failed: \['hotpotqa'\].*--allow-partial"):
        bc.main(["--out", str(out)])
    assert (out.read_bytes(), manifest.read_text()) == before


def test_a_partial_corpus_holds_nothing_of_a_source_that_failed_midway(hub, tmp_path: Path):
    hub.data["dgslibisey/MuSiQue"] = {"train": Flaky(MUSIQUE["train"]), "validation": MUSIQUE["validation"]}
    bc = load_usecase("agentic-search", "build_corpus")
    out = tmp_path / "corpus_big.parquet"
    assert bc.main(["--out", str(out), "--allow-partial"]) == 0
    assert {r["title"] for r in read(out)} == {"Hh1", "Hh2", "Other"}   # not one MuSiQue passage
    m = json.loads((tmp_path / "corpus_big.manifest.json").read_text())
    assert m["complete"] is False and [s["name"] for s in m["sources"]] == ["hotpotqa"]
    assert m["failed_sources"][0]["name"] == "musique" and "stream reset" in m["failed_sources"][0]["error"]


def test_a_requested_split_a_source_lacks_is_a_failure_not_a_skip(hub, tmp_path: Path):
    bc = load_usecase("agentic-search", "build_corpus")
    with pytest.raises(SystemExit, match="2 corpus source"):
        bc.main(["--out", str(tmp_path / "c.parquet"), "--splits", "train,dev"])


def test_passage_ids_are_the_ones_in_the_published_index():
    # a passage exactly as it sits in the published 603,607-passage corpus: a rebuild must keep its id
    bc = load_usecase("agentic-search", "build_corpus")
    assert bc._passage_id("London to Brighton (disambiguation)",
                          "London to Brighton is a 2006 film by Paul Andrew Williams.") == "p_001d85ff5b6ee6720bd8"
