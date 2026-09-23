"""MATH prep + the eval sets: pinned commits, mirrors only when they hold the same data (R17).

Before: prep walked a list of five hub ids and used the first that loaded -- two of which could
never load (one is gone, one is a loader script) -- and the eval sets read a moving `main`.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import datasets
import pytest

from support import env, load_usecase

MATH_ROWS = {
    "train": [{"problem": "What is 1+1?", "solution": "So \\boxed{2}.", "level": "Level 3", "type": "Algebra"},
              {"problem": "Easy one.", "solution": "\\boxed{1}", "level": "Level 1", "type": "Prealgebra"}],
    "test": [{"problem": "Half of 1?", "solution": "It is \\boxed{\\frac{1}{2}}", "level": "Level 4", "type": "Algebra"}],
}


@pytest.fixture
def hub(monkeypatch):
    """datasets.load_dataset over in-memory repos keyed by (repo, config); an Exception = unavailable."""
    class Hub:
        calls: list = []
        data: dict = {}

    def load(path, name=None, split=None, revision=None, **kw):
        Hub.calls.append((path, name, split, revision))
        got = Hub.data[(path, name)]
        if isinstance(got, Exception):
            raise got
        dd = datasets.DatasetDict({k: datasets.Dataset.from_list(v) for k, v in got.items()})
        return dd[split] if split else dd
    Hub.calls, Hub.data = [], {}
    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.delenv("ALLOW_FALLBACK_SOURCE", raising=False)
    return Hub


# ------------------------------------------------------------------------------ prep_data.py ----
@pytest.fixture
def prep(hub, monkeypatch):
    p = load_usecase("math", "prep_data")
    dm = p.dm
    monkeypatch.setattr(p, "MATH_CONTENT", {s: (len(r), dm.content_digest(r, ["problem", "solution"]))
                                            for s, r in MATH_ROWS.items()})
    monkeypatch.setattr(p, "_MATH_SUBJECTS", ["algebra"])     # the mirror's one subject holds it all
    hub.data[(p.MATH.hf_id, None)] = MATH_ROWS
    hub.data[(p.MATH_MIRROR.hf_id, "algebra")] = MATH_ROWS
    return p


def run_prep(prep, out: Path, *extra: str) -> tuple[list[dict], dict]:
    assert prep.main(["--local_save_dir", str(out), "--levels", "3,4,5", *extra]) == 0
    rows = datasets.Dataset.from_parquet(str(out / "train.parquet")).to_list()
    return rows, json.loads((out / "DATA_MANIFEST.json").read_text())


def test_math_is_read_at_the_pinned_commit_and_recorded(prep, hub, tmp_path: Path):
    rows, m = run_prep(prep, tmp_path)
    assert hub.calls == [(prep.MATH.hf_id, None, None, prep.MATH.revision)]
    assert [r["reward_model"]["ground_truth"] for r in rows] == ["2"]          # the level-1 row is filtered
    assert rows[0]["data_source"] == prep.MATH.hf_id
    src = m["sources"][0]
    assert (src["hf_id"], src["revision"], src["fallback"]) == (prep.MATH.hf_id, prep.MATH.revision, False)
    assert src["splits"]["train"]["source_rows"] == 2 and src["splits"]["train"]["kept"] == 1
    assert m["filters"]["levels"] == [3, 4, 5]


def test_an_unavailable_source_is_an_error_not_a_quiet_substitute(prep, hub, tmp_path: Path):
    hub.data[(prep.MATH.hf_id, None)] = ConnectionError("hub down")
    with pytest.raises(SystemExit, match="hub down.*ALLOW_FALLBACK_SOURCE=1"):
        prep.main(["--local_save_dir", str(tmp_path)])
    assert [c[0] for c in hub.calls] == [prep.MATH.hf_id]                    # the mirror was not touched


def test_an_equivalent_mirror_gives_the_same_dataset_when_allowed(prep, hub, tmp_path: Path, monkeypatch):
    primary_rows, _ = run_prep(prep, tmp_path / "primary")
    hub.data[(prep.MATH.hf_id, None)] = ConnectionError("hub down")
    monkeypatch.setenv("ALLOW_FALLBACK_SOURCE", "1")
    rows, m = run_prep(prep, tmp_path / "mirror")
    assert rows == primary_rows                                                # same rows, same data_source
    assert m["sources"][0]["hf_id"] == prep.MATH_MIRROR.hf_id and m["sources"][0]["fallback"] is True


def test_a_mirror_or_a_pin_with_other_content_is_refused(prep, hub, tmp_path: Path, monkeypatch):
    other = {"train": MATH_ROWS["train"][:1], "test": MATH_ROWS["test"]}
    hub.data[(prep.MATH.hf_id, None)] = other                                  # not the recorded content
    with pytest.raises(SystemExit, match="train: 1 rows"):
        prep.main(["--local_save_dir", str(tmp_path)])
    monkeypatch.setenv("ALLOW_FALLBACK_SOURCE", "1")
    hub.data[(prep.MATH_MIRROR.hf_id, "algebra")] = other
    with pytest.raises(SystemExit, match="no source holds the pinned data"):
        prep.main(["--local_save_dir", str(tmp_path)])


# ------------------------------------------------------------------------------ eval.py sets ----
def aime(answers) -> dict:
    return {"train": [{"problem": f"P{a}", "answer": a} for a in answers]}


def test_math500_is_read_at_its_pinned_commit(hub):
    ev = load_usecase("math", "eval")
    hub.data[("HuggingFaceH4/MATH-500", None)] = {"test": [
        {"problem": "1+1?", "answer": "2", "level": "Level 1", "subject": "Algebra", "solution": ""}]}
    assert [r["gt"] for r in ev._load_math500()] == ["2"]
    assert hub.calls == [("HuggingFaceH4/MATH-500", None, "test", ev.MATH500_REVISION)]
    assert ev._DATASET_META["revision"] == ev.MATH500_REVISION


def test_pointing_math500_elsewhere_requires_pinning_it():
    ev = load_usecase("math", "eval", MATH500_ID="someone/else")
    with pytest.raises(ev.dm.SourceError, match="40-hex commit"):
        ev._load_math500()


def test_an_aime_mirror_needs_permission_and_the_same_answers(hub, monkeypatch):
    ev = load_usecase("math", "eval")
    (primary, _), (mirror, _), (mirror2, _) = ev._AIME_2025
    want = [str(i) for i in range(30)]
    monkeypatch.setitem(ev._AIME_ANSWERS, 2025, ev.dm.content_digest([{"gt": a} for a in want], ["gt"]))
    hub.data[(primary.hf_id, None)] = ConnectionError("hub down")
    hub.data[(mirror.hf_id, None)] = aime(range(30))
    hub.data[(mirror2.hf_id, None)] = aime(range(1, 31))                       # a different exam
    with pytest.raises(ev.dm.SourceError, match="ALLOW_FALLBACK_SOURCE=1"):
        ev._load_aime_group(ev._AIME_2025, 2025, 10000)
    monkeypatch.setenv("ALLOW_FALLBACK_SOURCE", "1")
    rows = ev._load_aime_group(ev._AIME_2025, 2025, 10000)
    assert sorted(r["gt"] for r in rows) == sorted(want)
    assert ev._DATASET_META["sources"][-1]["hf_id"] == mirror.hf_id and ev._DATASET_META["sources"][-1]["fallback"]
    hub.data[(mirror.hf_id, None)] = aime(range(1, 31))
    with pytest.raises(ev.dm.SourceError, match="answers digest"):
        ev._load_aime_group(ev._AIME_2025, 2025, 10000)


def test_an_eval_set_that_cannot_be_loaded_stops_the_eval_before_any_problem(hub, tmp_path: Path):
    e = {"EVAL_BASE_URL": "http://127.0.0.1:9/v1", "EVAL_OUT": str(tmp_path / "out.json")}
    ev = load_usecase("math", "eval", **e)
    ev._load_tokenizer = lambda: None
    ev._load_parser = lambda tok: None
    hub.data[("HuggingFaceH4/MATH-500", None)] = ConnectionError("hub down")
    with env(**e):
        assert asyncio.run(ev._main_async()) == 2
    assert not (tmp_path / "out.json").exists()
