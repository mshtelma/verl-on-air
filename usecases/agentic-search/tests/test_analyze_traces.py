"""analyze_traces.py (R28): runs on a clean machine, never guesses, and reports the WHOLE EM identity.

Reviewer reproduction: the documented command crashed (it wrote to a /tmp/musique_diag it never
created), a file without "base" in its name raised KeyError, heuristic labels could collide, empty
input divided by zero, and the "EM = recall x conversion" identity dropped the not-retrieved term.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import USECASES, run

SCRIPT = USECASES / "agentic-search" / "analyze_traces.py"


def _rec(uid: int, *, em: float, surfaced: bool, q: str | None = None, **kw) -> dict:
    hit = "[0] Paris  score=0.900\n    Paris is the capital of France." if surfaced else \
          "[0] Lyon  score=0.500\n    Lyon is a city."
    return {"uid": str(uid), "question": q or f"q{uid}", "gt": ["Paris"], "pred": "Paris" if em else "Lyon",
            "em": em, "cover_em": em, "f1": em, "hop_type": "2hop",
            "trajectory": [{"turn": 0, "tool_results": [{"name": "vector_search", "result": hit}]}], **kw}


def _write(p: Path, recs: list[dict]) -> Path:
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def _analyze(tmp_path: Path, *files: Path, extra: tuple[str, ...] = ()) -> tuple[int, str, dict | None]:
    out = tmp_path / "diag"                      # does not exist yet: the script creates it
    r = run(["python3", str(SCRIPT), *map(str, files), "--out", str(out), *extra], cwd=tmp_path)
    doc = json.loads((out / "trace_diagnostic.json").read_text()) if r.returncode == 0 else None
    return r.returncode, r.stdout, doc


def test_both_terms_of_the_decomposition_are_reported_and_sum_to_em(tmp_path: Path):
    # 4 surfaced (3 correct), 2 not surfaced (1 correct from memory): EM = 4/6
    recs = [_rec(0, em=1, surfaced=True), _rec(1, em=1, surfaced=True), _rec(2, em=1, surfaced=True),
            _rec(3, em=0, surfaced=True), _rec(4, em=1, surfaced=False), _rec(5, em=0, surfaced=False)]
    rc, out, doc = _analyze(tmp_path, _write(tmp_path / "anything.jsonl", recs))
    assert rc == 0, out
    d = doc["runs"]["anything"]["decomposition"]
    assert d["p_surfaced"] == pytest.approx(4 / 6) and d["p_correct_given_surfaced"] == pytest.approx(3 / 4)
    assert d["p_correct_given_not_surfaced"] == pytest.approx(1 / 2)
    assert d["term_surfaced"] + d["term_not_surfaced"] == pytest.approx(doc["runs"]["anything"]["em_pct"] / 100)
    assert d["term_not_surfaced"] == pytest.approx(1 / 6)       # the term the old identity dropped
    assert doc["runs"]["anything"]["buckets"]["SURFACED_MISSED"] == 1
    assert doc["runs"]["anything"]["buckets"]["NOT_SURFACED"] == 1
    assert "hypothesis generator" in out


def test_two_files_are_paired_by_uid(tmp_path: Path):
    a = _write(tmp_path / "a.jsonl", [_rec(0, em=1, surfaced=True), _rec(1, em=0, surfaced=True),
                                       _rec(2, em=0, surfaced=False)])
    b = _write(tmp_path / "b.jsonl", [_rec(2, em=1, surfaced=True), _rec(1, em=1, surfaced=True),
                                       _rec(0, em=0, surfaced=True), _rec(9, em=1, surfaced=True)])
    rc, out, doc = _analyze(tmp_path, a, b, extra=("--label", "base", "--label", "step20"))
    assert rc == 0, out
    p = doc["paired"]["step20"]
    assert p["n_paired"] == 3 and p["unpaired_other"] == 1
    assert (p["only_ref_correct"], p["only_other_correct"], p["both_correct"]) == (1, 2, 0)
    assert p["surfaced_other_only"] == 1
    assert p["bucket_transitions"]["NOT_SURFACED->CORRECT"] == 1


def test_a_uid_naming_different_questions_is_refused(tmp_path: Path):
    a = _write(tmp_path / "a.jsonl", [_rec(0, em=1, surfaced=True, q="Who?")])
    b = _write(tmp_path / "b.jsonl", [_rec(0, em=1, surfaced=True, q="Where?")])
    rc, out, _ = _analyze(tmp_path, a, b)
    assert rc == 2 and "different questions" in out


@pytest.mark.parametrize("case", ["empty", "missing", "no_args", "dup_labels", "label_count", "dup_uid"])
def test_bad_input_is_a_clear_error_not_a_crash(tmp_path: Path, case):
    good = _write(tmp_path / "x_traces.jsonl", [_rec(0, em=1, surfaced=True)])
    files, extra, msg = {
        "empty": ([_write(tmp_path / "e.jsonl", [])], (), "no trace records"),
        "missing": ([tmp_path / "nope.jsonl"], (), "no such trace file"),
        "no_args": ([], (), "at least one trace file"),
        # two stems that collide once the _traces suffix is dropped
        "dup_labels": ([good, _write(tmp_path / "x.jsonl", [_rec(0, em=1, surfaced=True)])], (), "names two files"),
        "label_count": ([good], ("--label", "a", "--label", "b"), "--label"),
        "dup_uid": ([_write(tmp_path / "d.jsonl", [_rec(0, em=1, surfaced=True)] * 2)], (), "duplicate uids"),
    }[case]
    rc, out, _ = _analyze(tmp_path, *files, extra=extra)
    assert rc == 2 and msg in out and "Traceback" not in out, out


def test_supporting_titles_give_chain_recall(tmp_path: Path):
    two_hop = _rec(0, em=1, surfaced=True)
    two_hop["trajectory"].append({"turn": 1, "tool_results": [
        {"name": "read_article", "result": "Article: France\n\nFrance is a country."}]})
    half = _rec(1, em=0, surfaced=True)
    t = _write(tmp_path / "t.jsonl", [two_hop, half])
    sup = _write(tmp_path / "sup.jsonl", [{"question": "q0", "supporting_titles": ["France", "Paris"]},
                                          {"uid": "1", "supporting_titles": ["Paris", "Seine"]}])
    rc, out, doc = _analyze(tmp_path, t, extra=("--supporting", str(sup)))
    assert rc == 0, out
    s = doc["runs"]["t"]["supporting"]
    assert s["n_with_titles"] == 2 and s["mean_support_recall"] == pytest.approx(0.75)
    assert s["p_all_supporting_surfaced"] == pytest.approx(0.5) and s["em_pct_given_all_supporting"] == 100.0


def test_the_module_imports_nothing_heavy():
    """A clean machine: no torch / verl / datasets at import time."""
    r = run(["python3", "-c", "import sys, runpy; sys.argv=['x']; "
             f"sys.path.insert(0, {str(SCRIPT.parent)!r}); import analyze_traces; "
             "bad = [m for m in ('torch', 'verl', 'datasets', 'vllm') if m in sys.modules]; "
             "print('HEAVY', bad) if bad else print('OK')"])
    assert r.stdout.strip().endswith("OK"), r.stdout
