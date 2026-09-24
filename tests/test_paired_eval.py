"""scripts/paired_eval.py: paired statistics for base vs checkpoint evals (REVIEW.md R11)."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest

from support import REPO, load_module

pe = load_module(REPO / "scripts" / "paired_eval.py")


def art(path: Path, correct: list[bool], *, hop="2hop", extra=None) -> str:
    path.write_text(json.dumps({**(extra or {}), "results": [
        {"uid": str(i), "question": f"q{i}", "hop_type": hop, "correct": c} for i, c in enumerate(correct)]}))
    return str(path)


def test_exact_mcnemar_matches_a_hand_computation():
    # 13 gained vs 4 lost (the published headline): 2 * sum_{i<=4} C(17, i) / 2^17
    want = 2 * sum(comb(17, i) for i in range(5)) / 2 ** 17
    assert pe.exact_mcnemar(13, 4) == pytest.approx(want) and 0.04 < want < 0.05
    assert pe.exact_mcnemar(0, 0) == 1.0


def test_paired_report(tmp_path: Path):
    base = [True] * 10 + [False] * 10
    ckpt = [True] * 8 + [False] * 2 + [True] * 7 + [False] * 3     # +7 gained, -2 lost
    out = tmp_path / "out.json"
    pe.main([art(tmp_path / "b.json", base), art(tmp_path / "c.json", ckpt), "--json-out", str(out),
             "--permutations", "2000", "--bootstrap", "500"])
    r = json.loads(out.read_text())
    (c,) = r["checkpoints"]
    assert (c["gained"], c["lost"], c["correct"]) == (7, 2, 15)
    assert c["exact_mcnemar_p"] == pytest.approx(pe.exact_mcnemar(7, 2))
    assert c["delta"] == pytest.approx(0.25) and c["delta_ci95"][0] < 0.25 < c["delta_ci95"][1]
    assert c["by_hop_type"]["2hop"] == {"n": 20, "base_correct": 10, "correct": 15}


def test_one_checkpoint_permutation_p_tracks_exact_mcnemar(tmp_path: Path):
    base = [False] * 13 + [True] * 4 + [True] * 20     # 13 gained, 4 lost, 20 unchanged
    ckpt = [True] * 13 + [False] * 4 + [True] * 20
    out = tmp_path / "o.json"
    pe.main([art(tmp_path / "b.json", base), art(tmp_path / "c.json", ckpt), "--json-out", str(out),
             "--permutations", "40000"])
    r = json.loads(out.read_text())
    assert r["selection_adjusted_p"] == pytest.approx(r["checkpoints"][0]["exact_mcnemar_p"], abs=0.01)


def test_selecting_the_best_of_many_is_penalised(tmp_path: Path):
    # ten equally good checkpoints, each gaining 6 DIFFERENT questions: each looks "significant"
    # on its own (p = 2/2^6), but the best of ten is what chance alone produces ~27% of the time
    base = [False] * 60
    arts = [art(tmp_path / "b.json", base)]
    for k in range(10):
        arts.append(art(tmp_path / f"c{k}.json", [6 * k <= i < 6 * k + 6 for i in range(60)]))
    out = tmp_path / "o.json"
    pe.main(arts + ["--json-out", str(out), "--permutations", "20000"])
    r = json.loads(out.read_text())
    assert min(c["exact_mcnemar_p"] for c in r["checkpoints"]) == pytest.approx(2 / 64)
    assert 0.2 < r["selection_adjusted_p"] < 0.35


def test_different_question_sets_are_refused(tmp_path: Path):
    a = art(tmp_path / "a.json", [True, False, True])
    b = art(tmp_path / "b.json", [True, False])
    with pytest.raises(SystemExit, match="different questions"):
        pe.main([a, b])


def test_an_invalid_eval_is_refused(tmp_path: Path):
    a = art(tmp_path / "a.json", [True, False])
    b = art(tmp_path / "b.json", [True, True], extra={"valid": False, "invalid_reasons": ["infra"]})
    with pytest.raises(SystemExit, match="INVALID"):
        pe.main([a, b])


def test_a_different_eval_policy_is_a_different_measurement(tmp_path: Path):
    pol = {"version": 2, "max_turns": 12, "tool_schema_sha256": "aa"}
    base = art(tmp_path / "base.json", [True, False, True], extra={"eval_policy": pol})
    same = art(tmp_path / "same.json", [True, True, True], extra={"eval_policy": dict(pol)})
    other = art(tmp_path / "other.json", [True, True, True], extra={"eval_policy": {**pol, "max_turns": 8}})
    legacy = art(tmp_path / "legacy.json", [True, True, True])
    assert pe.main([base, same]) == 0
    with pytest.raises(SystemExit, match="max_turns: 12 vs 8"):
        pe.main([base, other])
    with pytest.raises(SystemExit, match="predates policies"):
        pe.main([base, legacy])
    out = tmp_path / "deliberate.json"
    assert pe.main([base, other, "--allow-policy-mismatch", "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["policy_mismatch_allowed"] is True
    # two legacy artifacts (no policy recorded: everything published so far) still pair
    assert pe.main([art(tmp_path / "l1.json", [True, False]), art(tmp_path / "l2.json", [True, True])]) == 0


def test_a_policy_field_added_later_defaults_to_what_older_artifacts_did(tmp_path: Path):
    # samples_per_question arrived with the variance probe; an artifact written before it sampled once
    pol = {"version": 2, "max_turns": 12}
    older = art(tmp_path / "older.json", [True, False, True], extra={"eval_policy": pol})
    one = art(tmp_path / "one.json", [True, True, True], extra={"eval_policy": {**pol, "samples_per_question": 1}})
    eight = art(tmp_path / "eight.json", [True, True, True], extra={"eval_policy": {**pol, "samples_per_question": 8}})
    assert pe.main([older, one]) == 0
    with pytest.raises(SystemExit, match="samples_per_question: 1 vs 8"):
        pe.main([older, eight])
