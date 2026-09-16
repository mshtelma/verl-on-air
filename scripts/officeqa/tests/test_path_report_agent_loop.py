#!/usr/bin/env python3
"""CPU tests for the path-report verl AGENT LOOP core (pure helpers + episode finalization).
No verl, no model, no network, no GPU -- the verl-dependent classes are import-guarded, so only
the stdlib core is exercised here (the classes are validated in-image by air/100).

Regression focus: verl's ``DataProto.concat`` asserts an IDENTICAL non_tensor_batch key schema
across a GRPO group. The first in-image run (air/100 run 159118978400452) FAILED with
``Key '_pr_obs' is not present in the keys of the first dictionary`` because finalization ran
only on submit -- turn-capped samples kept ``_pr_obs`` and lacked ``path_report_record``. These
tests lock the fix: EVERY episode finalizes to the SAME single output key, no scratch survives.

Run standalone:
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_path_report_agent_loop.py
Also importable by pytest.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", ".."),
           os.path.join(_HERE, "..", "..", "reward")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import path_report as pr                          # noqa: E402  (status constants + parser)
import path_report_agent_loop as al               # noqa: E402  (module under test; pure core)
import officeqa_path_report_reward as rw          # noqa: E402  (round-trip seam)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
# The pilot's known-good SCORED fixture (path_report_pilot._self_check_records): claims are
# faithful to the delivered observation text, so the value gate + references pass.
_BASE_OBS = [
    {"observation_id": "obs_1", "order": 1, "tool": "read_document",
     "delivered_text": "1941 total 1.47 trillion", "generated_by_request": 0, "delivered_to_request": 1},
    {"observation_id": "obs_2", "order": 2, "tool": "read_document",
     "delivered_text": "1942 total 1.30 trillion", "generated_by_request": 1, "delivered_to_request": 2},
    {"observation_id": "obs_3", "order": 3, "tool": "compute", "args": {"code": "print(1.47-1.30)"},
     "delivered_text": "0.17", "generated_by_request": 2, "delivered_to_request": 3},
]
_VALID_REPORT = {
    "answer": "0.17 trillion dollars",
    "path": [
        {"id": "a", "observation": "obs_1", "claim": "1941 total 1.47"},
        {"id": "b", "observation": "obs_2", "claim": "1942 total 1.30"},
        {"id": "c", "observation": "obs_3", "depends_on": ["a", "b"], "claim": "1.47-1.30=0.17"},
    ],
}


def _supported_judge(_request: dict) -> str:
    """A stub reviewer that would ALWAYS support -- proves an empty report is a hard ZERO on its
    own (never rescued by the judge) and that a valid report reaches SCORED."""
    return '{"path_status":"supported","path_score":0.85,"issues":[]}'


def _base_ef() -> dict:
    """The extra_fields keys ToolAgentLoop.run adds to EVERY sample (uniform baseline)."""
    return {"turn_scores": [], "tool_rewards": []}


# ---------------------------------------------------------------------------
# pure helpers: funnel
# ---------------------------------------------------------------------------
def test_funnel_phase_boundaries():
    # air/100 config: max_turns=16, nudge=4, submit_only=2 (remaining = max_turns - assistant_turns)
    f = lambda t: al.funnel_phase(t, 16, 4, 2)
    assert f(0) == "explore"
    assert f(11) == "explore"          # remaining 5 > 4
    assert f(12) == "retrieval_lock"   # remaining 4 <= 4
    assert f(13) == "retrieval_lock"   # remaining 3
    assert f(14) == "submit_only"      # remaining 2 <= 2
    assert f(15) == "submit_only"      # remaining 1


def test_funnel_phase_windows_disabled():
    # windows of 0 disable that phase entirely -> always explore
    assert al.funnel_phase(15, 16, 0, 0) == "explore"


def test_active_tool_names_per_phase():
    names = ["search_documents", "grep_documents", "read_document", "list_documents",
             "compute", al.SUBMIT_TOOL]
    assert al.active_tool_names("explore", names) == names
    lock = al.active_tool_names("retrieval_lock", names)
    assert set(lock) == {"compute", al.SUBMIT_TOOL}          # retrieval dropped, compute+submit stay
    assert al.active_tool_names("submit_only", names) == [al.SUBMIT_TOOL]
    # submit is offered in every phase; retrieval tools are exactly those dropped in the lock
    for rt in al.RETRIEVAL_TOOLS:
        assert rt not in lock


def test_record_observation_ids_and_delivery():
    obs = []
    oid1 = al.record_observation(obs, tool="read_document", args={"file_name": "x"},
                                 output="some text", turn=3)
    oid2 = al.record_observation(obs, tool="compute", args={"code": "1+1"}, output="2", turn=4)
    assert (oid1, oid2) == ("obs_1", "obs_2")
    assert obs[0]["order"] == 1 and obs[1]["order"] == 2
    # verl appends every tool response to the context -> delivered on the turn it was generated
    assert obs[0]["delivered_to_request"] == 3 and obs[0]["generated_by_request"] == 3
    assert obs[1]["tool"] == "compute"


# ---------------------------------------------------------------------------
# build_path_report_record: explicit terminal_text/termination (both loop exits)
# ---------------------------------------------------------------------------
def test_build_record_shape_terminal_report():
    rec = al.build_path_report_record(episode_id="E", question="q", requirements="r",
                                      terminal_text='{"answer":"1","path":[]}',
                                      termination="terminal_report", obs_list=_BASE_OBS)
    assert rec["episode_id"] == "E" and rec["termination"] == "terminal_report"
    assert rec["terminal_text"] == '{"answer":"1","path":[]}'
    assert rec["observations"] is _BASE_OBS
    assert rec["n_observations"] == 3 and rec["n_delivered"] == 3   # all three delivered
    assert rec["answer_correct"] is None                           # a reviewer/answer-key sets it


def test_build_record_shape_no_submit():
    rec = al.build_path_report_record(episode_id="E", question="q", requirements="r",
                                      terminal_text="", termination="no_submit_at_cap", obs_list=[])
    assert rec["terminal_text"] == "" and rec["termination"] == "no_submit_at_cap"
    assert rec["n_observations"] == 0 and rec["n_delivered"] == 0
    # required by the reward's record extractor (observations must be present, not None)
    assert rec["observations"] == []


# ---------------------------------------------------------------------------
# finalize_extra_fields: the fix -- uniform single output key, no scratch, both exits
# ---------------------------------------------------------------------------
def test_finalize_submit_builds_terminal_report():
    ef = _base_ef()
    ef.update({"_pr_question": "q", "_pr_episode_id": "A", "_pr_requirements": "req",
               "_pr_obs": list(_BASE_OBS), "_pr_submit": dict(_VALID_REPORT)})
    al.finalize_extra_fields(ef)
    assert not any(k.startswith("_pr_") for k in ef)             # scratch gone
    rec = ef["path_report_record"]
    assert rec["termination"] == "terminal_report"
    assert json.loads(rec["terminal_text"])["answer"] == "0.17 trillion dollars"
    assert rec["episode_id"] == "A" and rec["question"] == "q" and rec["question_requirements"] == "req"
    assert rec["n_observations"] == 3


def test_finalize_no_submit_builds_empty_report():
    ef = _base_ef()
    ef.update({"_pr_question": "q", "_pr_episode_id": "B", "_pr_requirements": "",
               "_pr_obs": list(_BASE_OBS)})                       # retrieved but NEVER submitted
    al.finalize_extra_fields(ef)
    assert not any(k.startswith("_pr_") for k in ef)
    rec = ef["path_report_record"]
    assert rec["termination"] == "no_submit_at_cap" and rec["terminal_text"] == ""
    assert rec["n_observations"] == 3                            # observations preserved for provenance


def test_finalize_idempotent():
    ef = _base_ef()
    ef.update({"_pr_episode_id": "C", "_pr_submit": dict(_VALID_REPORT)})
    al.finalize_extra_fields(ef)
    snap = json.dumps(ef, sort_keys=True, default=str)
    al.finalize_extra_fields(ef)                                 # second call: no-op, no raise
    assert json.dumps(ef, sort_keys=True, default=str) == snap


def test_finalize_uniform_schema_across_group():
    """THE regression: a GRPO group mixing submit / turn-cap / immediate-give-up samples must end
    with an IDENTICAL non_tensor_batch key schema, or DataProto.concat aborts the whole batch."""
    a = _base_ef()   # submitted
    a.update({"_pr_question": "q", "_pr_episode_id": "A", "_pr_requirements": "",
              "_pr_obs": list(_BASE_OBS), "_pr_submit": dict(_VALID_REPORT)})
    b = _base_ef()   # retrieved, hit the turn cap without submitting  (had _pr_obs)
    b.update({"_pr_question": "q", "_pr_episode_id": "B", "_pr_requirements": "",
              "_pr_obs": list(_BASE_OBS)})
    c = _base_ef()   # gave up on turn 0: never called a tool, never submitted  (no _pr_obs at all)
    c.update({"_pr_question": "q", "_pr_episode_id": "C", "_pr_requirements": ""})

    for ef in (a, b, c):
        al.finalize_extra_fields(ef)

    keysets = [frozenset(ef) for ef in (a, b, c)]
    assert keysets[0] == keysets[1] == keysets[2], f"non-uniform schema -> concat would abort: {keysets}"
    # the exact key the failed run tripped on is gone everywhere; the single output key is present
    for ef in (a, b, c):
        assert "_pr_obs" not in ef and not any(k.startswith("_pr_") for k in ef)
        assert "path_report_record" in ef
    assert keysets[0] == frozenset({"turn_scores", "tool_rewards", "path_report_record"})


# ---------------------------------------------------------------------------
# round-trip through the reward: submit-valid -> SCORED>0 ; no-submit -> ZERO=0.0 (not UNKNOWN)
# ---------------------------------------------------------------------------
def test_no_submit_scores_hard_zero_not_unknown():
    ef = _base_ef()
    ef.update({"_pr_episode_id": "nosub", "_pr_question": "decrease?", "_pr_requirements": "",
               "_pr_obs": list(_BASE_OBS)})
    al.finalize_extra_fields(ef)
    reward, out = rw.score_record(ef["path_report_record"], _supported_judge, "0.17 trillion dollars")
    assert reward == 0.0
    # empty report parses malformed -> ZERO BEFORE the judge; must NOT read as UNKNOWN (which the
    # contract treats as a verifier failure that should stay near-zero, not a wrong-answer signal)
    assert out["candidate"]["status"] == pr.ZERO, out["candidate"]


def test_submit_valid_scores_positive():
    ef = _base_ef()
    ef.update({"_pr_episode_id": "ok", "_pr_question": "decrease?", "_pr_requirements": "",
               "_pr_obs": list(_BASE_OBS), "_pr_submit": dict(_VALID_REPORT)})
    al.finalize_extra_fields(ef)
    rec = ef["path_report_record"]
    rec["answer_correct"] = True     # isolate the support+finalize path from the fuzzy answer scorer
    reward, out = rw.score_record(rec, _supported_judge, "0.17 trillion dollars")
    assert out["candidate"]["status"] == pr.SCORED, out["candidate"]
    assert reward > 0.0


# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed" + ("" if not failed else f", {len(failed)} FAILED: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
