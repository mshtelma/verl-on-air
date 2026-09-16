#!/usr/bin/env python3
"""CPU fixture matrix for the OfficeQA path-report feasibility test (pilot Section 7 step 1).

These are SOFTWARE tests of the deterministic layer + wiring, not OfficeQA question counts
and not a live-judge capability measurement. Semantic cases (wrong value claimed, wrong
period, code mismatch, missing coverage, genuine alternative) are exercised with FIXTURE
verdicts -- mocks prove the wiring, not the real GLM-5.3 judge (pilot Section 7.1).

Run standalone (no pytest, no GPU, no judge, no network):
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_path_report.py
Also importable by pytest (functions named test_*).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..")):  # scripts/officeqa , scripts/
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import path_report as pr  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------
def _obs(oid, tool, delivered_text, *, order=1, gen=0, deliver=1, args=None, outcome="ok"):
    return {"observation_id": oid, "order": order, "tool": tool, "args": args or {},
            "outcome": outcome, "delivered_text": delivered_text,
            "generated_by_request": gen, "delivered_to_request": deliver}


def _ledger(episode_id="ep1", observations=None):
    return pr.EpisodeLedger.from_record({"episode_id": episode_id, "observations": observations or []})


# a canonical valid 3-step report + matching ledger (subtraction over two reads)
_R1_TEXT = (
    '{"answer":"0.17 trillion dollars","path":['
    '{"id":"a","observation":"obs_1","claim":"earlier-period total is 1.47 trillion"},'
    '{"id":"b","observation":"obs_2","claim":"later-period total is 1.30 trillion"},'
    '{"id":"c","observation":"obs_3","depends_on":["a","b"],'
    '"claim":"compute subtracts 1.30 from 1.47 giving 0.17"}]}'
)


def _r1_ledger():
    return _ledger("ep1", [
        _obs("obs_1", "read_document", "1941-01 ... total 1.47 trillion", order=1, gen=0, deliver=1),
        _obs("obs_2", "read_document", "1942-01 ... total 1.30 trillion", order=2, gen=1, deliver=2),
        _obs("obs_3", "compute", "0.17", order=3, gen=2, deliver=3, args={"code": "print(1.47-1.30)"}),
    ])


def _verdict(raw):
    return pr.parse_support_verdict(raw)


_V_SUP = '{"path_status":"supported","path_score":0.8,"issues":[]}'
_V_UNSUP = '{"path_status":"unsupported","path_score":0,"issues":["a"]}'
_V_UNK = '{"path_status":"unknown","path_score":null}'


# ===========================================================================
# 1. Report parsing / interface (pilot Section 2)
# ===========================================================================
def test_parse_valid_report():
    r = pr.parse_report(_R1_TEXT)
    assert r.kind == "valid", r.errors
    assert r.answer.startswith("0.17")
    assert [s.id for s in r.steps] == ["a", "b", "c"]
    assert r.steps[2].depends_on == ["a", "b"]


def test_parse_abstention():
    r = pr.parse_report('{"answer":"DATA NOT AVAILABLE","path":[]}')
    assert r.kind == "abstention" and r.is_abstention


def test_parse_substantive_empty_path_is_malformed():
    r = pr.parse_report('{"answer":"42","path":[]}')
    assert r.kind == "malformed" and any("empty path" in e for e in r.errors)


def test_parse_duplicate_json_keys_is_malformed():
    r = pr.parse_report('{"answer":"1","answer":"2","path":[]}')
    assert r.kind == "malformed" and any("duplicate" in e.lower() for e in r.errors)


def test_parse_duplicate_step_ids_is_malformed():
    txt = ('{"answer":"1","path":['
           '{"id":"a","observation":"obs_1","claim":"x"},'
           '{"id":"a","observation":"obs_2","claim":"y"}]}')
    r = pr.parse_report(txt)
    assert r.kind == "malformed" and any("duplicate step id" in e for e in r.errors)


def test_parse_unresolved_dependency_is_malformed():
    txt = '{"answer":"1","path":[{"id":"a","observation":"obs_1","depends_on":["z"],"claim":"x"}]}'
    r = pr.parse_report(txt)
    assert r.kind == "malformed" and any("not an earlier step" in e for e in r.errors)


def test_parse_forward_dependency_is_malformed():
    # 'a' depends on 'b' which is defined later -> forward reference
    txt = ('{"answer":"1","path":['
           '{"id":"a","observation":"obs_1","depends_on":["b"],"claim":"x"},'
           '{"id":"b","observation":"obs_2","claim":"y"}]}')
    r = pr.parse_report(txt)
    assert r.kind == "malformed" and any("not an earlier step" in e for e in r.errors)


def test_parse_self_dependency_is_malformed():
    txt = '{"answer":"1","path":[{"id":"a","observation":"obs_1","depends_on":["a"],"claim":"x"}]}'
    r = pr.parse_report(txt)
    assert r.kind == "malformed" and any("itself" in e for e in r.errors)


def test_parse_bad_field_types_are_malformed():
    for txt in [
        '{"answer":42,"path":[]}',                                              # answer not str
        '{"answer":"1","path":{}}',                                            # path not list
        '{"answer":"1","path":[["not","an","object"]]}',                       # step not object
        '{"answer":"1","path":[{"id":"a","claim":"x"}]}',                       # missing observation
        '{"answer":"1","path":[{"id":"a","observation":"obs_1"}]}',             # missing claim
        '{"answer":"1","path":[{"id":"a","observation":"obs_1","claim":"x","depends_on":"a"}]}',  # deps not list
    ]:
        r = pr.parse_report(txt)
        assert r.kind == "malformed", f"expected malformed for {txt}"


def test_parse_oversize_is_malformed_and_records_limit():
    big = pr.parse_report('{"answer":"1","path":[]}', max_bytes=5)
    assert big.kind == "malformed" and big.limit["max_bytes"] == 5
    many = '{"answer":"1","path":[' + ",".join(
        '{"id":"s%d","observation":"obs_1","claim":"c"}' % i for i in range(10)) + ']}'
    r = pr.parse_report(many, max_steps=3)
    assert r.kind == "malformed" and any("steps" in e for e in r.errors)


def test_parse_ambiguous_multiple_objects_is_malformed():
    txt = '{"answer":"1","path":[{"id":"a","observation":"obs_1","claim":"c"}]} {"answer":"2","path":[]}'
    r = pr.parse_report(txt)
    assert r.kind == "malformed" and any("ambiguous" in e for e in r.errors)


def test_parse_code_fenced_report_ok():
    r = pr.parse_report("```json\n" + _R1_TEXT + "\n```")
    assert r.kind == "valid"


# ===========================================================================
# 2. Terminal-only extraction (Section 2: never scan tool output / drafts)
# ===========================================================================
def test_extract_terminal_ignores_tool_output():
    # A convenient JSON object living only in a tool result must never be the report.
    assistant_events = ["let me think...", "still working"]  # no terminal JSON
    terminal = pr.extract_terminal_text(assistant_events)
    assert terminal == "still working"
    assert pr.parse_report(terminal).kind == "malformed"  # no answer object


def test_extract_terminal_takes_last_assistant():
    events = ['{"answer":"draft","path":[]}', _R1_TEXT]
    assert pr.extract_terminal_text(events) == _R1_TEXT
    assert pr.parse_report(pr.extract_terminal_text(events)).kind == "valid"


def test_compute_emitted_json_is_not_a_report():
    # compute stdout that looks like a report is compute OUTPUT, not a terminal submission.
    led = _ledger("ep1", [_obs("obs_1", "compute", '{"answer":"999","path":[]}', order=1, gen=0, deliver=1)])
    # it lives in the ledger as a compute observation, never parsed as the report:
    assert led.get("obs_1").is_compute
    # and the report is parsed only from the terminal text we pass, not from the ledger.


# ===========================================================================
# 3. Deterministic reference checks (Section 3/4)
# ===========================================================================
def test_refs_valid_delivered():
    r = pr.parse_report(_R1_TEXT)
    assert pr.check_references(r, _r1_ledger()).valid


def test_refs_nonexistent():
    r = pr.parse_report('{"answer":"1","path":[{"id":"a","observation":"obs_404","claim":"c"}]}')
    chk = pr.check_references(r, _r1_ledger())
    assert not chk.valid and "nonexistent" in chk.codes


def test_refs_cross_episode():
    r = pr.parse_report('{"answer":"1","path":[{"id":"a","observation":"obs_from_ep2","claim":"c"}]}')
    chk = pr.check_references(r, _r1_ledger(), foreign_ids=frozenset({"obs_from_ep2"}))
    assert not chk.valid and "cross_episode" in chk.codes


def test_refs_undelivered():
    led = _ledger("ep1", [_obs("obs_1", "read_document", None, order=1, gen=0, deliver=None)])
    r = pr.parse_report('{"answer":"1","path":[{"id":"a","observation":"obs_1","claim":"c"}]}')
    chk = pr.check_references(r, led)
    assert not chk.valid and "undelivered" in chk.codes


def test_refs_compute_chronology_ok():
    # reads delivered at req 1 and 2; compute generated at req 2 -> inputs visible.
    r = pr.parse_report(_R1_TEXT)
    assert pr.check_references(r, _r1_ledger()).valid


def test_refs_compute_chronology_impossible():
    # the second read is delivered only at req 3, but compute was generated at req 2:
    # executing the read in the same parallel batch does not make it visible to compute.
    led = _ledger("ep1", [
        _obs("obs_1", "read_document", "1.47", order=1, gen=0, deliver=1),
        _obs("obs_2", "read_document", "1.30", order=2, gen=2, deliver=3),
        _obs("obs_3", "compute", "0.17", order=3, gen=2, deliver=3, args={"code": "print(1.47-1.30)"}),
    ])
    r = pr.parse_report(_R1_TEXT)
    chk = pr.check_references(r, led)
    assert not chk.valid and "impossible_chronology" in chk.codes


def test_compute_observation_surfaces_as_compute_to_judge():
    # build_support_request tags the cited observation with its AUTHORITATIVE tool name,
    # so the judge can never be told a compute result is a document read.
    r = pr.parse_report(_R1_TEXT)
    req = pr.build_support_request("Q", "", r, _r1_ledger())
    tools = {c["step_id"]: c["cited_tool"] for c in req["report"]}
    assert tools["c"] == "compute" and tools["a"] == "read_document"


def test_support_request_excludes_gold_and_labels():
    r = pr.parse_report(_R1_TEXT)
    req = pr.build_support_request("Q", "requires 1941 & 1942", r, _r1_ledger())
    blob = str(req).lower()
    for forbidden in ("gold", "reference_answer", "answer_correct", "mutation", "family", "label"):
        assert forbidden not in req, f"{forbidden} leaked into judge request keys"
    # the question's own requirements ARE allowed to be visible
    assert "1941" in req["question_requirements"]
    # tool history is the runtime-owned delivered bytes
    assert any("1.47" in h["delivered_text"] for h in req["tool_history"])


def test_support_request_history_bytes_bound_and_configurable():
    # _bounded_history truncates the TAIL once the byte cap is hit (keeps earliest, in order).
    # A long trajectory whose LATER observation carries the answer needs a big enough cap, or
    # that observation is dropped/truncated -> the judge cannot verify it. Verify both the bound
    # and that raising max_history_bytes recovers the full tail.
    big = "X" * 5000
    led = _ledger("ep1", [
        _obs("obs_1", "read_document", big, order=1, gen=0, deliver=1),
        _obs("obs_2", "read_document", big, order=2, gen=1, deliver=2),
        _obs("obs_3", "read_document", "TAIL_ANSWER_1941 total 1.47", order=3, gen=2, deliver=3),
    ])
    r = pr.parse_report('{"answer":"1.47","path":[{"id":"a","observation":"obs_3","claim":"c"}]}')
    # small cap: the tail observation is truncated away / empty
    tight = pr.build_support_request("Q", "", r, led, max_history_bytes=4000)
    tail_tight = [h for h in tight["tool_history"] if h["observation_id"] == "obs_3"]
    assert tight["limits"]["max_history_bytes"] == 4000
    assert not tail_tight or "TAIL_ANSWER" not in (tail_tight[0]["delivered_text"] if tail_tight else "")
    # big cap: the tail observation (which the report cites) is fully present
    loose = pr.build_support_request("Q", "", r, led, max_history_bytes=200_000)
    assert any("TAIL_ANSWER" in h["delivered_text"] for h in loose["tool_history"])


# ===========================================================================
# 4. Support-verdict parsing (Section 4) -- never a silent positive
# ===========================================================================
def test_verdict_supported_ok():
    v = _verdict(_V_SUP)
    assert v["status"] == "ok" and v["path_status"] == "supported" and v["path_score"] == 0.8


def test_verdict_supported_zero_is_contradiction():
    assert _verdict('{"path_status":"supported","path_score":0}')["status"] == "unknown"


def test_verdict_supported_out_of_range_is_unknown():
    assert _verdict('{"path_status":"supported","path_score":1.5}')["status"] == "unknown"


def test_verdict_unsupported_ok():
    v = _verdict(_V_UNSUP)
    assert v["status"] == "ok" and v["path_status"] == "unsupported" and v["path_score"] == 0.0
    v2 = _verdict('{"path_status":"unsupported","path_score":null}')
    assert v2["status"] == "ok" and v2["path_score"] == 0.0


def test_verdict_unsupported_nonzero_is_contradiction():
    assert _verdict('{"path_status":"unsupported","path_score":0.6}')["status"] == "unknown"


def test_verdict_unknown_status():
    assert _verdict(_V_UNK)["status"] == "unknown"


def test_verdict_missing_status_is_unknown():
    assert _verdict('{"path_score":0.9}')["status"] == "unknown"


def test_verdict_nan_inf_bool_string_score_is_unknown():
    for raw in [
        '{"path_status":"supported","path_score":NaN}',
        '{"path_status":"supported","path_score":Infinity}',
        '{"path_status":"supported","path_score":true}',
        '{"path_status":"supported","path_score":"0.9"}',   # stringy score -> UNKNOWN, not a positive
    ]:
        assert _verdict(raw)["status"] == "unknown", raw


def test_verdict_no_object_is_unknown():
    assert _verdict("the path looks fine to me")["status"] == "unknown"
    assert _verdict("")["status"] == "unknown"


def test_verdict_picks_last_object_after_think():
    raw = ('<think>maybe {"path_status":"supported","path_score":1.0} but wait</think>\n'
           '{"path_status":"unsupported","path_score":0,"issues":["b: wrong period"]}')
    v = _verdict(raw)
    assert v["status"] == "ok" and v["path_status"] == "unsupported"


def test_verdict_duplicate_key_is_unknown():
    assert _verdict('{"path_status":"supported","path_status":"unsupported","path_score":0.5}')["status"] == "unknown"


# ===========================================================================
# 5. Candidate-reward PREVIEW wiring (Section 4). Semantic cases use fixture verdicts.
# ===========================================================================
def _preview(report_text, ledger, verdict_raw, answer_correct, foreign=frozenset()):
    rep = pr.parse_report(report_text)
    ref = pr.check_references(rep, ledger, foreign_ids=foreign)
    verdict = pr.parse_support_verdict(verdict_raw) if verdict_raw is not None else None
    return pr.candidate_preview(report=rep, ref=ref, verdict=verdict, answer_correct=answer_correct)


def test_candidate_correct_supported_is_scored():
    p = _preview(_R1_TEXT, _r1_ledger(), '{"path_status":"supported","path_score":0.7}', True)
    assert p.status == pr.SCORED and p.score == 0.7


def test_candidate_correct_unsupported_is_zero():
    p = _preview(_R1_TEXT, _r1_ledger(), _V_UNSUP, True)
    assert p.status == pr.ZERO and p.score == 0.0


def test_candidate_correct_but_invalid_refs_is_zero():
    # fabricated / invalid path disqualifies EVEN with a correct answer, before any judge.
    txt = '{"answer":"0.17 trillion dollars","path":[{"id":"a","observation":"obs_404","claim":"c"}]}'
    p = _preview(txt, _r1_ledger(), _V_SUP, True)
    assert p.status == pr.ZERO and "invalid path" in p.reason


def test_candidate_invalid_refs_ignores_would_be_supported_verdict():
    txt = '{"answer":"0.17 trillion dollars","path":[{"id":"a","observation":"obs_404","claim":"c"}]}'
    rep = pr.parse_report(txt)
    ref = pr.check_references(rep, _r1_ledger())
    assert not ref.valid
    # even a glowing supported verdict cannot rescue an invalid path
    p = pr.candidate_preview(report=rep, ref=ref, verdict=pr.parse_support_verdict(_V_SUP), answer_correct=True)
    assert p.status == pr.ZERO


def test_candidate_wrong_answer_supported_is_zero_retains_label():
    p = _preview(_R1_TEXT, _r1_ledger(), _V_SUP, False)
    assert p.status == pr.ZERO and p.score == 0.0
    assert p.labels["path_status"] == "supported"          # report-fidelity/support label retained
    assert "report-fidelity" in p.reason


def test_candidate_correct_supported_judge_unknown_is_unknown():
    p = _preview(_R1_TEXT, _r1_ledger(), _V_UNK, True)
    assert p.status == pr.UNKNOWN and p.score is None


def test_candidate_malformed_report_is_zero():
    p = _preview('{"answer":"1","path":[]}', _r1_ledger(), _V_SUP, True)
    assert p.status == pr.ZERO


def test_candidate_abstention_is_recorded_apart():
    p = _preview('{"answer":"DATA NOT AVAILABLE","path":[]}', _r1_ledger(), None, None)
    assert p.status == pr.ABSTENTION and p.score == 0.0


def test_candidate_answer_unresolved_is_unknown():
    p = _preview(_R1_TEXT, _r1_ledger(), _V_SUP, None)
    assert p.status == pr.UNKNOWN and p.score is None


def test_candidate_fabricated_value_semantic_is_zero():
    # SEMANTIC: real call but a different value claimed -> the (fixture) judge rejects it.
    p = _preview(_R1_TEXT, _r1_ledger(),
                 '{"path_status":"unsupported","path_score":0,"issues":["a: claimed 2.0, delivered 1.47"]}', True)
    assert p.status == pr.ZERO


def test_candidate_missing_coverage_semantic_is_zero():
    p = _preview(_R1_TEXT, _r1_ledger(),
                 '{"path_status":"unsupported","path_score":0,"issues":["missing months for the aggregate"]}', True)
    assert p.status == pr.ZERO


def test_candidate_genuine_alternative_is_scored():
    # a genuine alternative source that supports the same value must NOT be false-rejected.
    p = _preview(_R1_TEXT, _r1_ledger(),
                 '{"path_status":"supported","path_score":1.0,"issues":[]}', True)
    assert p.status == pr.SCORED and p.score == 1.0


# ===========================================================================
# 5b. Deterministic value faithfulness (check_claimed_values) -- truth-by-design.
# Catches value FABRICATION with no model judgment; must NEVER false-reject an
# honest report (mis-attribution / unit paraphrase / derived values stay valid).
# ===========================================================================
def _cv(report_text, ledger):
    return pr.check_claimed_values(pr.parse_report(report_text), ledger)


def test_value_present_is_valid():
    led = _ledger("ep1", [_obs("obs_1", "read_document", "National defense December 1940 = 375", order=1, gen=0, deliver=1)])
    txt = '{"answer":"375","path":[{"id":"a","observation":"obs_1","claim":"December 1940 column = 375 (millions)"}]}'
    assert _cv(txt, led).valid


def test_value_fabricated_leaf_is_invalid():
    led = _ledger("ep1", [_obs("obs_1", "read_document", "December 1940 = 375; November = 287", order=1, gen=0, deliver=1)])
    txt = '{"answer":"400","path":[{"id":"a","observation":"obs_1","claim":"December 1940 column = 400 (millions)"}]}'
    chk = _cv(txt, led)
    assert not chk.valid and "fabricated_value" in chk.codes


def test_value_misattributed_but_present_is_left_to_judge():
    # 375 IS delivered (it is December); claiming it for January is a wrong-row error the JUDGE owns.
    led = _ledger("ep1", [_obs("obs_1", "read_document", "Jan 125 ... Dec 375", order=1, gen=0, deliver=1)])
    txt = '{"answer":"375","path":[{"id":"a","observation":"obs_1","claim":"January column = 375"}]}'
    assert _cv(txt, led).valid          # deterministic gate does NOT fire; the judge catches the mis-attribution


def test_value_compute_and_dependent_steps_exempt():
    # a derived value (0.17) need not appear in a read; compute / depends_on steps are not leaf lookups.
    assert _cv(_R1_TEXT, _r1_ledger()).valid


def test_value_unit_paraphrase_tolerated():
    # leaf claims "1.47 trillion" while the delivered cell is "1,470,000" (thousands) -> scale match, valid.
    led = _ledger("ep1", [_obs("obs_1", "read_document", "grand total 1,470,000 (thousands of dollars)", order=1, gen=0, deliver=1)])
    txt = '{"answer":"1.47 trillion","path":[{"id":"a","observation":"obs_1","claim":"grand total = 1.47 trillion"}]}'
    assert _cv(txt, led).valid


def test_value_years_and_small_ints_not_flagged():
    # a plausible year (1940) and a 1-2 digit count absent from bytes must NOT be flagged as fabricated.
    led = _ledger("ep1", [_obs("obs_1", "read_document", "Veterans Administration table", order=1, gen=0, deliver=1)])
    txt = '{"answer":"x","path":[{"id":"a","observation":"obs_1","claim":"the 1940 column across 3 rows"}]}'
    assert _cv(txt, led).valid


def test_candidate_fabricated_value_deterministic_zero_before_judge():
    # a glowing supported verdict + correct answer cannot rescue a deterministically fabricated value.
    led = _ledger("ep1", [_obs("obs_1", "read_document", "December 1940 = 375", order=1, gen=0, deliver=1)])
    txt = '{"answer":"400","path":[{"id":"a","observation":"obs_1","claim":"December 1940 = 400 (millions)"}]}'
    rep = pr.parse_report(txt)
    ref = pr.check_references(rep, led)
    vc = pr.check_claimed_values(rep, led)
    p = pr.candidate_preview(report=rep, ref=ref, verdict=pr.parse_support_verdict(_V_SUP),
                             answer_correct=True, values=vc)
    assert p.status == pr.ZERO and "fabricated value" in p.reason


def test_value_check_noop_on_malformed():
    led = _ledger("ep1", [_obs("obs_1", "read_document", "x", order=1, gen=0, deliver=1)])
    assert pr.check_claimed_values(pr.parse_report('{"answer":"1","path":[]}'), led).valid


# ===========================================================================
# 6. Bounded judge retries (Section 4) -- never "retry until positive"
# ===========================================================================
def test_retry_stops_at_max_unknown():
    calls = {"n": 0}

    def jf(_req):
        calls["n"] += 1
        return _V_UNK
    v = pr.judge_with_retries({}, jf, max_retries=2)
    assert v["status"] == "unknown" and v["attempts"] == 3 and calls["n"] == 3


def test_retry_returns_first_ok():
    v = pr.judge_with_retries({}, lambda _r: _V_SUP, max_retries=2)
    assert v["status"] == "ok" and v["attempts"] == 1


def test_retry_recovers_on_second():
    seq = iter([_V_UNK, _V_SUP])

    def jf(_req):
        return next(seq)
    v = pr.judge_with_retries({}, jf, max_retries=2)
    assert v["status"] == "ok" and v["attempts"] == 2


def test_retry_does_not_retry_unsupported():
    calls = {"n": 0}

    def jf(_req):
        calls["n"] += 1
        return _V_UNSUP
    v = pr.judge_with_retries({}, jf, max_retries=3)
    assert v["status"] == "ok" and v["path_status"] == "unsupported" and calls["n"] == 1


def test_retry_none_judge_makes_no_call():
    v = pr.judge_with_retries({}, None, max_retries=2)
    assert v["status"] == "unknown" and v["attempts"] == 0 and v["kind"] == "no_judge"


def test_retry_judge_fault_is_unknown():
    def jf(_req):
        raise RuntimeError("judge 500")
    v = pr.judge_with_retries({}, jf, max_retries=1)
    assert v["status"] == "unknown" and v["kind"] == "judge_fault" and v["attempts"] == 2


# ===========================================================================
# 7. Result integrity (Section 8)
# ===========================================================================
def _res(eid, status, score):
    return {"episode_id": eid, "candidate": {"status": status, "score": score}}


def test_integrity_ok():
    rep = pr.check_result_integrity(
        [_res("e1", pr.SCORED, 0.5), _res("e2", pr.ZERO, 0.0), _res("e3", pr.UNKNOWN, None)],
        expected_ids=["e1", "e2", "e3"])
    assert rep["ok"] and rep["counts"][pr.UNKNOWN] == 1


def test_integrity_duplicate_ids():
    rep = pr.check_result_integrity([_res("e1", pr.SCORED, 0.5), _res("e1", pr.ZERO, 0.0)])
    assert not rep["ok"] and any("duplicate" in p for p in rep["problems"])


def test_integrity_id_mismatch():
    rep = pr.check_result_integrity([_res("e1", pr.SCORED, 0.5)], expected_ids=["e1", "e2"])
    assert not rep["ok"] and any("mismatch" in p for p in rep["problems"])


def test_integrity_scored_nonfinite_flagged():
    rep = pr.check_result_integrity([_res("e1", pr.SCORED, float("nan"))])
    assert not rep["ok"]


def test_integrity_unknown_carrying_score_flagged():
    rep = pr.check_result_integrity([_res("e1", pr.UNKNOWN, 0.9)])
    assert not rep["ok"] and any("UNKNOWN but carries score" in p for p in rep["problems"])


def test_integrity_zero_with_nonzero_score_flagged():
    rep = pr.check_result_integrity([_res("e1", pr.ZERO, 0.4)])
    assert not rep["ok"]


def test_integrity_unknown_not_counted_as_rejection():
    # UNKNOWN is retained in its own bucket, never folded into ZERO/rejections.
    rep = pr.check_result_integrity([_res("e1", pr.UNKNOWN, None), _res("e2", pr.ZERO, 0.0)])
    assert rep["counts"][pr.UNKNOWN] == 1 and rep["counts"][pr.ZERO] == 1


# ===========================================================================
# 8. No-network property (Section 7.1): the pure layer imports no network libs
# ===========================================================================
def test_no_network_imports_in_pure_layer():
    src = open(os.path.join(_HERE, "..", "path_report.py"), encoding="utf-8").read()
    for banned in ("import requests", "import aiohttp", "import http.client",
                   "urllib.request", "import socket", "from urllib"):
        assert banned not in src, f"pure layer must not import networking ({banned})"


def test_end_to_end_pipeline_needs_no_model():
    # parse -> refs -> build request -> stub judge -> candidate, entirely on CPU.
    rep = pr.parse_report(_R1_TEXT)
    led = _r1_ledger()
    ref = pr.check_references(rep, led)
    req = pr.build_support_request("Q", "", rep, led)
    verdict = pr.judge_with_retries(req, lambda _r: _V_SUP, max_retries=1)
    prev = pr.candidate_preview(report=rep, ref=ref, verdict=verdict, answer_correct=True)
    assert prev.status == pr.SCORED


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
