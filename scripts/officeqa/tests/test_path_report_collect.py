#!/usr/bin/env python3
"""CPU tests for the path-report COLLECTOR state machine (observation-ID issuance,
delivery tracking, terminal-event handling). No model, no network, no GPU.

Run standalone:
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_path_report_collect.py
Also importable by pytest.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import path_report as pr                       # noqa: E402
import path_report_collect as prc              # noqa: E402
import path_report_pilot as prp                # noqa: E402  (score_episode: the Deliverable-B seam)


# ---------------------------------------------------------------------------
def _actor(script, seen=None):
    """Replay a scripted list of ActorTurns; optionally snapshot messages seen per call."""
    def actor_fn(messages, req):
        if seen is not None:
            seen.append([dict(m) for m in messages])
        if req < len(script):
            return script[req]
        return {"text": '{"answer":"DATA NOT AVAILABLE","path":[]}', "tool_calls": [], "error": None}
    return actor_fn


def _tools(mapping):
    def tool_fn(name, args):
        return mapping.get(name, f"No results for {name}.")
    return tool_fn


def _turn(text, calls=None, error=None):
    return {"text": text, "tool_calls": calls or [], "error": error}


def _submit(answer, path):
    """A turn that FINISHES via the submit_report tool (the Replace contract)."""
    return _turn("", calls=[("submit_report", {"answer": answer, "path": path})])


# a reusable happy-path script: reads @turn0, compute @turn1, terminal @turn2
_TERMINAL = ('{"answer":"0.17","path":['
             '{"id":"a","observation":"obs_1","claim":"1941 total 1.47"},'
             '{"id":"b","observation":"obs_2","claim":"1942 total 1.30"},'
             '{"id":"c","observation":"obs_3","depends_on":["a","b"],"claim":"1.47-1.30=0.17"}]}')
_HAPPY = [
    _turn("read both", [("read_document", {"file_name": "b1941.txt"}),
                        ("read_document", {"file_name": "b1942.txt"})]),
    _turn("compute", [("compute", {"code": "print(1.47-1.30)"})]),
    _turn(_TERMINAL),
]
_HAPPY_TOOLS = {"read_document": "grand total 1.47/1.30 trillion", "compute": "0.17"}


def _run_happy(**kw):
    return prc.run_episode(_actor(_HAPPY), _tools(_HAPPY_TOOLS), "decrease?", episode_id="ep", **kw)


# ---------------------------------------------------------------------------
# observation IDs + delivery
# ---------------------------------------------------------------------------
def test_observation_ids_issued_in_order_and_unique():
    rec = _run_happy()
    ids = [o["observation_id"] for o in rec["observations"]]
    assert ids == ["obs_1", "obs_2", "obs_3"]
    assert len(set(ids)) == 3


def test_delivered_text_is_exact_tool_output():
    rec = _run_happy()
    o = {x["observation_id"]: x for x in rec["observations"]}
    assert o["obs_1"]["delivered_text"] == "grand total 1.47/1.30 trillion"
    assert o["obs_3"]["delivered_text"] == "0.17"


def test_marker_prefixes_delivered_content_actor_sees():
    seen = []
    prc.run_episode(_actor(_HAPPY, seen), _tools(_HAPPY_TOOLS), "decrease?", episode_id="ep")
    # at turn 1 the actor sees the two tool messages from turn 0, each marked with its obs id
    tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"].startswith("[observation obs_1]\n")
    assert tool_msgs[1]["content"].startswith("[observation obs_2]\n")


def test_delivery_tracking_generated_then_delivered_next_request():
    rec = _run_happy()
    o = {x["observation_id"]: x for x in rec["observations"]}
    assert o["obs_1"]["generated_by_request"] == 0 and o["obs_1"]["delivered_to_request"] == 1
    assert o["obs_2"]["generated_by_request"] == 0 and o["obs_2"]["delivered_to_request"] == 1
    assert o["obs_3"]["generated_by_request"] == 1 and o["obs_3"]["delivered_to_request"] == 2
    assert rec["n_delivered"] == 3


def test_actor_cannot_mint_ids_fake_ref_scores_zero():
    # actor references obs_99 which the runtime never issued -> nonexistent -> ZERO
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"7","path":[{"id":"a","observation":"obs_99","claim":"x"}]}')]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="ep")
    ledger = pr.EpisodeLedger.from_record(rec)
    report = pr.parse_report(rec["terminal_text"])
    ref = pr.check_references(report, ledger)
    assert not ref.valid and "nonexistent" in ref.codes


# ---------------------------------------------------------------------------
# parallel-batch chronology (the subtle one)
# ---------------------------------------------------------------------------
def test_parallel_batch_makes_compute_input_invisible():
    # read AND compute issued in the SAME turn 0 -> both delivered at turn 1; the compute
    # (generated at 0) could not have seen the read (delivered at 1) -> impossible chronology.
    terminal = ('{"answer":"0.17","path":['
                '{"id":"a","observation":"obs_1","claim":"1941 1.47"},'
                '{"id":"b","observation":"obs_2","depends_on":["a"],"claim":"compute uses obs_1"}]}')
    script = [_turn("read+compute together",
                    [("read_document", {"file_name": "b.txt"}), ("compute", {"code": "print(1)"})]),
              _turn(terminal)]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "1.47", "compute": "0.17"}), "q", episode_id="ep")
    o = {x["observation_id"]: x for x in rec["observations"]}
    assert o["obs_1"]["generated_by_request"] == 0 and o["obs_1"]["delivered_to_request"] == 1
    assert o["obs_2"]["generated_by_request"] == 0            # compute generated in the same batch
    ref = pr.check_references(pr.parse_report(rec["terminal_text"]), pr.EpisodeLedger.from_record(rec))
    assert not ref.valid and "impossible_chronology" in ref.codes


def test_end_to_end_impossible_chronology_scores_zero_even_if_supported():
    terminal = ('{"answer":"0.17","path":['
                '{"id":"a","observation":"obs_1","claim":"1.47"},'
                '{"id":"b","observation":"obs_2","depends_on":["a"],"claim":"compute uses obs_1"}]}')
    script = [_turn("batch", [("read_document", {"file_name": "b.txt"}), ("compute", {"code": "print(1)"})]),
              _turn(terminal)]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "1.47", "compute": "0.17"}), "q", episode_id="ep")
    report = pr.parse_report(rec["terminal_text"])
    ref = pr.check_references(report, pr.EpisodeLedger.from_record(rec))
    prev = pr.candidate_preview(report=report, ref=ref,
                                verdict=pr.parse_support_verdict('{"path_status":"supported","path_score":0.9}'),
                                answer_correct=True)
    assert prev.status == pr.ZERO


# ---------------------------------------------------------------------------
# undelivered results
# ---------------------------------------------------------------------------
def test_undelivered_when_consuming_actor_errors():
    # read @turn0, then the actor request that would consume it (turn1) errors -> undelivered.
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn("", error="boom")]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="ep")
    assert rec["termination"] == "actor_error"
    o = rec["observations"][0]
    assert o["delivered_to_request"] is None and o["delivered_text"] is None
    assert rec["n_delivered"] == 0
    # the raw execution is still retained for audit
    assert rec["raw_executions"][0]["full_output"] == "x"


def test_undelivered_reference_scores_zero():
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn("", error="boom")]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="ep")
    # a report citing the undelivered obs would be rejected deterministically
    report = pr.parse_report('{"answer":"1","path":[{"id":"a","observation":"obs_1","claim":"c"}]}')
    ref = pr.check_references(report, pr.EpisodeLedger.from_record(rec))
    assert not ref.valid and "undelivered" in ref.codes


# ---------------------------------------------------------------------------
# terminal handling (no synthesis, terminal-only)
# ---------------------------------------------------------------------------
def test_terminal_is_last_assistant_event_not_tool_output():
    # a tool returns a JSON blob that LOOKS like a report; it must never become the terminal.
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"real","path":[{"id":"a","observation":"obs_1","claim":"c"}]}')]
    rec = prc.run_episode(_actor(script), _tools({"read_document": '{"answer":"FAKE","path":[]}'}), "q", episode_id="ep")
    assert rec["termination"] == "terminal_report"
    assert json.loads(rec["terminal_text"])["answer"] == "real"     # not "FAKE" from the tool output


def test_no_final_answer_forcing():
    rec = _run_happy()
    assert "<FINAL_ANSWER>" not in rec["terminal_text"]
    assert rec["termination"] == "terminal_report"


def test_turn_cap_tool_call_is_not_synthesized():
    # model keeps calling tools even on the final nudge turn -> recorded, NOT turned into an answer.
    script = [_turn("call", [("read_document", {"file_name": "b.txt"})]),
              _turn("still calling", [("read_document", {"file_name": "b2.txt"})])]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="ep", max_turns=2)
    assert rec["termination"] == "turn_cap_tool_call"
    # the final turn's tool call was NOT dispatched (only turn-0's read exists)
    assert rec["n_observations"] == 1
    assert pr.parse_report(rec["terminal_text"]).kind == "malformed"   # no report object -> interface failure


def test_turn_cap_empty_terminal():
    script = [_turn("call", [("read_document", {"file_name": "b.txt"})]), _turn("")]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="ep", max_turns=2)
    assert rec["termination"] == "empty_terminal"


def test_immediate_terminal_no_tools():
    script = [_turn('{"answer":"DATA NOT AVAILABLE","path":[]}')]
    rec = prc.run_episode(_actor(script), _tools({}), "q", episode_id="ep")
    assert rec["termination"] == "terminal_report" and rec["n_observations"] == 0
    assert pr.parse_report(rec["terminal_text"]).is_abstention


# ---------------------------------------------------------------------------
# failed exploration retained; tool faults; outcome classification
# ---------------------------------------------------------------------------
def test_failed_exploration_retained_in_raw_log():
    # an early failed grep (not cited in the final report) is still recorded.
    script = [_turn("grep", [("grep_documents", {"pattern": "nope"})]),
              _turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"1","path":[{"id":"a","observation":"obs_2","claim":"c"}]}')]
    tools = {"grep_documents": "No matches for 'nope'.", "read_document": "1.47"}
    rec = prc.run_episode(_actor(script), _tools(tools), "q", episode_id="ep")
    execs = {e["observation_id"]: e for e in rec["raw_executions"]}
    assert execs["obs_1"]["full_output"].startswith("No matches")     # the dead end is preserved
    assert execs["obs_1"]["outcome"] == "empty"


def test_tool_fault_becomes_delivered_error():
    def boom_tool(name, args):
        raise RuntimeError("kaboom")
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"1","path":[{"id":"a","observation":"obs_1","claim":"c"}]}')]
    rec = prc.run_episode(_actor(script), boom_tool, "q", episode_id="ep")
    o = rec["observations"][0]
    assert o["outcome"] == "error" and o["delivered_text"].startswith("Error: tool read_document failed")


def test_outcome_classification():
    assert prc.classify_outcome("Error: no document") == "error"
    assert prc.classify_outcome("No matches for 'x'.") == "empty"
    assert prc.classify_outcome("No passages found for x.") == "empty"
    assert prc.classify_outcome("   ") == "empty"
    assert prc.classify_outcome("treasury_1941.txt:14: National defense 2,602") == "ok"


# ---------------------------------------------------------------------------
# schema / integration
# ---------------------------------------------------------------------------
def test_record_schema_has_required_fields():
    rec = _run_happy()
    for k in ("episode_id", "question", "question_requirements", "terminal_text", "termination",
              "answer_correct", "observations", "raw_executions", "raw_turns", "collector_version"):
        assert k in rec, f"missing {k}"
    assert rec["answer_correct"] is None                     # reviewer labels this separately
    for o in rec["observations"]:
        for k in ("observation_id", "order", "tool", "args", "outcome",
                  "delivered_text", "generated_by_request", "delivered_to_request"):
            assert k in o, f"observation missing {k}"


def test_record_consumable_by_ledger_and_scorer():
    rec = _run_happy()
    ledger = pr.EpisodeLedger.from_record(rec)              # must not raise (unique ids etc.)
    assert ledger.get("obs_3").is_compute
    report = pr.parse_report(rec["terminal_text"])
    ref = pr.check_references(report, ledger)
    assert report.ok and ref.valid


# ---------------------------------------------------------------------------
# Deliverable-B seam: a REAL collector record -> path_report_pilot.score_episode.
# This is the exact end-to-end path the real-rollout usability run will take (minus
# the served judge). It locks the auto-labeling wiring AND the contract invariants
# (three separate labels; zero candidate reward for a fabricated/unsupported path
# even when the answer is correct) at the integration boundary, not just the unit.
# ---------------------------------------------------------------------------
def _supported_judge(_req):
    return '{"path_status":"supported","path_score":0.85,"issues":[]}'


def test_score_episode_auto_labels_answer_from_key_by_uid():
    # the answer key is keyed by uid; the collector stamps episode_id = uid, so a valid
    # supported report is auto-labeled answer_correct + graded, with NO human and NO judge
    # touching the answer label.
    rec = prc.run_episode(_actor(_HAPPY), _tools(_HAPPY_TOOLS), "decrease?", episode_id="UID0001")
    out = prp.score_episode(rec, _supported_judge, answer_key={"UID0001": "0.17"})
    assert out["episode_id"] == "UID0001"
    assert out["answer_correct"] is True and out["answer_correct_source"] == "answer_key"
    assert out["ref_valid"] is True and out["value_valid"] is True
    assert out["support_verdict"]["status"] == "ok"          # judge consulted (report valid + gates pass)
    assert out["candidate"]["status"] == pr.SCORED and out["candidate"]["score"] > 0


def test_score_episode_wrong_gold_zeros_candidate_but_records_support_separately():
    # answer-correctness and support are SEPARATE labels. A wrong final answer zeros the
    # CANDIDATE reward (there is nothing to reward for a wrong answer), yet the judge's support
    # verdict is still recorded independently -- the two are never conflated into one number.
    rec = prc.run_episode(_actor(_HAPPY), _tools(_HAPPY_TOOLS), "decrease?", episode_id="UID0001")
    out = prp.score_episode(rec, _supported_judge, answer_key={"UID0001": "9.99"})
    assert out["answer_correct"] is False and out["answer_correct_source"] == "answer_key"
    assert out["ref_valid"] is True and out["value_valid"] is True
    assert out["support_verdict"]["status"] == "ok"              # judge WAS consulted ...
    assert out["labels"]["path_status"] == pr.SUPPORTED          # ... and support recorded separately
    assert out["candidate"]["status"] == pr.ZERO                 # but a wrong answer earns no reward
    assert "wrong final answer" in out["candidate"]["reason"]


def test_score_episode_without_key_leaves_answer_unlabeled():
    rec = prc.run_episode(_actor(_HAPPY), _tools(_HAPPY_TOOLS), "decrease?", episode_id="UID0001")
    out = prp.score_episode(rec, _supported_judge, answer_key=None)
    assert out["answer_correct"] is None and out["answer_correct_source"] is None


def test_score_episode_correct_answer_but_fake_ref_scores_zero():
    # THE core contract invariant at the seam: the answer is correct AND the judge would rubber-
    # stamp it, but the cited observation was never issued by the runtime -> deterministic ZERO,
    # and the judge is never even consulted. answer_correct stays True (labels not conflated).
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"7","path":[{"id":"a","observation":"obs_99","claim":"defense outlay"}]}')]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x"}), "q", episode_id="UID0007")
    out = prp.score_episode(rec, _supported_judge, answer_key={"UID0007": "7"})
    assert out["answer_correct"] is True                      # answer is right ...
    assert out["ref_valid"] is False                          # ... but the path is fabricated
    assert out["support_verdict"]["status"] == "skipped"      # judge never consulted on an invalid path
    assert out["candidate"]["status"] == pr.ZERO              # zero reward despite the correct answer


def test_score_episode_correct_answer_but_fabricated_value_scores_zero():
    # defense-in-depth: a leaf claim asserts a value absent from the entire delivered trace.
    # The deterministic value gate zeros it BEFORE the judge, even with a correct answer.
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"9999","path":[{"id":"a","observation":"obs_1","claim":"1941 total = 9,999 billion"}]}')]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "grand total 1.47 trillion"}),
                          "q", episode_id="UID0009")
    out = prp.score_episode(rec, _supported_judge, answer_key={"UID0009": "9999"})
    assert out["answer_correct"] is True                      # answer matches the (planted) key ...
    assert out["value_valid"] is False                        # ... but 9,999 is nowhere in the trace
    assert out["support_verdict"]["status"] == "skipped"
    assert out["candidate"]["status"] == pr.ZERO


def test_score_episodes_batch_summary_keeps_labels_separate():
    # the batch summarizer reports deterministic-gate failures and answer-correctness as
    # DISTINCT tallies (never a single conflated "reward-correct" count).
    good = prc.run_episode(_actor(_HAPPY), _tools(_HAPPY_TOOLS), "q", episode_id="UID0001")
    fake = prc.run_episode(_actor([_turn("read", [("read_document", {"file_name": "b.txt"})]),
                                   _turn('{"answer":"7","path":[{"id":"a","observation":"obs_99","claim":"c"}]}')]),
                           _tools({"read_document": "x"}), "q", episode_id="UID0007")
    key = {"UID0001": "0.17", "UID0007": "7"}
    _res, summary = prp.score_episodes([good, fake], _supported_judge, answer_key=key)
    assert summary["n_episodes"] == 2
    assert summary["n_ref_invalid"] == 1                      # the fake-ref episode
    assert summary["answer_correct_true"] == 2                # BOTH answers are correct ...
    assert summary["n_scored"] == 1                           # ... yet only one path earns a score


# ---------------------------------------------------------------------------
# SUBMIT-TOOL contract (Replace) + endgame window (the report-yield fix)
# ---------------------------------------------------------------------------
def test_submit_report_call_is_terminal():
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _submit("0.17", [{"id": "a", "observation": "obs_1", "claim": "1941 total 1.47"}])]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "1.47"}), "q",
                          episode_id="ep", require_submit=True)
    assert rec["termination"] == "terminal_report"
    rep = pr.parse_report(rec["terminal_text"])
    assert rep.ok and rep.answer == "0.17"                      # the report came from the tool ARGS


def test_submit_report_stringified_path_is_decoded():
    # REGRESSION (Deliverable B v2): the parser returned the nested `path` array as a JSON
    # STRING, so every report parsed as "'path' must be a list". The collector must decode it.
    path_str = json.dumps([{"id": "a", "observation": "obs_1", "claim": "1941 total 1.47"}])
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn("", calls=[("submit_report", {"answer": "0.17", "path": path_str})])]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "1.47"}), "q",
                          episode_id="ep", require_submit=True)
    rep = pr.parse_report(rec["terminal_text"])
    assert rep.ok and rep.answer == "0.17" and len(rep.steps) == 1     # path decoded to a real list


def test_submit_report_whole_args_stringified_is_decoded():
    # some parsers stringify the ENTIRE arguments object, not just the nested field.
    args_str = json.dumps({"answer": "DATA NOT AVAILABLE", "path": []})
    rec = prc.run_episode(_actor([_turn("", calls=[("submit_report", args_str)])]),
                          _tools({}), "q", episode_id="ep", require_submit=True)
    assert pr.parse_report(rec["terminal_text"]).is_abstention


def test_require_submit_no_tool_call_turn_is_nudged_then_submits():
    # in Replace mode a plain-text turn (looks like a report, but is not a submit call) does NOT
    # finish -> the actor is nudged and the episode continues; it then submits.
    seen = []
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _turn('{"answer":"0.17","path":[]}'),             # TEXT, not a submit -> must be nudged
              _submit("0.17", [{"id": "a", "observation": "obs_1", "claim": "1.47"}])]
    rec = prc.run_episode(_actor(script, seen), _tools({"read_document": "1.47"}), "q",
                          episode_id="ep", require_submit=True)
    assert rec["termination"] == "terminal_report"
    last_user = [m for m in seen[2] if m.get("role") == "user"][-1]
    assert "submit_report" in last_user["content"]              # the nudge was shown before the 3rd request
    assert pr.parse_report(rec["terminal_text"]).answer == "0.17"


def test_require_submit_never_submits_hits_cap_with_no_report():
    # keeps emitting text, never submits -> no_submit_at_cap, and NOTHING is synthesized.
    script = [_turn("thinking"), _turn("still thinking"), _turn("more")]
    rec = prc.run_episode(_actor(script), _tools({}), "q", episode_id="ep",
                          require_submit=True, max_turns=3)
    assert rec["termination"] == "no_submit_at_cap"
    assert pr.parse_report(rec["terminal_text"]).kind == "malformed"


def test_submit_terminates_even_in_legacy_mode():
    # a submit_report call always finishes, regardless of require_submit.
    rec = prc.run_episode(_actor([_submit("DATA NOT AVAILABLE", [])]), _tools({}), "q",
                          episode_id="ep", require_submit=False)
    assert rec["termination"] == "terminal_report"
    assert pr.parse_report(rec["terminal_text"]).is_abstention


def test_endgame_locks_search_grep_allows_compute():
    # max_turns=4, window=2 -> turns 2,3 in-window. turn2 issues search (LOCKED) + compute (allowed).
    script = [_turn("r0", [("read_document", {"file_name": "b.txt"})]),          # turn0 obs_1
              _turn("r1", [("read_document", {"file_name": "b2.txt"})]),         # turn1 obs_2
              _turn("t2", [("search_documents", {"query": "x"}),                 # turn2 obs_3 (locked)
                           ("compute", {"code": "print(1)"})]),                  # turn2 obs_4 (ok)
              _submit("1", [{"id": "a", "observation": "obs_1", "claim": "c"}])]  # turn3 submit
    tools = {"read_document": "1.47", "search_documents": "SECRET_HIT", "compute": "1"}
    rec = prc.run_episode(_actor(script), _tools(tools), "q", episode_id="ep",
                          require_submit=True, nudge_window=2, max_turns=4)
    o = {x["observation_id"]: x for x in rec["observations"]}
    assert o["obs_3"]["tool"] == "search_documents" and o["obs_3"]["outcome"] == "locked"
    assert "SECRET_HIT" not in o["obs_3"]["delivered_text"] and "disabled" in o["obs_3"]["delivered_text"]
    assert o["obs_4"]["tool"] == "compute" and o["obs_4"]["outcome"] == "ok" and o["obs_4"]["delivered_text"] == "1"


def test_endgame_nudge_injected_in_window():
    seen = []
    script = [_turn("r0", [("read_document", {"file_name": "b.txt"})]),          # turn0 (not in window)
              _turn("r1", [("compute", {"code": "print(1)"})]),                  # turn1 (in window)
              _submit("1", [{"id": "a", "observation": "obs_1", "claim": "c"}])]  # turn2 (is_last)
    prc.run_episode(_actor(script, seen), _tools({"read_document": "x", "compute": "1"}), "q",
                    episode_id="ep", require_submit=True, nudge_window=2, max_turns=3)
    user_msgs = [m["content"] for m in seen[1] if m.get("role") == "user"]
    assert any("step(s) left" in c or "disabled" in c for c in user_msgs)


def test_submit_only_window_locks_all_tools_except_submit():
    # Innermost endgame: only submit_report may run; compute (and any retrieval) gets a 'locked'
    # redirect. Here nudge_window=3 (outer, search/grep) but submit_only_window=2 (inner, all tools).
    script = [_turn("gather", [("read_document", {"file_name": "b.txt"})]),          # turn0: allowed
              _turn("calc",   [("compute", {"code": "print(1)"})]),                  # turn1: submit-only -> locked
              _submit("1", [{"id": "a", "observation": "obs_1", "claim": "c"}])]      # turn2: is_last submit
    rec = prc.run_episode(_actor(script), _tools({"read_document": "x", "compute": "1"}), "q",
                          episode_id="ep", require_submit=True,
                          nudge_window=3, submit_only_window=2, max_turns=3)
    o = {x["observation_id"]: x for x in rec["observations"]}
    assert o["obs_1"]["tool"] == "read_document" and o["obs_1"]["outcome"] != "locked"   # turn0 ran for real
    assert o["obs_2"]["tool"] == "compute" and o["obs_2"]["outcome"] == "locked"          # turn1 in submit-only
    assert "submit_report" in o["obs_2"]["delivered_text"]                                # got the redirect
    assert rec["termination"] == "terminal_report"


def test_submit_mode_record_scores_end_to_end():
    # a submit-produced record flows through the SAME scorer as a text-produced one.
    script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
              _submit("0.17", [{"id": "a", "observation": "obs_1", "claim": "1941 total 1.47"}])]
    rec = prc.run_episode(_actor(script), _tools({"read_document": "grand total 1.47 trillion"}),
                          "decrease?", episode_id="UID0001", require_submit=True)
    out = prp.score_episode(rec, _supported_judge, answer_key={"UID0001": "0.17"})
    assert out["answer_correct"] is True and out["ref_valid"] is True
    assert out["candidate"]["status"] == pr.SCORED


# --- parallel collection driver (the 133-hard enabler) ---------------------
def test_collect_records_sequential_preserves_order():
    qs = [{"uid": f"U{i}"} for i in range(6)]
    out = prc._collect_records(qs, lambda idx, q: {"episode_id": q["uid"], "idx": idx}, concurrency=1)
    assert [r["episode_id"] for r in out] == [f"U{i}" for i in range(6)]
    assert [r["idx"] for r in out] == list(range(6))


def test_collect_records_parallel_overlaps_and_preserves_order():
    # A runner that records how many episodes are in-flight at once. If _collect_records
    # secretly ran sequentially, max_live would be exactly 1; >=2 proves real overlap.
    # Completion order is deliberately shuffled by staggered sleeps, yet output stays in
    # INPUT order -- the contract the offline scorer + A/B/C comparison rely on.
    import threading
    import time
    qs = [{"uid": f"U{i}"} for i in range(8)]
    lock = threading.Lock()
    state = {"live": 0, "max": 0}

    def runner(idx, q):
        with lock:
            state["live"] += 1
            state["max"] = max(state["max"], state["live"])
        time.sleep(0.05 * ((idx % 3) + 1))     # stagger completions out of input order
        with lock:
            state["live"] -= 1
        return {"episode_id": q["uid"], "idx": idx}

    out = prc._collect_records(qs, runner, concurrency=4)
    assert [r["episode_id"] for r in out] == [f"U{i}" for i in range(8)]
    assert [r["idx"] for r in out] == list(range(8))
    assert state["max"] >= 2                    # concurrency actually engaged


def test_score_episodes_parallel_matches_sequential_and_order():
    # Scoring many hard reports must parallelize (judge HTTP is I/O-bound) WITHOUT changing
    # results or order. Build 8 distinct submit-produced records, score both ways, compare.
    def judge(_req):
        return '{"path_status":"supported","path_score":0.9,"issues":[]}'
    recs = []
    for i in range(8):
        script = [_turn("read", [("read_document", {"file_name": "b.txt"})]),
                  _submit(str(i), [{"id": "a", "observation": "obs_1", "claim": f"total is {i}"}])]
        recs.append(prc.run_episode(_actor(script), _tools({"read_document": f"grand total {i}"}),
                                    "q?", episode_id=f"UID{i:04d}", require_submit=True))
    seq, s_sum = prp.score_episodes(recs, judge, max_retries=1, concurrency=1)
    par, p_sum = prp.score_episodes(recs, judge, max_retries=1, concurrency=4)
    order = [f"UID{i:04d}" for i in range(8)]
    assert [r["episode_id"] for r in seq] == order            # sequential keeps input order
    assert [r["episode_id"] for r in par] == order            # parallel preserves it too
    # identical per-episode verdicts + identical summary regardless of concurrency
    assert [(r["candidate"]["status"], r["candidate"]["score"]) for r in par] == \
           [(r["candidate"]["status"], r["candidate"]["score"]) for r in seq]
    assert p_sum == s_sum


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
