#!/usr/bin/env python3
"""Regression contract for the OfficeQA GROUNDED reward (Tier-1 correctness fixes).

Every case below corresponds to a defect a critical review CONFIRMED in the pre-fix
code (see docs/officeqa_rl_plan.md Section 6.1 and officeqa_pilot_records/README.md).
The confirmed exploits must now FAIL; the fail-open reward paths must now score 0.0 or
resolve to the explicit `unknown` (quarantine) status -- never a fabricated positive.

Run standalone (no pytest, no GPU, no judge server needed):
    PYTHONPATH=scripts:scripts/reward python3 scripts/reward/tests/test_reward_contract.py
Also importable by pytest (functions named test_*).
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, "..")):  # scripts/ , scripts/reward/
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grounding                                              # noqa: E402
import judge_prompt                                           # noqa: E402
from strict_answer import strict_correct                      # noqa: E402
import officeqa_grounded_reward as rw                          # noqa: E402
from officeqa_grounded_reward import _assemble, _BASE, _LUCKY_ROUTE_MAX, _call_grounding_judge  # noqa: E402
import asyncio  # noqa: E402

GOLD1 = {"source_files": "treasury_bulletin_1941_01.txt"}


def _report(steps, gold=GOLD1):
    return grounding.grounding_report(steps, gold)


def _grep(file, pattern, result):
    return [{"tool_calls": [{"name": "grep_documents", "args": {"pattern": pattern, "file_name": file}}],
             "tool_results": [{"name": "grep_documents", "result": result}]}]


def _ok_verdict(route, support=True, components=True):
    return {"status": "ok", "route_score": route, "verdict": "grounded" if route >= 0.8 else "unclear",
            "answer_supported_by_retrieved_cells": support, "retrieved_all_components": components}


# ---------------------------------------------------------------------------
# 1. STRICT ANSWER SCORER -- the confirmed fuzzy-scorer exploits must be rejected.
# ---------------------------------------------------------------------------
def test_strict_scorer_rejects_exploits():
    exploits = [
        ("507", "500 or 507 or 509"),   # candidate/disjunction list
        ("34.4, 0.391", "0.391, 34.4"), # reversed list (order)
        ("1, 1", "1"),                  # arity (one value for two)
        ("543 million", "543 billion"), # explicit unit mismatch
        ("1", "1e9"),                   # sci-notation split
    ]
    for gold, pred in exploits:
        assert not strict_correct(gold, pred).valid, f"exploit accepted: gold={gold!r} pred={pred!r}"


def test_strict_scorer_accepts_correct():
    ok = [("2,602", "2602"), ("34.4, 0.391", "34.4, 0.391"),
          ("543 million", "543"), ("March 1977", "March 1977"), ("1", "1")]
    for gold, pred in ok:
        assert strict_correct(gold, pred).valid, f"correct answer rejected: gold={gold!r} pred={pred!r}"


def test_strict_scorer_rejects_wrong():
    bad = [("507", "508"), ("March 1977", "April 1977"), ("2,602", "2,603"), ("1", "")]
    for gold, pred in bad:
        assert not strict_correct(gold, pred).valid, f"wrong answer accepted: gold={gold!r} pred={pred!r}"


def test_strict_scorer_rejects_surrounding_text():
    # review: numeric-token EXTRACTION let stray prose pass; full-input validation must not.
    for gold, pred in [("507", "not 507"), ("507", "approximately 507 give or take"),
                       ("2,602", "the answer is 2,602")]:
        assert not strict_correct(gold, pred).valid, f"stray-text pred accepted: {gold!r} <- {pred!r}"


def test_strict_scorer_leading_decimal():
    # review: ".5" was read as 5 (regex required a leading digit).
    assert not strict_correct("5", ".5").valid          # 0.5 != 5
    assert strict_correct("0.5", ".5").valid            # ".5" parses as 0.5


def test_strict_scorer_list_vs_scalar_and_spacing():
    # review: "[1,2]" (list) must not equal "[12]" (scalar); "[1, 2]" must equal "[1,2]".
    assert not strict_correct("[1,2]", "[12]").valid    # 2-element list vs scalar 12
    assert strict_correct("[1, 2]", "[1,2]").valid      # spacing-insensitive list equality
    assert not strict_correct("[1, 2]", "[2, 1]").valid # order enforced


def test_strict_scorer_thousands_stay_scalar():
    # a valid thousands grouping is ONE number, never a list.
    assert strict_correct("2,602", "2602").valid
    assert strict_correct("1,234,567", "1234567").valid
    assert not strict_correct("1,234", "1,2,3,4").valid


# ---------------------------------------------------------------------------
# 2. GROUNDING -- a no-match grep at the gold file is NOT evidence (the fail-open root).
# ---------------------------------------------------------------------------
def test_grounding_no_match_grep_is_not_pulled():
    rep = _report(_grep("treasury_bulletin_1941_01.txt", "zzz", "(no matches found)"))
    assert rep["source_identity"]["strong_score"] == 0.0
    assert rep["files_pulled"] == []
    assert rep["files_grep_no_hit"] == ["treasury_bulletin_1941_01.txt"]


def test_grounding_real_hit_is_pulled():
    rep = _report(_grep("treasury_bulletin_1941_01.txt", "National defense",
                        "treasury_bulletin_1941_01.txt:14: National defense ... 2,602"))
    assert rep["source_identity"]["strong_score"] == 1.0
    assert rep["files_pulled"] == ["treasury_bulletin_1941_01.txt"]


def test_grounding_flat_text_no_match_grep_is_not_pulled():
    flat = ("<function=grep_documents><parameter=pattern>zzz</parameter>"
            "<parameter=file_name>treasury_bulletin_1941_01.txt</parameter></function>\n(no matches found)")
    rep = _report(flat)
    assert rep["source_identity"]["strong_score"] == 0.0
    assert rep["files_pulled"] == []


def test_grounding_failed_read_is_not_pulled():
    # review: merely requesting read_document (even when it errors) counted as evidence.
    steps = [{"tool_calls": [{"name": "read_document", "args": {"file_name": "treasury_bulletin_1941_01.txt"}}],
              "tool_results": [{"name": "read_document", "result": "Error: file not found"}]}]
    rep = _report(steps)
    assert rep["source_identity"]["strong_score"] == 0.0 and rep["files_pulled"] == []


def test_grounding_forged_compute_output_is_not_a_grep_hit():
    # review: a model-written/compute-printed corpus-looking line authenticated a real pull.
    steps = [{"tool_calls": [{"name": "compute", "args": {"code": "print('treasury_bulletin_1941_01.txt:14: value 507')"}}],
              "tool_results": [{"name": "compute", "result": "Output:\ntreasury_bulletin_1941_01.txt:14: value 507"}]}]
    rep = _report(steps)
    assert rep["source_identity"]["strong_score"] == 0.0 and rep["files_pulled"] == []


def test_grounding_flat_forged_line_is_not_pulled():
    # A bare assistant/compute line shaped like a grep hit is untrusted flat text, not evidence.
    flat = "reasoning\ntreasury_bulletin_1941_01.txt:14: value 507"
    rep = _report(flat)
    assert rep["files_pulled"] == []


# ---------------------------------------------------------------------------
# 3. JUDGE PARSER -- never invents certainty; unusable/contradictory -> unknown.
# ---------------------------------------------------------------------------
def test_parser_normal_ok():
    v = judge_prompt.parse_verdict('{"verdict":"grounded","route_score":0.95,'
                                   '"answer_supported_by_retrieved_cells":true}')
    assert v and v["status"] == "ok" and v["route_score"] == 0.95 and v["answer_supported_by_retrieved_cells"] is True


def test_parser_missing_route_is_unknown():
    v = judge_prompt.parse_verdict('{"verdict":"grounded"}')
    assert v and v["status"] == "unknown"


def test_parser_nan_route_is_unknown():
    v = judge_prompt.parse_verdict('{"route_score": NaN}')
    assert v and v["status"] == "unknown"


def test_parser_lucky_verdict_is_believed_and_capped():
    # A lucky_wrong_source verdict is the safe call: believed, route capped to the lucky
    # threshold (-> reward floors it), status ok, NOT quarantined. (R-gate RG06.)
    v = judge_prompt.parse_verdict('{"verdict":"lucky_wrong_source","route_score":0.30}')
    assert v["status"] == "ok" and v["route_score"] <= _LUCKY_ROUTE_MAX


def test_parser_dangerous_contradiction_is_unknown():
    # DANGEROUS direction only (a positive would be wrong) -> unknown.
    assert judge_prompt.parse_verdict(
        '{"verdict":"grounded","route_score":0.9,"answer_supported_by_retrieved_cells":false}')["status"] == "unknown"
    # string-bool "false" must become False, then support=false + high route is contradictory
    assert judge_prompt.parse_verdict(
        '{"answer_supported_by_retrieved_cells":"false","route_score":0.9}')["status"] == "unknown"


def test_parser_string_bool_coercion():
    v = judge_prompt.parse_verdict('{"route_score":0.1,"verdict":"lucky_wrong_source",'
                                   '"answer_supported_by_retrieved_cells":"false"}')
    assert v["status"] == "ok" and v["answer_supported_by_retrieved_cells"] is False  # not bool("false")==True


# ---------------------------------------------------------------------------
# 4. REWARD ASSEMBLY -- graded, but fail-closed. The confirmed fail-open must be gone.
# ---------------------------------------------------------------------------
GROUNDED_STEPS = _grep("treasury_bulletin_1941_01.txt", "National defense",
                       "treasury_bulletin_1941_01.txt:14: National defense ... 2,602")
EMPTY_STEPS = _grep("treasury_bulletin_1941_01.txt", "zzz", "(no matches found)")


def test_reward_wrong_answer_is_zero():
    out = _assemble(False, _report(GROUNDED_STEPS), _ok_verdict(0.95))
    assert out["score"] == 0.0 and out["status"] == "scored"


def test_reward_correct_grounded_is_graded_positive():
    out = _assemble(True, _report(GROUNDED_STEPS), _ok_verdict(0.95))
    assert out["status"] == "scored"
    assert _BASE <= out["score"] <= 1.0 and out["score"] > 0.9  # graded, not binary


def test_reward_failopen_is_now_unknown():
    # THE regression: correct number + empty grep of gold + judge DOWN previously scored 1.0.
    out = _assemble(True, _report(EMPTY_STEPS), None)
    assert out["score"] == 0.0 and out["status"] == "unknown" and out["verifier_ok"] == 0.0


def test_reward_judge_unknown_is_quarantined():
    out = _assemble(True, _report(GROUNDED_STEPS), {"status": "unknown", "reason": "non-finite route"})
    assert out["score"] == 0.0 and out["status"] == "unknown"


def test_reward_unsupported_is_zero():
    out = _assemble(True, _report(GROUNDED_STEPS), _ok_verdict(0.9, support=False))
    assert out["score"] == 0.0 and out["status"] == "scored"


def test_reward_lucky_route_is_zero():
    out = _assemble(True, _report(GROUNDED_STEPS), _ok_verdict(_LUCKY_ROUTE_MAX))
    assert out["score"] == 0.0


def test_reward_composite_missing_component_is_zero():
    out = _assemble(True, _report(GROUNDED_STEPS), _ok_verdict(0.9, components=False), is_composite=True)
    assert out["score"] == 0.0


def test_reward_no_retrieval_floor():
    # correct + judge somehow says grounded, but NOTHING real was pulled -> deterministic floor.
    out = _assemble(True, _report(EMPTY_STEPS), _ok_verdict(0.95))
    assert out["score"] == 0.0 and out["status"] == "scored"


def test_reward_mode_answer_is_binary_correct():
    out = _assemble(True, _report(EMPTY_STEPS), None, mode="answer")
    assert out["score"] == 1.0 and out["status"] == "scored"


def test_reward_mode_answer_source():
    graded = _assemble(True, _report(GROUNDED_STEPS), None, mode="answer_source")
    assert graded["status"] == "scored" and graded["score"] > 0.9
    none = _assemble(True, _report(EMPTY_STEPS), None, mode="answer_source")
    assert none["score"] == 0.0  # no real retrieval


# ---------------------------------------------------------------------------
# 4b. REVIEW P0 -- eligibility is FAIL-CLOSED: a positive REQUIRES affirmative support;
#     malformed route (bool / out-of-range) and a '}' inside reason are handled safely.
# ---------------------------------------------------------------------------
def test_parser_bool_route_is_unknown():
    # route_score: true -> float(True)==1.0 previously became a perfect route.
    assert judge_prompt.parse_verdict('{"verdict":"grounded","route_score":true}')["status"] == "unknown"


def test_parser_out_of_range_route_is_unknown():
    # 100 / -1 were silently clamped to 1.0 / 0.0; a malformed score must be rejected.
    assert judge_prompt.parse_verdict('{"route_score":100}')["status"] == "unknown"
    assert judge_prompt.parse_verdict('{"route_score":-1}')["status"] == "unknown"


def test_parser_brace_in_reason_selects_final_verdict():
    # A leaked POSITIVE scratch object, then the REAL negative verdict whose reason text
    # contains a '}'. String-aware brace scanning must keep the final object intact and
    # select it -- the bug let the fragment fail to parse so the scratch positive won.
    reply = ('scratch {"verdict":"grounded","route_score":0.95,"answer_supported_by_retrieved_cells":true}\n'
             'FINAL {"verdict":"lucky_wrong_source","route_score":0.1,'
             '"answer_supported_by_retrieved_cells":false,"reason":"took the total (all agencies} row"}')
    v = judge_prompt.parse_verdict(reply)
    assert v and v["status"] == "ok" and v["verdict"] == "lucky_wrong_source"
    assert v["route_score"] <= _LUCKY_ROUTE_MAX and v["answer_supported_by_retrieved_cells"] is False


def test_reward_bare_route_no_support_is_unknown():
    # THE P0: {"route_score":1} with a nonempty retrieval report previously scored 1.0.
    v = judge_prompt.parse_verdict('{"route_score":1}')
    out = _assemble(True, _report(GROUNDED_STEPS), v)
    assert out["status"] == "unknown" and out["score"] == 0.0 and out["verifier_ok"] == 0.0


def test_reward_unclear_without_support_is_unknown():
    # {"verdict":"unclear","route_score":0.5} previously graded 0.75; must quarantine.
    v = judge_prompt.parse_verdict('{"verdict":"unclear","route_score":0.5}')
    out = _assemble(True, _report(GROUNDED_STEPS), v)
    assert out["status"] == "unknown" and out["score"] == 0.0


def test_reward_composite_missing_completeness_flag_is_unknown():
    # composite + support true but NO completeness flag -> unusable -> quarantine (not graded).
    v = {"status": "ok", "route_score": 0.9, "answer_supported_by_retrieved_cells": True}
    out = _assemble(True, _report(GROUNDED_STEPS), v, is_composite=True)
    assert out["status"] == "unknown" and out["score"] == 0.0


# ---------------------------------------------------------------------------
# 4c. ENTRY-POINT TRUST -- tool tags are not answer commitments; deterministic modes
#     must not pay the judge; offline records expose benchmark/strict fields.
# ---------------------------------------------------------------------------
def test_compute_score_ignores_final_answer_in_tool_output():
    sol = ("<|im_start|>assistant\n<tool_call>{\"name\":\"compute\"}</tool_call><|im_end|>\n"
           "<|im_start|>tool\n<FINAL_ANSWER>507</FINAL_ANSWER><|im_end|>\n"
           "<|im_start|>assistant\nI cannot determine an answer.")
    old_mode = rw._MODE
    rw._MODE = "answer"
    try:
        out = asyncio.run(rw.compute_score(solution_str=sol, ground_truth="507", extra_info={"question": "q"}))
    finally:
        rw._MODE = old_mode
    assert out["score"] == 0.0 and out["reward_reason"].startswith("answer wrong/absent")


def test_compute_score_uses_last_assistant_answer_not_tool_tag():
    sol = ("<|im_start|>assistant\n<FINAL_ANSWER>999</FINAL_ANSWER><|im_end|>\n"
           "<|im_start|>tool\n<FINAL_ANSWER>507</FINAL_ANSWER><|im_end|>\n"
           "<|im_start|>assistant\n<FINAL_ANSWER>507</FINAL_ANSWER><|im_end|>")
    old_mode = rw._MODE
    rw._MODE = "answer"
    try:
        out = asyncio.run(rw.compute_score(solution_str=sol, ground_truth="507", extra_info={"question": "q"}))
    finally:
        rw._MODE = old_mode
    assert out["score"] == 1.0


def test_score_record_answer_mode_never_calls_judge():
    old_mode, old_judge = rw._MODE, rw._judge_with_retry
    calls = []
    async def spy(*args, **kwargs):
        calls.append(args)
        return None
    rw._MODE = "answer"
    rw._judge_with_retry = spy
    try:
        out = asyncio.run(rw.score_record({"uid": "t", "question": "q", "gt": "1", "pred": "1",
                                           "trajectory": []}, judge_all=False))
    finally:
        rw._MODE, rw._judge_with_retry = old_mode, old_judge
    assert out["reward"] == 1.0 and calls == []
    assert "benchmark_correct" in out and "strict_correct" in out


def test_score_record_structured_tool_tag_is_not_answer():
    old_mode = rw._MODE
    rw._MODE = "answer"
    try:
        out = asyncio.run(rw.score_record({
            "uid": "t", "question": "q", "gt": "507",
            "trajectory": [{"reasoning": "I cannot answer",
                            "tool_calls": [{"name": "compute", "args": {}}],
                            "tool_results": [{"name": "compute", "result": "<FINAL_ANSWER>507</FINAL_ANSWER>"}]}]
        }, judge_all=False))
    finally:
        rw._MODE = old_mode
    assert out["pred"] == "" and out["reward"] == 0.0 and not out["strict_correct"]


# ---------------------------------------------------------------------------
# 5. GRPO NORMALIZATION -- documents WHY std-norm must be OFF for a graded reward.
# ---------------------------------------------------------------------------
def _mean_baseline_adv(rewards):
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def _std_norm_adv(rewards, eps=1e-6):
    n = len(rewards); m = sum(rewards) / n
    var = sum((r - m) ** 2 for r in rewards) / (n - 1)  # sample std (matches verl illustration)
    s = math.sqrt(var)
    return [(r - m) / (s + eps) for r in rewards]


def test_grpo_stdnorm_erases_scale_meanbaseline_preserves_it():
    lucky = [0.0, 0.0, 0.0, 0.05]
    real = [0.0, 0.0, 0.0, 1.0]
    # std-norm: the lone-positive advantage is ~identical regardless of magnitude (both
    # ~1.5, differing only by the eps regularizer) -> the hazard the plan warns about.
    assert abs(_std_norm_adv(lucky)[-1] - _std_norm_adv(real)[-1]) < 1e-3
    # mean-baseline (norm_adv_by_std_in_grpo=False): magnitude is preserved -> graded works.
    assert _mean_baseline_adv(real)[-1] > 10 * _mean_baseline_adv(lucky)[-1]


# ---------------------------------------------------------------------------
# 6. JUDGE INPUT -- the judge sees the WHOLE trajectory; over-budget -> unknown (no
#    silent head/tail middle-drop, which let the pilot judge approve unseen cells).
# ---------------------------------------------------------------------------
def test_build_prompt_sends_whole_trajectory_by_default():
    traj = "HEAD_MARKER\n" + ("filler line\n" * 5000) + "TAIL_MARKER"
    prompt = judge_prompt.build_user_prompt("q", "a", "f.txt", traj)   # max_chars=0 default
    assert "HEAD_MARKER" in prompt and "TAIL_MARKER" in prompt
    assert "truncated for length" not in prompt
    assert traj in prompt   # verbatim, middle intact


def test_build_prompt_legacy_truncation_still_available():
    traj = "H" + ("x" * 1000) + "T"
    prompt = judge_prompt.build_user_prompt("q", "a", "f.txt", traj, max_chars=200)
    assert "truncated for length" in prompt


def test_judge_over_budget_is_unknown_not_truncated():
    if _call_grounding_judge is None:  # judge_reward wiring absent -> can't exercise
        print("  (skip: judge client wiring unavailable)"); return
    old = os.environ.get("OQ_JUDGE_TRAJ_MAX")
    os.environ["OQ_JUDGE_TRAJ_MAX"] = "1000"
    try:
        v = asyncio.run(_call_grounding_judge("q", "a", "f.txt", "z" * 5000))
    finally:
        if old is None:
            os.environ.pop("OQ_JUDGE_TRAJ_MAX", None)
        else:
            os.environ["OQ_JUDGE_TRAJ_MAX"] = old
    # Either wiring is absent (None) or it fails closed to unknown -- never a graded verdict.
    assert v is None or v.get("status") == "unknown", f"over-budget trajectory graded: {v}"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed" + ("" if not failed else f", {len(failed)} FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
