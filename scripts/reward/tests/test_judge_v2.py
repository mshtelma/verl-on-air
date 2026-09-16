#!/usr/bin/env python3
"""Gate-R v2 contract: deterministic verification of the judge's evidence.

Air/82 (expanded suite, real GLM-5.3) proved the bare judge CONFABULATES support:
121/220 planted negatives credited (55%), including quotes of lines that were never
in the transcript and anachronistic bulletins waved through as "alternative sources".
The v2 layer makes the judge's evidence mechanical: verbatim quote verification,
anachronism arithmetic, and period-label matching -- all deterministic.

Run standalone:
    PYTHONPATH=scripts:scripts/reward python3 scripts/reward/tests/test_judge_v2.py
"""

from __future__ import annotations

import asyncio
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, "..")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import judge_prompt                                           # noqa: E402
import officeqa_grounded_reward as rw                          # noqa: E402

Q1945 = "What were total expenditures in FY 1945?"
EV = ("treasury_bulletin_1945_09.txt:266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |\n"
      "267: | 1946 | 9,999 | 1 | 2 | 3 | 4 |")
GOOD_QUOTE = "| 1945 | 2,681 | 341 | 480 | 507 | 154 |"


def _verdict(route=0.95, quotes=None, support=True, components=True, verdict="grounded"):
    v = {"status": "ok", "route_score": route, "verdict": verdict,
         "answer_supported_by_retrieved_cells": support,
         "retrieved_all_components": components}
    if quotes is not None:
        v["supporting_quotes"] = quotes
    return v


def _report_with_pull():
    steps = [{"tool_calls": [{"name": "grep_documents", "args": {"pattern": "x", "file_name": "treasury_bulletin_1945_09.txt"}}],
              "tool_results": [{"name": "grep_documents",
                                "result": "treasury_bulletin_1945_09.txt:266: | 1945 | 2,681 | 507 |"}]}]
    import grounding
    return grounding.grounding_report(steps, {"source_files": "treasury_bulletin_1945_09.txt"})


REPORT = _report_with_pull()


def test_parse_verdict_extracts_quotes():
    raw = ('{"verdict":"grounded","route_score":0.95,"answer_supported_by_retrieved_cells":true,'
           '"supporting_quotes":["| 1945 | 507 |", "  "],"reason":"ok"}')
    v = judge_prompt.parse_verdict(raw)
    assert v["status"] == "ok" and v["supporting_quotes"] == ["| 1945 | 507 |"], v
    v2 = judge_prompt.parse_verdict('{"verdict":"grounded","route_score":0.9,'
                                    '"answer_supported_by_retrieved_cells":true}')
    assert v2["supporting_quotes"] == []                       # absent -> empty, demoted later


def test_quote_verification_verbatim():
    matched = rw._verify_supporting_quotes([GOOD_QUOTE], EV)
    assert matched and "507" in matched[0]
    assert rw._verify_supporting_quotes(["| 1999 | 0 | 0 |"], EV) == []
    assert rw._verify_supporting_quotes([GOOD_QUOTE], "") == []


def test_v2_accepts_verified_grounded():
    out = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE]),
                       question=Q1945, evidence_text=EV, gt="507")
    assert out["status"] == "scored" and out["score"] > 0.3, out


def test_v2_demotes_quote_without_gold_value():
    # THE dominant air/82-v2 false-accept: judge quoted a surviving search-snippet
    # line that contained NO gold value at all.
    ev = "[0] treasury_bulletin_1945_09.txt (1945-09)  score=8.1\n    National defense expenditures by year"
    out = rw._assemble(True, REPORT, _verdict(quotes=["National defense expenditures by year"]),
                       question=Q1945, evidence_text=ev, gt="507")
    assert out["score"] == 0.0 and "do not contain the committed value" in out["reward_reason"], out


def test_v2_demotes_missing_quotes():
    out = rw._assemble(True, REPORT, _verdict(quotes=None), question=Q1945, evidence_text=EV)
    assert out["score"] == 0.0 and "NO verbatim quote" in out["reward_reason"], out


def test_v2_demotes_confabulated_quotes():
    out = rw._assemble(True, REPORT, _verdict(quotes=["| 1999 | 0 | confabulated |"]),
                       question=Q1945, evidence_text=EV)
    assert out["score"] == 0.0 and "confabulated evidence" in out["reward_reason"], out


def test_v2_demotes_anachronistic_source():
    ev = "treasury_bulletin_1943_01.txt:266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |"
    out = rw._assemble(True, REPORT, _verdict(quotes=["| 1945 | 2,681 | 341 | 480 | 507 | 154 |"]),
                       question=Q1945, evidence_text=ev)
    assert out["score"] == 0.0 and "anachronistic" in out["reward_reason"], out
    # control: same-year-or-later bulletin is fine
    out_ok = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE]), question=Q1945,
                          evidence_text=EV)
    assert out_ok["score"] > 0.3


def test_v2_demotes_period_label_mismatch():
    ev = "treasury_bulletin_1945_09.txt:267: | 1946 | 2,681 | 341 | 480 | 507 | 154 |"
    out = rw._assemble(True, REPORT, _verdict(quotes=["| 1946 | 2,681 | 341 | 480 | 507 | 154 |"]),
                       question=Q1945, evidence_text=ev)
    assert out["score"] == 0.0 and "period label" in out["reward_reason"], out


def test_v2_composite_needs_two_distinct_lines():
    v = _verdict(quotes=[GOOD_QUOTE], components=True)
    out = rw._assemble(True, REPORT, v, is_composite=True, question=Q1945, evidence_text=EV)
    assert out["score"] == 0.0 and "two distinct" in out["reward_reason"], out
    ev2 = EV + "\ntreasury_bulletin_1945_09.txt:301: | Navy Department | 30,047 | 12.5 |"
    out2 = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE, "| Navy Department | 30,047 | 12.5 |"]),
                        is_composite=True, question=Q1945, evidence_text=ev2, gt="80,882")
    assert out2["score"] > 0.3, out2
    # component quotes must carry real cells, not prose
    out3 = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE, "the navy figure was found"]),
                        is_composite=True, question=Q1945,
                        evidence_text=EV + "\nthe navy figure was found", gt="80,882")
    assert out3["score"] == 0.0 and "component cells" in out3["reward_reason"], out3


def test_v2_legacy_path_unchanged_without_evidence():
    out = rw._assemble(True, REPORT, _verdict(quotes=None))     # no evidence_text
    assert out["score"] > 0.3, out                              # legacy trust path intact


def test_v2_anachronism_ignores_unrelated_listing_files():
    # Regression for the v2 true-accept collapse: a search LISTING mentioning older
    # bulletins far above the quote must NOT mis-attribute the quote's source.
    ev = ("[0] treasury_bulletin_1939_01.txt (1939-01)  score=7.7\n"
          "[1] treasury_bulletin_1945_09.txt (1945-09)  score=8.2\n"
          "treasury_bulletin_1945_09.txt:266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |")
    out = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE]),
                       question=Q1945, evidence_text=ev, gt="507")
    assert out["score"] > 0.3, out
    # but a quote line under an anachronistic chunk header still demotes
    ev2 = ("treasury_bulletin_1943_01.txt lines 260-359 of 5049:\n"
           "266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |")
    out2 = rw._assemble(True, REPORT, _verdict(quotes=[GOOD_QUOTE]),
                        question=Q1945, evidence_text=ev2, gt="507")
    assert out2["score"] == 0.0 and "anachronistic" in out2["reward_reason"], out2


def test_score_record_threads_tool_outputs_as_evidence():
    rec = {"uid": "X1", "question": Q1945, "gt": "507", "pred": "507",
           "source_files": "treasury_bulletin_1945_09.txt",
           "trajectory": [
               {"turn": 0, "reasoning": "grep",
                "tool_calls": [{"name": "grep_documents", "args": {"pattern": "507", "file_name": "treasury_bulletin_1945_09.txt"}}],
                "tool_results": [{"name": "grep_documents",
                                  "result": "treasury_bulletin_1945_09.txt:266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |"}]},
               {"turn": 1, "reasoning": "<FINAL_ANSWER>507</FINAL_ANSWER>",
                "tool_calls": [], "tool_results": []}]}
    real = rw._judge_with_retry
    try:
        async def fake_judge(*a, **k):
            return _verdict(quotes=[GOOD_QUOTE])
        rw._judge_with_retry = fake_judge
        out = asyncio.run(rw.score_record(rec, judge_all=True))
        assert out["reward"] > 0.3, out

        async def fake_confab(*a, **k):
            return _verdict(quotes=["| 1999 | fabricated |"])
        rw._judge_with_retry = fake_confab
        out2 = asyncio.run(rw.score_record(rec, judge_all=True))
        assert out2["reward"] == 0.0 and "confabulated" in out2["reward_reason"], out2
    finally:
        rw._judge_with_retry = real


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
