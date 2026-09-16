#!/usr/bin/env python3
"""Grounding-AUDIT judge prompt for OfficeQA trajectories (the semantic half of the
process reward; the deterministic half is ``grounding.py``).

Design decision (with the user): the judge does NOT re-run tools and does NOT
re-derive the answer -- full tool-repl following the same extraction path is
overkill. Instead we hand the judge the ANSWER KEY (correct answer + the source
file(s), and for synth the exact table/row/period the atoms came from) and ask it
to AUDIT the trajectory: did the agent target the right CONCEPT/CATEGORY and open
the right TABLE(S) -- and, for a composite, ALL the components -- or did it reach
the number by a wrong/lucky route? Auditing against a key is far cheaper and more
reliable than open-ended "does this look good" grading, and it is exactly what
catches the OfficeQA hazard: a wrong path (TOTAL vs ARMY, a neighbouring period)
that lands on a number close to / equal to the gold.

The judge is the self-hosted GLM-5.3 (see officeqa_grounded_reward.py for the
serve/rendezvous wiring). Output is strict JSON so the reward can consume it and
every field is loggable for the human validation pass.
"""

from __future__ import annotations

import json
import math
import re

JUDGE_SYSTEM = (
    "You are a strict AUDITOR of a research agent's RETRIEVAL PROCESS over U.S. "
    "Treasury Bulletins (monthly financial tables, 1939 onward). You are given: the "
    "QUESTION, the correct REFERENCE ANSWER, the SOURCE location(s) that answer comes "
    "from (the bulletin file(s); when provided, the exact table / row / period and how "
    "the answer is composed), and the agent's full TRAJECTORY (its tool calls, the "
    "queries/patterns it searched for, the tool OUTPUTS it saw, and its final answer).\n\n"
    "Your job is NOT to re-derive the answer -- assume the reference answer is correct. "
    "It is to judge whether the agent reached ITS answer by GENUINELY GROUNDING in the "
    "correct data, or by a WRONG / LUCKY route. This is critical: in this corpus a wrong "
    "path very often lands on a number close to or equal to the correct one -- e.g. "
    "reading TOTAL national defense (or total war expenditures) when the question asks "
    "for the ARMY component, taking a neighbouring year/period, or a different but "
    "similarly-labelled row. A rollout like that must NOT be credited even though its "
    "number matches.\n\n"
    "Assess, using the retrieved tool outputs as evidence:\n"
    "1. RIGHT CATEGORY -- did it target the exact concept/line item the question asks "
    "for, not a sibling total or a look-alike row?\n"
    "2. RIGHT SOURCE + COMPLETENESS -- did it open/inspect source table(s) that "
    "legitimately contain the answer? The listed correct source file(s) are ONE valid "
    "source, not the only one: Treasury Bulletins republish historical series, so a "
    "DIFFERENT bulletin whose table shows the SAME concept, period, and value is an "
    "EQUALLY VALID alternative source -- do NOT demote the route for using it, and if "
    "the supporting cell is visible in a tool output, treat it exactly like the listed "
    "file. What is NOT acceptable: a neighbouring PERIOD, a sibling TOTAL / look-alike "
    "ROW, or -- for a multi-part question (a sum, difference, ratio, %-change, or a "
    "span of months) -- ANY required component (each period, each series) not "
    "retrieved.\n"
    "3. SUPPORTED -- is the final figure actually SUPPORTED by cells it retrieved, or is "
    "it unsupported / guessed / read off the wrong row?\n"
    "4. SANITY -- was the trajectory a set of targeted, sensible lookups, or flailing?\n\n"
    "CRITICAL EVIDENCE RULE (verbatim quotes): every positive MUST be backed by "
    "\"supporting_quotes\": an array of the EXACT lines, copied character-for-character, "
    "from the tool OUTPUTS in the trajectory, containing the supporting cell(s) -- for a "
    "composite, one line per required component. Do NOT paraphrase, do NOT invent line "
    "numbers, do NOT quote the agent's reasoning prose, do NOT reconstruct a line from "
    "memory: a deterministic checker verifies your quotes appear VERBATIM in the tool "
    "outputs, and anything it cannot find is treated as UNSUPPORTED. If you cannot "
    "quote the exact supporting line(s), set answer_supported_by_retrieved_cells to "
    "false and leave supporting_quotes empty.\n\n"
    "ANACHRONISM RULE: a bulletin can only contain actuals for a period that had "
    "ENDED (or begun, for in-year progress tables) when it was published. A source file "
    "whose date is EARLIER than the asked period (e.g. treasury_bulletin_2010_01.txt "
    "cited for June-2011 data) is IMPOSSIBLE, not an 'alternative source' -- reject it. "
    "An alternative file is valid ONLY when its date is consistent with the asked period.\n\n"
    "PERIOD-LABEL RULE: when you quote a table row, CHECK the row's own period label "
    "(its year/month cell) against the asked period. If the row you quote is labelled "
    "1935 and the question asks 1934, that is lucky_wrong_source -- never describe a "
    "row by the asked period when its own label says otherwise.\n\n"
    "Respond with ONLY a JSON object and nothing else, of exactly this shape:\n"
    '{"verdict": "grounded" | "lucky_wrong_source" | "wrong_but_reasonable" | "unclear", '
    '"route_score": <float 0..1>, '
    '"targeted_right_category": true|false, '
    '"retrieved_all_components": true|false, '
    '"answer_supported_by_retrieved_cells": true|false, '
    '"supporting_quotes": ["<verbatim line from a tool output>", ...], '
    '"reason": "<one or two sentences citing exactly what it opened / omitted>"}\n\n'
    "route_score guidance: 1.0 = clearly opened the right table(s) and row(s) (all "
    "components for a composite) and the answer is supported by them -- a VALID "
    "ALTERNATIVE FILE with the same concept+period+value scores exactly the same as "
    "the listed file; 0.6-0.8 = right "
    "area with a minor gap; 0.3-0.5 = partially right or one required component missing; "
    "0.0-0.2 = wrong table/category/period or the answer is unsupported by anything "
    "retrieved (a lucky match). Use verdict \"lucky_wrong_source\" specifically when the "
    "final number matches the reference but the retrieval was of the WRONG series/total/"
    "period. If you genuinely cannot tell from the trajectory, use \"unclear\" with "
    "route_score 0.5.\n\n"
    "STRICT OUTPUT RULES: fill EVERY field. route_score must be a finite number in "
    "[0,1] -- never omit it. The three flags must be literal JSON true/false (not the "
    "strings \"true\"/\"false\"). Set answer_supported_by_retrieved_cells to true ONLY "
    "if the exact supporting cell/value actually APPEARS in a tool OUTPUT shown in the "
    "trajectory AND you copied the exact line(s) into supporting_quotes; if the "
    "supporting cell is not visible in the transcript (e.g. it would be in a part you "
    "were not shown), set it to false -- do NOT infer support from the reference answer "
    "or from the agent's own prose. Cite the supporting line in \"reason\". Keep "
    "verdict and route_score consistent (grounded => high route and support true; "
    "lucky_wrong_source => low route)."
)


def build_user_prompt(
    question: str,
    reference_answer: str,
    source_files: str,
    trajectory: str,
    *,
    atom_spec: str = "",
    max_chars: int = 0,
) -> str:
    """Assemble the judge's user message.

    trajectory: the agent's full working as text (reasoning + tool calls + tool
    OUTPUTS + final answer). By default (``max_chars=0``) the WHOLE trajectory is sent
    -- the judge must grade the evidence the agent actually saw, not a head/tail view.
    The pilot's head+tail truncation dropped the middle of 47/51 correct traces, so the
    judge approved cells it could not inspect; the caller now enforces a fail-closed
    ceiling instead (over-length -> ``unknown``/quarantine, never a silent middle-drop).
    ``max_chars>0`` re-enables head/tail truncation only for explicit legacy use.
    ``atom_spec`` is the optional fine-grained key for synthesised questions.
    """
    traj = trajectory or ""
    if max_chars and len(traj) > max_chars:
        head = max_chars // 3
        traj = traj[:head] + "\n...(trajectory truncated for length)...\n" + traj[-(max_chars - head):]
    src_block = source_files.strip() if source_files else "(not provided)"
    key = f"\n[How the answer is composed]\n{atom_spec}\n" if atom_spec else ""
    return (
        f"[Question]\n{question}\n\n"
        f"[Reference answer -- assume correct]\n{reference_answer}\n\n"
        f"[Correct source file(s)]\n{src_block}\n"
        f"{key}\n"
        f"[Agent trajectory -- tool calls, arguments, tool outputs, final answer]\n{traj}\n\n"
        "Audit the trajectory now. Return ONLY the JSON object."
    )


# verdict -> a coarse ordering, so callers can reason about it without string checks.
_VERDICTS = {"grounded", "lucky_wrong_source", "wrong_but_reasonable", "unclear"}
# A route at/below this is a lucky/unsupported match (mirrors the reward's LUCKY_ROUTE_MAX);
# used only for CONTRADICTION detection here, not for scoring.
_LUCKY_ROUTE_MAX = 0.20


def _coerce_bool(x) -> bool | None:
    """Strict boolean coercion. A real bool passes through; the STRINGS "true"/"false"
    (and yes/no/1/0) map correctly; everything else -> None (missing/unusable).
    This kills the confirmed ``bool("false") == True`` bug -- a stringy "false" from the
    judge previously flipped a negative support flag into a positive one."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if x in (0, 1):
            return bool(x)
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def _finite_float(x) -> float | None:
    """float(x) only if it is a real finite number; NaN / inf / garbage / BOOL -> None.
    Kills two confirmed bugs: ``max(0, min(1, NaN)) == 1.0`` (a NaN route became full
    confidence) and ``float(True) == 1.0`` (a JSON boolean ``route_score: true`` silently
    became a perfect 1.0 route). A JSON bool is NOT a numeric score, so reject it here."""
    if isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_objects(text: str) -> list[str]:
    """Every balanced-brace ``{...}`` substring, in order (nested- and multiline-safe),
    IGNORING braces that appear inside JSON string literals. String-awareness matters:
    a valid final verdict whose ``reason`` text contains a ``}`` (e.g. "took the total
    (all agencies} row") must NOT be split -- the confirmed bug let the truncated fragment
    fail to parse so an earlier POSITIVE scratchpad object was selected instead. Also robust
    to a GLM-5.3 reply that carries a leaked <think> scratchpad before the final JSON."""
    out, depth, start = [], 0, -1
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:                       # inside a "..." literal: braces/quotes are data
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
                start = -1
    return out


def parse_verdict(content: str) -> dict | None:
    """Extract the audit verdict from the judge reply, tolerant of code fences / stray
    prose / a leaked <think> preamble.

    Returns one of:
      * None                          -- no verdict-shaped JSON object at all.
      * {"status": "unknown", ...}    -- a verdict object was found but is UNUSABLE
                                         (missing route_score, non-finite score, string
                                         booleans aside, or self-contradictory). The
                                         caller must treat this as a verifier failure
                                         (retry / quarantine) -- NEVER as a positive.
      * {"status": "ok", "route_score": float, "verdict": str,
         "targeted_right_category"/"retrieved_all_components"/
         "answer_supported_by_retrieved_cells": bool|None, "reason": str}

    Design (see docs/officeqa_rl_plan.md Section 6.4.D): the parser NEVER invents
    certainty. A missing route is not defaulted to 1.0; a missing/garbage boolean is
    ``None`` (not ``True``); NaN is rejected; a contradictory verdict (e.g.
    ``lucky_wrong_source`` with route 1.0, or ``grounded`` with support ``false``)
    is rejected as unknown rather than silently scored."""
    if not content:
        return None
    text = content.strip().replace("```json", "```")
    # A GLM-5.3 reply may carry a leaked <think> scratchpad + the final JSON. Scan ALL
    # balanced-brace objects and take the last verdict-shaped one (a scratch object first).
    for block in reversed(_json_objects(text)):
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        has_rs = "route_score" in obj
        verdict = str(obj.get("verdict", "")).strip().lower()
        if not has_rs and verdict not in _VERDICTS:
            continue  # not verdict-shaped; keep scanning

        reason = str(obj.get("reason", ""))[:500]
        # route_score is REQUIRED and must be finite. Present-but-unusable (NaN/inf/
        # garbage) is a malformed verdict -> unknown. Absent -> we will not grade -> unknown.
        route = _finite_float(obj.get("route_score")) if has_rs else None
        if has_rs and route is None:
            return {"status": "unknown", "reason": f"non-finite/boolean route_score; {reason}",
                    "raw_verdict": verdict}
        if route is None:
            return {"status": "unknown", "reason": f"verdict without a route_score; {reason}",
                    "raw_verdict": verdict}
        if not (0.0 <= route <= 1.0):        # out-of-range is malformed -- DO NOT silently clamp
            return {"status": "unknown", "reason": f"route_score {route} outside [0,1]; {reason}",
                    "raw_verdict": verdict}

        support = _coerce_bool(obj.get("answer_supported_by_retrieved_cells"))
        components = _coerce_bool(obj.get("retrieved_all_components"))
        category = _coerce_bool(obj.get("targeted_right_category"))
        if verdict not in _VERDICTS:
            verdict = "grounded" if route >= 0.8 else "lucky_wrong_source" if route <= _LUCKY_ROUTE_MAX else "unclear"

        # A lucky_wrong_source verdict is the CONSERVATIVE (safe) call. Believe it and cap
        # the route to the lucky threshold so the reward floors it as a real negative --
        # rather than quarantining the whole group on a route/verdict mismatch. (R-gate
        # RG06: the judge picked the right verdict but a generous route 0.30; that should
        # score 0 as lucky, not become an 'unknown' that drops the sibling group.)
        if verdict == "lucky_wrong_source":
            route = min(route, _LUCKY_ROUTE_MAX)

        # CONTRADICTIONS in the DANGEROUS direction (a positive would be wrong) -> unknown;
        # do not guess which half the judge meant.
        contradiction = None
        if verdict == "grounded" and support is False:
            contradiction = "verdict grounded but answer_supported=false"
        elif support is False and route >= 0.6:
            contradiction = f"answer_supported=false but route {route:.2f}"
        if contradiction:
            return {"status": "unknown", "reason": f"contradictory verdict: {contradiction}; {reason}",
                    "raw_verdict": verdict, "raw_route": route}

        # supporting_quotes: verbatim evidence lines the deterministic layer verifies
        # against the tool outputs (confabulation detector). Normalize to a list of
        # non-empty strings; absent/garbled -> [] (a positive with no verifiable quote
        # is demoted downstream, so a lazy judge cannot skip the evidence rule).
        raw_quotes = obj.get("supporting_quotes")
        if isinstance(raw_quotes, str):
            raw_quotes = [raw_quotes]
        quotes = []
        if isinstance(raw_quotes, list):
            for q in raw_quotes:
                if isinstance(q, str) and q.strip():
                    quotes.append(q.strip()[:500])
        return {
            "status": "ok",
            "route_score": route,
            "verdict": verdict,
            "targeted_right_category": category,
            "retrieved_all_components": components,
            "answer_supported_by_retrieved_cells": support,
            "supporting_quotes": quotes,
            "reason": reason,
        }
    return None


if __name__ == "__main__":
    demo = build_user_prompt(
        question="What were total U.S. national defense expenditures in calendar 1940 (millions)?",
        reference_answer="2,602",
        source_files="treasury_bulletin_1941_01.txt",
        trajectory="<function=grep_documents><parameter=pattern>National defense</parameter>"
                   "<parameter=file_name>treasury_bulletin_1941_01.txt</parameter></function>\n"
                   "treasury_bulletin_1941_01.txt:14: National defense ... 2,602\n<FINAL_ANSWER>2,602</FINAL_ANSWER>",
    )
    print(JUDGE_SYSTEM[:200], "...\n")
    print(demo)
    print("\n-- parse_verdict fail-closed checks --")
    for raw in [
        '{"verdict":"grounded","route_score":0.95,"answer_supported_by_retrieved_cells":true,"reason":"opened 1941_01 defense row"}',
        '{"verdict":"grounded"}',                                   # no route -> unknown
        '{"route_score": NaN}',                                     # NaN -> unknown
        '{"verdict":"lucky_wrong_source","route_score":1.0}',       # contradiction -> unknown
        '{"answer_supported_by_retrieved_cells":"false","route_score":0.9}',  # str-bool + contradiction
    ]:
        v = parse_verdict(raw)
        print(f"  {raw[:64]:64s} -> status={v.get('status') if v else None} "
              f"route={v.get('route_score') if v else None} support={v.get('answer_supported_by_retrieved_cells') if v else None}")
