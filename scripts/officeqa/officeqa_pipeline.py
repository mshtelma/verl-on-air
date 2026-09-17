#!/usr/bin/env python3
"""Source-verified hard-question generation pipeline (Phase-2), built on officeqa_source_verify.

Flow (SOURCE-JSON-FIRST):
  enumerate verified series (identities + actor-agree)  -> dedup across editions (canonical)
  -> compose questions from VERIFIED inputs             -> content-hash qids -> emit JSONL

Only source-verified, actor-agreeing series become inputs, so a question's gold is the unique
correct value under a NAMED edition (vintage). No table/page/line locators appear in the public
question (see docs/officeqa_phase2_scale_pipeline_plan.md sec 1.2); we publish concepts, the
aggregate, the period, the edition, and the rounding rule.

Families:
  quarter_share_gap  (pool_role=main)    -- 4 quarterly shares vs the annual share; dependent
                                            multi-step analysis over a component and its Total.
  year_sum           (pool_role=control) -- single-table aggregation; machinery calibration only.
(relative_growth_gap / cross_table_ratio_change need cross-edition operand assembly -- next increment.)

This is CPU-only and emits offline artifacts; it launches no jobs and trains nothing.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
from scripts.officeqa import officeqa_source_verify as V   # noqa: E402
from scripts.officeqa.hard_question_gen import MONTHS       # noqa: E402

QUARTERS = [("first", 0, 3), ("second", 3, 6), ("third", 6, 9), ("fourth", 9, 12)]


def _issue(edition: str) -> str:
    """treasury_bulletin_1949_02.json -> 'February 1949' (the bulletin's issue month)."""
    m = re.search(r"(\d{4})_(\d{2})", edition)
    return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}" if m else edition


def _qid(spec: dict) -> str:
    """Deterministic content hash of the immutable question specification (no gold/subtotals)."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- compositions
def compose_quarter_share_gap(vs, denominator_unique: bool = True) -> dict | None:
    """MAIN: by how many percentage points does the largest quarterly share of `concept` in the
    table's Total exceed its full-year share? Requires a positive component and positive quarterly
    and annual denominators; the answer (a share difference) is unique even if two quarters tie.

    MAIN wording is HINT-FREE (no table title); answer-unique only when `denominator_unique`. A
    non-unique concept keeps its title and is tagged `control` (see compose_fy_share_change)."""
    A, T = vs.vector, vs.total_vector
    sa, st = sum(A), sum(T)
    if sa <= 0 or st <= 0:
        return None
    qshares = []
    for name, lo, hi in QUARTERS:
        qt = sum(T[lo:hi])
        qa = sum(A[lo:hi])
        if qt <= 0 or qa < 0:      # ill-posed share (nonpositive denominator or net-credit component)
            return None
        qshares.append((name, qa / qt))
    ann = sa / st
    if ann > 1 or any(s > 1 for _n, s in qshares):
        return None            # not a proper subset share (component exceeds a net total)
    mname, mshare = max(qshares, key=lambda x: x[1])
    ans = round(100.0 * (mshare - ann), 2)
    spec = {"t": "quarter_share_gap", "concept": V._leaf(vs.concept), "caption": vs.caption,
            "year": vs.year, "edition": vs.edition, "unit": "percentage_points", "precision": 2}
    if denominator_unique:
        role = "main"
        q = (f"In the {_issue(vs.edition)} Treasury Bulletin, find the schedule that reports monthly "
             f'figures for "{vs.concept}" as one component of a monthly total, and for {vs.year} form '
             f"{vs.concept}'s share of that schedule's total (summed monthly {vs.concept} divided by "
             f"the summed monthly total) in each of the four calendar quarters and for the full "
             f"calendar year. By how many percentage points does the largest quarterly share exceed "
             f"the full-year share? Compute the shares from the summed monthly figures and round only "
             f"the final answer to two decimal places (report as percentage points, e.g. 3.14).")
    else:
        role = "control"
        q = (f"In the {_issue(vs.edition)} Treasury Bulletin, using only the reported monthly figures "
             f'for "{vs.concept}" and the corresponding monthly totals for {vs.caption} during {vs.year}, '
             f"form the share of the total (summed monthly {vs.concept} divided by the summed monthly "
             f"total) for each of the four calendar quarters of {vs.year} and for the full calendar year. "
             f"By how many percentage points does the largest quarterly share exceed the full-year share? "
             f"Compute the shares from the summed monthly figures and round only the final answer to two "
             f"decimal places (report as percentage points, e.g. 3.14).")
    return {"qid": _qid(spec), "template": "quarter_share_gap", "pool_role": role,
            "question": q, "answer": ans, "unit": "percentage points",
            "denominator_unique": denominator_unique,
            "golden_path": {
                "operation": "100*(max_q(sum(A,q)/sum(T,q)) - sum(A,year)/sum(T,year))",
                "edition": vs.edition, "caption": vs.caption, "concept": vs.concept, "year": vs.year,
                "component_monthly": A, "total_monthly": T,
                "quarterly_shares": [(n, round(s, 6)) for n, s in qshares],
                "annual_share": round(ann, 6), "max_quarter": mname}}


def compose_year_sum(vs) -> dict:
    """CONTROL: single-table aggregation (machinery calibration; excluded from the main pool)."""
    ans = round(sum(vs.vector), 4)
    spec = {"t": "year_sum", "concept": V._leaf(vs.concept), "caption": vs.caption,
            "year": vs.year, "edition": vs.edition, "unit": "millions_of_dollars", "precision": 0}
    q = (f"In the {_issue(vs.edition)} Treasury Bulletin, what is the total of the twelve reported "
         f'monthly figures for "{vs.concept}" in {vs.caption} during {vs.year}? '
         f"Report the sum in millions of dollars.")
    return {"qid": _qid(spec), "template": "year_sum", "pool_role": "control",
            "question": q, "answer": ans, "unit": "millions of dollars",
            "golden_path": {"operation": "sum(12 monthly values)", "edition": vs.edition,
                            "caption": vs.caption, "concept": vs.concept, "year": vs.year,
                            "component_monthly": vs.vector}}


def compose_fy_share_change(concept, caption, edition, y0, y1, v0, t0, v1, t1,
                            denominator_unique: bool = True) -> dict | None:
    """MAIN (fiscal-year): by how many percentage points did `concept`'s share of the table Total
    change between two fiscal years? Two dependent shares over reassembled annual data. Requires
    positive totals and nonnegative components.

    The MAIN wording is HINT-FREE (plan sec 1.2): it names the concept and the vintage but NOT the
    table title -- the actor must locate the schedule reporting `concept` as a component of a total.
    That is answer-unique only when `denominator_unique` (concept maps to a single schedule/total in
    the edition); otherwise the table title is load-bearing for uniqueness, so we keep it and tag the
    question `control` (a paired location-hinted calibration task, excluded from the main pool)."""
    if t0 <= 0 or t1 <= 0 or v0 < 0 or v1 < 0:
        return None
    s0, s1 = v0 / t0, v1 / t1
    if not (0.0 <= s0 <= 1.0 and 0.0 <= s1 <= 1.0):
        return None            # not a proper subset share (component exceeds/opposes a net total)
    ans = round(100.0 * (s1 - s0), 2)
    spec = {"t": "fy_share_change", "concept": V._leaf(concept), "caption": caption,
            "y0": y0, "y1": y1, "edition": edition, "unit": "percentage_points", "precision": 2}
    if denominator_unique:
        role = "main"
        q = (f"In the {_issue(edition)} Treasury Bulletin, find the schedule that reports "
             f'"{concept}" as one component of a larger total, and form {concept}\'s share of that '
             f"schedule's total (its own reported figure divided by the schedule's total). By how "
             f"many percentage points did that share change from fiscal year {y0} to fiscal year "
             f"{y1}? Report the change in percentage points, rounded to two decimal places (e.g. -1.23).")
    else:
        role = "control"     # concept appears under >1 total in this edition; title needed for uniqueness
        q = (f"In the {_issue(edition)} Treasury Bulletin, using the reported fiscal-year figures for "
             f'"{concept}" and the total for {caption}, by how many percentage points did {concept}\'s '
             f"share of that total change from fiscal year {y0} to fiscal year {y1}? "
             f"(Share = {concept} divided by the total; report the change in percentage points, rounded "
             f"to two decimal places, e.g. -1.23.)")
    return {"qid": _qid(spec), "template": "fy_share_change", "pool_role": role,
            "question": q, "answer": ans, "unit": "percentage points",
            "denominator_unique": denominator_unique,
            "golden_path": {"operation": "100*(A_y1/T_y1 - A_y0/T_y0)", "edition": edition,
                            "caption": caption, "concept": concept, "fiscal_years": [y0, y1],
                            "value_y0": v0, "total_y0": t0, "value_y1": v1, "total_y1": t1,
                            "share_y0": round(s0, 6), "share_y1": round(s1, 6)}}


# --------------------------------------------------------------------------- driver
def dedup_canonical(series: list) -> list:
    """Keep only actor-agreeing series; one canonical per (concept-leaf, caption, year) = the
    earliest edition that reports it (a named-vintage answer; later editions are revisions)."""
    by = defaultdict(list)
    for s in series:
        if s.actor_agrees:
            by[(V._leaf(s.concept), s.caption, s.year)].append(s)
    return [sorted(lst, key=lambda s: s.edition)[0] for lst in by.values()]


def _leaf_schedules(items, edition_of, leaf_of, caption_of) -> dict:
    """(edition, concept-leaf) -> set of distinct schedule captions that report it as a verified
    component. Used to decide `denominator_unique`: if a concept appears under exactly one total in
    an edition, the hint-free wording ('its share of that schedule's total') is answer-unique."""
    locs = defaultdict(set)
    for it in items:
        locs[(edition_of(it), leaf_of(it))].add(caption_of(it))
    return locs


def build_all(src_dir: str, clean_dir: str):
    alls = []
    for jp in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        alls += V.enumerate_verified_series(jp, clean_dir)
    canon = dedup_canonical(alls)
    locs = _leaf_schedules([vs for vs in alls if vs.actor_agrees],
                           lambda v: v.edition, lambda v: V._leaf(v.concept), lambda v: v.caption)
    mains, seen = [], set()
    for vs in canon:
        du = len(locs[(vs.edition, V._leaf(vs.concept))]) <= 1
        q = compose_quarter_share_gap(vs, denominator_unique=du)
        if q and q["qid"] not in seen:
            mains.append(q); seen.add(q["qid"])
    controls = []
    for vs in canon:
        q = compose_year_sum(vs)
        if q["qid"] not in seen:
            controls.append(q); seen.add(q["qid"])
    return alls, canon, mains, controls


def build_annual(src_dir: str, clean_dir: str):
    """1970s+ path: reassemble annual tables, build fy_share_change MAIN questions from actor-agreeing
    verified fiscal-year cells, dedup across editions (earliest edition per (concept,caption,y0,y1))."""
    groups = defaultdict(dict)     # (edition, caption, concept-leaf) -> {fiscal_year: AnnualCell}
    n_cells = n_agree = 0
    for jp in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        for c in V.enumerate_annual_series(jp, clean_dir):
            n_cells += 1
            if c.actor_agrees:
                n_agree += 1
                groups[(c.edition, c.caption, V._leaf(c.concept))][c.fiscal_year] = c
    locs = defaultdict(set)        # (edition, concept-leaf) -> distinct schedule captions
    for (ed, cap, cleaf) in groups:
        locs[(ed, cleaf)].add(cap)
    canon = {}                     # (concept-leaf, caption, y0, y1) -> question (earliest edition wins)
    for (ed, cap, cleaf), fy in groups.items():
        du = len(locs[(ed, cleaf)]) <= 1        # concept maps to a single total in this edition
        yrs = sorted(fy)
        pairs = set(zip(yrs, yrs[1:]))          # consecutive fiscal years
        if len(yrs) >= 2:
            pairs.add((yrs[0], yrs[-1]))        # plus first-vs-last
        for y0, y1 in pairs:
            c0, c1 = fy[y0], fy[y1]
            q = compose_fy_share_change(c0.concept, cap, ed, y0, y1, c0.value, c0.total,
                                        c1.value, c1.total, denominator_unique=du)
            if not q:
                continue
            k = (cleaf, cap, y0, y1)
            if k not in canon or ed < canon[k]["golden_path"]["edition"]:
                canon[k] = q
    return (n_cells, n_agree), list(canon.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/oqa_src")
    ap.add_argument("--clean", default="/tmp/oqa_corpus/unzipped")
    ap.add_argument("--out", default="/tmp/oqa_pipeline/questions.jsonl")
    a = ap.parse_args()
    alls, canon, m_month, controls = build_all(a.src, a.clean)
    (n_cells, n_agree), m_annual = build_annual(a.src, a.clean)
    n_ed = len(glob.glob(os.path.join(a.src, "*.json")))
    print(f"editions={n_ed}")
    print(f"  MONTHLY: enumerated={len(alls)} agreeing={sum(s.actor_agrees for s in alls)} "
          f"canonical={len(canon)} -> MAIN quarter_share_gap={len(m_month)} CONTROL year_sum={len(controls)}")
    print(f"  ANNUAL : cells={n_cells} agreeing={n_agree} -> MAIN fy_share_change={len(m_annual)}")
    allq = m_month + m_annual + controls
    HINT = re.compile(r"\bTable\s+[A-Za-z0-9]|\bcolumn\b|\bpage\b|line\s+\d|FFO|FF0|header_line", re.I)
    dropped = [q for q in allq if HINT.search(q["question"])]
    allq = [q for q in allq if not HINT.search(q["question"])]
    mains = [q for q in allq if q["pool_role"] == "main"]
    controls = [q for q in allq if q["pool_role"] == "control"]
    if dropped:
        print(f"  public-text hint audit dropped {len(dropped)} question(s)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for q in allq:
            fh.write(json.dumps(q) + "\n")
    from collections import Counter
    print(f"TOTAL: {len(allq)} questions ({len(mains)} main, {len(controls)} control) -> {a.out}")
    print(f"  main by template: {dict(Counter(q['template'] for q in mains))}")
    print("\nsample fiscal-year MAIN questions:")
    for q in m_annual[:4]:
        print(f"  [{q['qid']}] ans={q['answer']:>7} pp | {q['question'][:140]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
