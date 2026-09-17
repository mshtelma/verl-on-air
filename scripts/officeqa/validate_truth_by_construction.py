#!/usr/bin/env python3
"""Targeted validator for the 'truth by construction' claim of the hard-question generator.

This is a READ-ONLY diagnostic (no GPU, no image, no write to the pipeline). It answers the
question the Phase-2 plan (docs/officeqa_phase2_scale_pipeline_plan.md, section 1) raises:
*is a generated answer actually the unique, correct value, or only arithmetically correct
conditional on cells the generator happened to pick?*

It reuses the generator's PARSER (extract_tables / parse_number / month detection) so the
extraction checks see exactly what the generator sees, but it recomputes every template's
answer with an INDEPENDENT implementation, and it relaxes only the 12-month completeness gate
so the conflict audit can see the partial reprints the generator deliberately ignores.

Checks (grouped):
  CORPUS-FREE (from the committed golden paths alone):
    A. cell-parse consistency     -- raw string re-parses to the recorded value; subtotal ok
    B. independent arithmetic      -- recorded answer == operation applied to recorded cells
    C. in-table Cal.-yr. crosscheck-- year_sum vs the bulletin's own published annual total
    D. two-year operand coherence  -- both operands drawn from the same source edition/file
    E. degeneracy / semantics      -- all-negative "flows", tiny/zero denominators, sign flips
    P. parse_number adversarial    -- independent expectations for hostile cell strings
  CORPUS-DEPENDENT (need the unzipped .txt corpus; --corpus DIR):
    F. cell readback               -- each cited (file,line,month) really holds the raw/value
    G. partial-reprint conflict    -- does ANY bulletin (incl. partial-year) report a different
                                      value for a month this question uses? (the 313/318 case)
    H. answer ambiguity            -- how many distinct (file,table,column) sites match the
                                      public (category, year); do they yield different sums?

Usage:
  python3 scripts/officeqa/validate_truth_by_construction.py \
      --jsonl officeqa_pilot_records/hard_synth_pilot_150.jsonl \
      [--corpus /tmp/oqa_corpus/unzipped] [--max-conflicts 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
from scripts.officeqa import hard_question_gen as g  # noqa: E402

EPS = 1e-3  # a genuine value conflict must exceed this (313 vs 318 = 5.0, far above)


# ------------------------------------------------------------------ helpers
def _vals(inp: dict) -> list[float]:
    return [c["value"] for c in inp["cited_cells"]]


def _by_role(rec: dict) -> dict:
    return {inp["role"]: inp for inp in rec["golden_path"]["inputs"]}


def _months_of(inp: dict) -> dict:
    """month-name -> value for one input's cited cells."""
    return {c["month"]: c["value"] for c in inp["cited_cells"]}


# ------------------------------------------------------------------ CORPUS-FREE
def check_cell_parse(recs) -> list[str]:
    """A: every cited raw string re-parses to its recorded value; subtotal_sum == sum(values)."""
    bad = []
    for r in recs:
        for inp in r["golden_path"]["inputs"]:
            s = 0.0
            for c in inp["cited_cells"]:
                p = g.parse_number(c["raw"])
                if p is None or abs(p - c["value"]) > EPS:
                    bad.append(f'{r["template"]} "{r["golden_path"]["category"]}" {inp.get("year")}: '
                               f'raw={c["raw"]!r} parses {p} != recorded {c["value"]}')
                s += c["value"]
            if "subtotal_sum" in inp and abs(round(s, 4) - inp["subtotal_sum"]) > EPS:
                bad.append(f'{r["template"]} {inp.get("year")}: sum(cells)={round(s,4)} '
                           f'!= subtotal_sum {inp["subtotal_sum"]}')
    return bad


def _independent_answer(r: dict):
    """Recompute the answer from cited cells with an implementation independent of the generator."""
    t = r["template"]
    role = _by_role(r)
    if t == "year_sum":
        return round(sum(_vals(role["year"])), 4)
    if t == "year_mean":
        v = _vals(role["year"]); return round(sum(v) / 12.0, 2)
    if t == "halfyear_mean":
        m = _months_of(role["year"]); sub = [m[k] for k in g.MONTHS[6:12] if k in m]
        return round(sum(sub) / len(sub), 2) if sub else None
    if t == "year_geomean":
        v = _vals(role["year"])
        if any(x <= 0 for x in v):
            return None
        return round(math.exp(sum(math.log(x) for x in v) / 12.0), 2)
    if t == "two_year_diff":
        return round(abs(sum(_vals(role["year_b"])) - sum(_vals(role["year_a"]))), 4)
    if t == "two_year_pct_change":
        sa = sum(_vals(role["year_a"]))
        if sa == 0:
            return None
        return round(abs((sum(_vals(role["year_b"])) - sa) / sa * 100.0), 2)
    return None


def check_arithmetic(recs) -> list[str]:
    """B: recorded answer equals the operation independently applied to the recorded cells."""
    bad = []
    for r in recs:
        exp = _independent_answer(r)
        if exp is None:
            bad.append(f'{r["template"]} "{r["golden_path"]["category"]}": could not recompute')
        elif abs(exp - r["answer"]) > EPS:
            bad.append(f'{r["template"]} "{r["golden_path"]["category"]}" {r["golden_path"]["years"]}: '
                       f'recorded {r["answer"]} != independent {exp}')
    return bad


def check_calyr(recs):
    """C: year_sum vs in-table published Cal.-yr. total (independent number in the bulletin)."""
    present = [r for r in recs if r["template"] == "year_sum" and r.get("xcheck_calyr_total") is not None]
    disagree = [r for r in present if (r.get("xcheck_abs_diff") or 0) > EPS]
    return present, disagree


def check_two_year_coherence(recs):
    """D: two-year templates should draw both operands from one edition (unit/vintage coherence)."""
    tv = [r for r in recs if r["template"] in ("two_year_diff", "two_year_pct_change")]
    cross = []
    for r in tv:
        files = {inp["file"] for inp in r["golden_path"]["inputs"]}
        if len(files) > 1:
            cross.append((r["golden_path"]["category"], r["golden_path"]["years"], sorted(files)))
    return tv, cross


def check_degeneracy(recs):
    """E: semantic red flags -- summing all-negative 'flows', explosive/zero % denominators, sign flips."""
    flags = defaultdict(list)
    for r in recs:
        t = r["template"]
        role = _by_role(r)
        if t in ("year_sum", "year_mean", "halfyear_mean"):
            v = _vals(role["year"])
            if all(x < 0 for x in v):
                flags["all_negative_series"].append((t, r["golden_path"]["category"], r["golden_path"]["years"], round(sum(v), 2)))
        if t == "two_year_pct_change":
            sa, sb = sum(_vals(role["year_a"])), sum(_vals(role["year_b"]))
            if sa != 0 and (sa < 0) != (sb < 0):
                flags["sign_flip_pct"].append((r["golden_path"]["category"], r["golden_path"]["years"], round(sa, 2), round(sb, 2)))
            elif sa != 0 and abs(sa) < 0.05 * max(abs(sb), 1.0):
                flags["tiny_base_pct"].append((r["golden_path"]["category"], r["golden_path"]["years"], round(sa, 2), r["answer"]))
    return flags


# ------------------------------------------------------------------ parse_number adversarial
def check_parse_number_adversarial():
    """P: independent expectations for hostile cell strings. Each case is (input, expected,
    note); expected None means 'must reject'. Mismatches expose silent mis-parses."""
    cases = [
        ("1,902", 1902.0, "clean"),
        ("(32)", -32.0, "paren-negative"),
        ("1,580 3/", 1580.0, "footnote tail"),
        ("$2,253", 2253.0, "dollar prefix"),
        ("n.a.", None, "null spelling"),
        ("--", None, "null dash"),
        # --- adversarial: these SHOULD be rejected but the current parser may accept a prefix ---
        ("12-14", None, "range -> must not silently become 12"),
        ("1 234", None, "space-thousands -> must not become 1"),
        ("3.2%", None, "percent cell -> must not become 3.2 in a summed column"),
        ("12 months", None, "textual qty -> must not become 12"),
        ("2/ 3/", None, "pure footnote markers, no value"),
        ("1.2.3", None, "malformed -> must not become 1.2"),
    ]
    rows = []
    for s, exp, note in cases:
        got = g.parse_number(s)
        ok = (got is None and exp is None) or (got is not None and exp is not None and abs(got - exp) <= EPS)
        rows.append((s, exp, got, ok, note))
    return rows


# ------------------------------------------------------------------ CORPUS index (relaxed)
def build_relaxed_index(files):
    """Scan ALL flow tables and record EVERY (norm_cat, year, month) cell present -- WITHOUT the
    generator's 12-month completeness or cross-source consistency gates. This is what lets us see
    partial reprints. Returns:
      cell_index[(norm_cat, year, month)] -> list of (file, header_line, value, raw)
      site_index[(norm_cat, year)]        -> set of (file, header_line, col)  (matching sites)
    Uses the generator's own month-col / numeric-col detection (identical to what it would pick).
    """
    cell_index = defaultdict(list)
    site_index = defaultdict(set)
    for f in files:
        for t in g.extract_tables(f):
            if not g.is_flow_table(t):
                continue
            label_col = g._month_label_col(t)
            if label_col is None:
                continue
            cat_cols = g._numeric_category_cols(t, label_col)
            if not cat_cols:
                continue
            for c in cat_cols:
                cat = g._clean_cat(t.headers[c].strip())
                if not g.is_good_category(cat):
                    continue
                nc = g._norm_cat(cat)
                cur_year = None
                for ln, cells in t.rows:
                    if label_col >= len(cells):
                        continue
                    mn, yr = g.month_of(cells[label_col])
                    if yr:
                        cur_year = yr
                    if mn and cur_year and c < len(cells):
                        v = g.parse_number(cells[c])
                        if v is not None:
                            cell_index[(nc, cur_year, g.MONTHS[mn - 1])].append(
                                (t.file, t.header_line, v, cells[c].strip()))
                            site_index[(nc, cur_year)].add((t.file, t.header_line, c))
    return cell_index, site_index


def check_readback(recs, cell_index) -> list[str]:
    """F: each cited (category, year, month) value actually appears at a matching site in the corpus."""
    bad = []
    for r in recs:
        for inp in r["golden_path"]["inputs"]:
            nc = g._norm_cat(r["golden_path"]["category"])
            yr = inp.get("year")
            for c in inp["cited_cells"]:
                hits = cell_index.get((nc, yr, c["month"]), [])
                if not any(abs(v - c["value"]) <= EPS for _f, _h, v, _raw in hits):
                    seen = sorted({round(v, 3) for _f, _h, v, _raw in hits})
                    bad.append(f'{r["template"]} "{r["golden_path"]["category"]}" {yr} {c["month"]}: '
                               f'cited {c["value"]} not found in corpus (found {seen or "nothing"})')
    return bad


def check_conflicts(recs, cell_index, max_report=40):
    """G: for every month a question uses, does ANY bulletin report a different value? (the crux)."""
    per_q = []  # (template, category, years, n_conflict_months, [examples])
    for r in recs:
        cat = r["golden_path"]["category"]
        nc = g._norm_cat(cat)
        conflicts = []
        for inp in r["golden_path"]["inputs"]:
            yr = inp.get("year")
            for c in inp["cited_cells"]:
                hits = cell_index.get((nc, yr, c["month"]), [])
                distinct = sorted({round(v, 3) for _f, _h, v, _raw in hits})
                if len(distinct) > 1:
                    files = sorted({os.path.basename(f) for f, _h, _v, _raw in hits})
                    conflicts.append((yr, c["month"], distinct, files))
        if conflicts:
            per_q.append((r["template"], cat, r["golden_path"]["years"], len(conflicts), conflicts[:3]))
    per_q.sort(key=lambda x: -x[3])
    return per_q


def check_ambiguity(recs, site_index):
    """H: how many distinct (file,table,column) sites match the public (category, year)?"""
    multi = []
    for r in recs:
        nc = g._norm_cat(r["golden_path"]["category"])
        sites_all = set()
        for inp in r["golden_path"]["inputs"]:
            sites_all |= site_index.get((nc, inp.get("year")), set())
        n_files = len({f for f, _h, _c in sites_all})
        if len(sites_all) > 1:
            multi.append((r["template"], r["golden_path"]["category"], r["golden_path"]["years"],
                          len(sites_all), n_files))
    multi.sort(key=lambda x: -x[3])
    return multi


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="officeqa_pilot_records/hard_synth_pilot_150.jsonl")
    ap.add_argument("--corpus", default=os.environ.get("OFFICEQA_CORPUS_DIR", ""))
    ap.add_argument("--max-conflicts", type=int, default=40)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.jsonl)]
    print(f"# Truth-by-construction validation over {len(recs)} questions from {a.jsonl}")
    print(f"# templates: {dict(Counter(r['template'] for r in recs))}\n")

    print("=" * 78)
    print("CORPUS-FREE CHECKS")
    print("=" * 78)

    bad_a = check_cell_parse(recs)
    print(f"[A] cell-parse consistency: {'PASS' if not bad_a else f'{len(bad_a)} FAIL'}")
    for m in bad_a[:10]:
        print("    -", m)

    bad_b = check_arithmetic(recs)
    print(f"[B] independent arithmetic re-derivation: {'PASS' if not bad_b else f'{len(bad_b)} FAIL'}")
    for m in bad_b[:10]:
        print("    -", m)

    present, disagree = check_calyr(recs)
    print(f"[C] in-table Cal.-yr. cross-check: {len(present)}/25 year_sum have one; "
          f"{len(disagree)} disagree (>{EPS})")
    for r in disagree[:10]:
        print(f'    - "{r["golden_path"]["category"]}" {r["golden_path"]["years"]}: '
              f'sum={r["answer"]} calyr={r["xcheck_calyr_total"]} |diff|={r["xcheck_abs_diff"]}')

    tv, cross = check_two_year_coherence(recs)
    print(f"[D] two-year operand coherence: {len(tv)} two-year Qs; "
          f"{len(cross)} draw operands from >1 source file (unit/vintage-mismatch risk)")
    for cat, yrs, files in cross[:10]:
        print(f'    - "{cat}" {yrs}: {files}')

    flags = check_degeneracy(recs)
    print(f"[E] semantic degeneracy flags:")
    for k in ("all_negative_series", "sign_flip_pct", "tiny_base_pct"):
        items = flags.get(k, [])
        print(f"    {k}: {len(items)}")
        for it in items[:5]:
            print("        ", it)

    print(f"\n[P] parse_number adversarial:")
    rows = check_parse_number_adversarial()
    n_bad = sum(1 for *_x, ok, _n in [(r[0], r[1], r[2], r[3], r[4]) for r in rows] if not ok)
    for s, exp, got, ok, note in rows:
        print(f"    {'ok ' if ok else 'BAD'}  {s!r:14} -> {got!r:10} (want {exp!r})  # {note}")
    print(f"    => {n_bad} adversarial case(s) mis-parsed")

    if not a.corpus or not glob.glob(os.path.join(a.corpus, "treasury_bulletin_*.txt")):
        print("\n(no --corpus supplied or no docs found; skipping corpus-dependent checks F/G/H)")
        return 0

    files = sorted(glob.glob(os.path.join(a.corpus, "treasury_bulletin_*.txt")))
    print("\n" + "=" * 78)
    print(f"CORPUS-DEPENDENT CHECKS ({len(files)} bulletins @ {a.corpus})")
    print("=" * 78)
    cell_index, site_index = build_relaxed_index(files)
    print(f"relaxed index: {len(cell_index)} (cat,year,month) cells; "
          f"{len(site_index)} (cat,year) series (partial reprints INCLUDED)\n")

    bad_f = check_readback(recs, cell_index)
    print(f"[F] cell readback: {'PASS' if not bad_f else f'{len(bad_f)} cell(s) not found'}")
    for m in bad_f[:10]:
        print("    -", m)

    per_q = check_conflicts(recs, cell_index, a.max_conflicts)
    n_q = len(recs)
    print(f"\n[G] partial-reprint conflict audit: {len(per_q)}/{n_q} questions use >=1 month whose "
          f"value another bulletin reports DIFFERENTLY")
    for t, cat, yrs, ncf, ex in per_q[:a.max_conflicts]:
        print(f'    - [{t}] "{cat}" {yrs}: {ncf} conflicting month-cell(s)')
        for yr, mon, distinct, fls in ex:
            print(f'          {yr} {mon}: values {distinct} across {fls}')

    multi = check_ambiguity(recs, site_index)
    print(f"\n[H] answer-ambiguity: {len(multi)}/{n_q} questions match >1 (file,table,column) site "
          f"for their (category, year)")
    for t, cat, yrs, nsites, nfiles in multi[:15]:
        print(f'    - [{t}] "{cat}" {yrs}: {nsites} sites across {nfiles} file(s)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
