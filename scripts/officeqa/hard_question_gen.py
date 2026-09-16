#!/usr/bin/env python3
"""Truth-by-construction HARD question generator for OfficeQA path-report training (Phase 2).

WHY (see memory/officeqa-grpo-phase1-result.md): the real bank is tiny (113 easy / 133 hard);
the 133 hard are the HELD-OUT benchmark and must stay untouched. To train grounding on hard
multi-cell table arithmetic we synthesize NEW hard questions whose answers are correct BY
CONSTRUCTION -- we pick the (table, cells, operation), compute the answer from the real
Treasury-Bulletin bytes, and record the exact cited lines as the golden path. No model is needed
for the answer (only, optionally, to polish wording), which is why "lighter validation" suffices
(owner decision): the reward already gates fabrication/answer/support at rollout time.

The corpus (treasury_bulletins_clean.zip) is line-numbered .txt with Markdown tables (from
parse_html_tables). Two monthly layouts occur in the wild:
  A) rows = months (a "... or month" column holds Jan..Dec), columns = expenditure categories
     -> sum a CATEGORY COLUMN across the 12 month-rows of a year.  (matches UID0003)
  B) rows = categories, columns = 12 month cells
     -> sum a CATEGORY ROW's 12 month-cells.
This first cut implements layout A + the year-sum template (the most common hard shape). More
templates (two-year %-change/difference, month-range mean, CPI-adjust) slot in behind TEMPLATES.

Line numbers are 0-indexed to match what the agent's read_document tool prints (start_line=0).

Modes:
  --explore FILE...   : print monthly tables found + a few constructed triples (dry run, no write)
  --emit   FILE...    : write constructed triples as JSONL to --out
Run without a Databricks dep; operates on a local unzipped corpus dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
# Full names + Treasury abbreviations (with/without trailing '.'), month -> 1..12.
_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_ALPHA_RE = re.compile(r"[A-Za-z]{3,9}")
_FOOTNOTE_TAIL = re.compile(r"\s+\d+/\s*$")          # trailing "3/", "12/" footnote markers
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# in-table calendar-year total rows (independent cross-check on a 12-month sum)
_CALYR_RE = re.compile(r"\bcal(?:\.|endar)?\s*(?:yr\.?|year)\b", re.IGNORECASE)


def month_of(label: str) -> tuple[int | None, str | None]:
    """Return (month_number 1..12 or None, 4-digit year or None) parsed from a row label.

    Handles 'January', 'Jan.', '1953-Jan.', 'Sept.', bare 'Feb.' etc. Takes the first alpha
    token that is a known month; year is any 4-digit 19xx/20xx in the label.
    """
    year_m = _YEAR_RE.search(label)
    year = year_m.group(0) if year_m else None
    for tok in _ALPHA_RE.findall(label):
        mn = _MONTH_MAP.get(tok.lower())
        if mn:
            return mn, year
    return None, year


def parse_number(raw: str):
    """Parse a Treasury-table cell to float, or None if not a number.

    Handles: thousands commas (1,902), leading +/-, parenthesized negatives, trailing footnote
    markers (1,580 3/), $ prefix, and the many null spellings (-, --, blank, n.a.).
    """
    if raw is None:
        return None
    s = raw.strip()
    if s in ("", "-", "--", "---", "----", "(-)", "n.a.", "N.A.", "*", "..."):
        return None
    s = _FOOTNOTE_TAIL.sub("", s).strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    s = s.replace(",", "").replace("$", "").strip()
    if s.startswith("+"):
        s = s[1:].strip()
    if s.startswith("-"):
        neg, s = True, s[1:].strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)", s)
    if not m:
        return None
    val = float(m.group(1))
    return -val if neg else val


def _split_row(line: str) -> list[str]:
    """Split a Markdown '| a | b |' row into unescaped cell strings."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.replace("\\|", "|").strip() for c in s.split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?", line)) and set(line.strip()) <= set("|-: ")


_CAPTION_RE = re.compile(r"^\s*Table\s+[0-9]+\.?-?\s*(.+?)\s*$", re.IGNORECASE)
_UNITS_RE = re.compile(r"\(\s*(?:in\s+)?((?:millions?|thousands?|billions?)\s+of\s+dollars"
                       r"|dollars?)\b[^)]*\)", re.IGNORECASE)
# a table is a monthly DOLLAR-FLOW table (summable) when its caption is an expenditure/receipt
# statement -- exactly the benchmark's distribution; excludes debt-maturity/price-index/livestock.
_FLOW_CAPTION_RE = re.compile(r"\b(expenditure|receipt|outlay|outgo|revenue|collection)", re.IGNORECASE)


@dataclass
class Table:
    file: str
    header_line: int            # 0-indexed line of the header row
    headers: list[str]
    rows: list[tuple[int, list[str]]] = field(default_factory=list)  # (line_no, cells)
    caption: str = ""           # nearest preceding 'Table N.- ...' title
    units: str = ""             # e.g. 'millions of dollars' from a '(In ...)' line


def _caption_units(lines: list[str], header_idx: int) -> tuple[str, str]:
    """Scan up to 6 non-table lines above a table header for its 'Table N.-' caption + units."""
    caption = units = ""
    seen = 0
    for k in range(header_idx - 1, max(-1, header_idx - 8), -1):
        ln = lines[k].strip()
        if not ln or ln.startswith("|"):
            if ln.startswith("|"):
                break  # ran into the previous table
            continue
        seen += 1
        if not units:
            um = _UNITS_RE.search(ln)
            if um:
                units = um.group(1).lower()
        cm = _CAPTION_RE.match(ln)
        if cm:
            caption = cm.group(1)
            break
        if seen >= 6:
            break
    return caption, units


def extract_tables(path: str) -> list[Table]:
    """Extract Markdown tables from a line-numbered .txt corpus file (0-indexed lines)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    tables: list[Table] = []
    i, n = 0, len(lines)
    fname = os.path.basename(path)
    while i < n:
        # a table = a '|' header line, then a '---' separator, then '|' data rows
        if lines[i].lstrip().startswith("|") and i + 1 < n and _is_separator(lines[i + 1]):
            headers = _split_row(lines[i])
            caption, units = _caption_units(lines, i)
            t = Table(file=fname, header_line=i, headers=headers, caption=caption, units=units)
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|") and not _is_separator(lines[j]):
                t.rows.append((j, _split_row(lines[j])))
                j += 1
            if t.rows:
                tables.append(t)
            i = j
        else:
            i += 1
    return tables


def is_flow_table(t: Table) -> bool:
    """True if the table caption marks it as a monthly dollar-flow statement (summable)."""
    return bool(t.caption) and bool(_FLOW_CAPTION_RE.search(t.caption))


def _month_label_col(t: Table) -> int | None:
    """Layout A: index of the column whose cells are month names, else None.

    Requires >=6 distinct months so we don't match a stray 'January 31, 1934' header.
    """
    for c in range(len(t.headers)):
        months_seen = set()
        for _, cells in t.rows:
            if c < len(cells):
                mn, _yr = month_of(cells[c])
                if mn:
                    months_seen.add(mn)
        if len(months_seen) >= 6:
            return c
    return None


def _numeric_category_cols(t: Table, label_col: int) -> list[int]:
    """Columns (other than label_col) that are mostly numeric across month rows."""
    out = []
    for c in range(len(t.headers)):
        if c == label_col:
            continue
        num = tot = 0
        for _, cells in t.rows:
            if label_col < len(cells) and month_of(cells[label_col])[0]:
                tot += 1
                if c < len(cells) and parse_number(cells[c]) is not None:
                    num += 1
        if tot and num / tot >= 0.6:
            out.append(c)
    return out


def _calyr_total(t: Table, label_col: int, cat_col: int, year: str):
    """If the table has a 'Cal. yr.' summary row within `year`'s block, return its value for
    cat_col -- an independent in-table cross-check on a computed 12-month sum. Else None."""
    cur_year = None
    for _, cells in t.rows:
        if label_col >= len(cells):
            continue
        lab = cells[label_col]
        ym = _YEAR_RE.search(lab)
        if ym:
            cur_year = ym.group(0)
        if _CALYR_RE.search(lab) and cur_year == year and cat_col < len(cells):
            return parse_number(cells[cat_col])
    return None


def _clean_cat(cat: str) -> str:
    """Cleaned display category: drop footnote markers / cross-ref parentheticals / all
    double-quotes / the HTML-parser dedup suffix ('debt_2' -> 'debt'), collapse whitespace.
    Footnote numbers are 1-2 digits so we never strip a 4-digit year ('Act of 1937')."""
    c = (cat or "").strip().replace('"', "")
    c = re.sub(r"\s*\((?:see|footnote)[^)]*\)\s*$", "", c, flags=re.IGNORECASE)  # cross-ref
    c = re.sub(r"\s+(?:\d{1,2}[a-z]?/\s*)+$", "", c)      # trailing footnotes '2h/', '3/5/', '2/ 3/'
    c = re.sub(r"\s+\d{1,2}[a-z]?$", "", c)               # trailing lone footnote number '1'
    c = re.sub(r"_\d+$", "", c)                           # parser dedup suffix
    return re.sub(r"\s+", " ", c).strip()


_CAT_LEAD_BLOCK = re.compile(r"^(change|total|memorandum|category|net change|col_)\b", re.IGNORECASE)
# Unit-less / ratio-type columns: summing or averaging these across months is not meaningful
# (the benchmark aggregates dollar flows, not percentages/rates/indexes).
_CAT_SEMANTIC_BLOCK = re.compile(
    r"%|\bper ?cent|\bpercent|\brate\b|\byield|\bratio\b|\bindex\b|\baverage\b|"
    r"\bper capita\b|\bprice\b|\bweight|\bcents\b|\bnumber\b", re.IGNORECASE)
# Accounting reconciliation / balancing lines: arithmetically summable but not meaningful
# Treasury quantities (near-zero-crossing -> absurd %-changes), so drop them.
_CAT_ARTIFACT_BLOCK = re.compile(
    r"\badjustment\b|\bclearing account\b|\bnet difference\b|\breporting method\b", re.IGNORECASE)


def is_good_category(cat: str) -> bool:
    """Keep only clean, unambiguous, aggregatable dollar-flow line items; drop artifacts.

    Rejects: col_N, pure years/numbers, delta columns ('Change 1940 to 1941'), 'Total'/
    'Memorandum' rollups, unit-less ratio/rate/index/price columns (summing them is
    meaningless), and anything too short/long or not mostly a noun phrase.
    """
    c = _clean_cat(cat)
    leaf = c.split(">")[-1].strip()
    if not c or len(c) < 4 or len(c) > 90:
        return False
    if re.search(r"col_\d+", c):
        return False
    if _YEAR_RE.fullmatch(leaf) or leaf.replace(".", "").isdigit():
        return False
    if _CAT_LEAD_BLOCK.match(leaf) or re.search(r"\bchange\b", c, re.IGNORECASE):
        return False
    if _CAT_SEMANTIC_BLOCK.search(c) or _CAT_ARTIFACT_BLOCK.search(c):
        return False
    if leaf[:1].isdigit():          # numeric-code / maturity-bucket leaves ('440 fire', '1-5 years')
        return False
    if sum(ch.isalpha() for ch in c) < 3:
        return False
    return True


def _norm_cat(cat: str) -> str:
    return re.sub(r"\s+", " ", _clean_cat(cat).lower())


# --------------------------------------------------------------------------------------------
# Canonical cross-source index: (category, year) -> the 12-month vector, kept ONLY if every
# bulletin that reports it agrees (well-posedness: the answer must not depend on which reprint
# the agent finds). Templates then compute answers purely from these canonical vectors, so all
# of them are unambiguous by construction.
# --------------------------------------------------------------------------------------------

@dataclass
class Canon:
    category: str            # display (cleaned) category header
    year: str
    vector: list[float]      # 12 monthly values, month order Jan..Dec
    consistent: bool         # every reporting bulletin agreed on the vector
    n_sources: int
    src_file: str            # representative source
    src_header_line: int
    cited_cells: list        # [{line, month, raw, value}] from the representative source
    calyr_total: float | None  # in-table published Cal.-yr. total (extra cross-check), if any
    units: str = ""          # 'millions of dollars' etc., from the table caption


def _monthly_series(t: Table):
    """Yield (display_cat, year, vector[12], cited_cells, calyr_total) for every category column
    that is fully present (12 months, no nulls) for a year -- flow tables only (summable)."""
    if not is_flow_table(t):
        return
    label_col = _month_label_col(t)
    if label_col is None:
        return
    cat_cols = _numeric_category_cols(t, label_col)
    if not cat_cols:
        return
    years: dict[str, list] = {}
    cur_year = None
    for ln, cells in t.rows:
        if label_col >= len(cells):
            continue
        mn, yr = month_of(cells[label_col])
        if yr:
            cur_year = yr
        if mn and cur_year:
            years.setdefault(cur_year, []).append((ln, mn, cells))
    for year, mrows in years.items():
        if sorted({m for _, m, _ in mrows}) != list(range(1, 13)) or len(mrows) != 12:
            continue
        mrows = sorted(mrows, key=lambda r: r[1])  # month order
        for c in cat_cols:
            cat = _clean_cat(t.headers[c].strip())
            if not is_good_category(cat):
                continue
            vec, cited, ok = [], [], True
            for ln, mn, cells in mrows:
                v = parse_number(cells[c]) if c < len(cells) else None
                if v is None:
                    ok = False
                    break
                vec.append(v)
                cited.append({"line": ln, "month": MONTHS[mn - 1], "raw": cells[c].strip(),
                              "value": v})
            if ok and len(vec) == 12:
                yield cat, year, vec, cited, _calyr_total(t, label_col, c, year), t.units


def build_canonical_index(files: list[str]) -> dict[tuple[str, str], Canon]:
    """Aggregate monthly series across all bulletins; keep (category, year) only where every
    reporting bulletin agrees on the full 12-value vector (consistent == True)."""
    agg: dict[tuple[str, str], dict] = {}
    for f in files:
        for t in extract_tables(f):
            for cat, year, vec, cited, calyr, units in _monthly_series(t):
                key = (_norm_cat(cat), year)
                rounded = tuple(round(v, 4) for v in vec)
                e = agg.setdefault(key, {"display": cat, "vectors": {}, "n": 0,
                                         "rep": None, "calyr": None, "units": ""})
                e["n"] += 1
                e["vectors"][rounded] = e["vectors"].get(rounded, 0) + 1
                if e["rep"] is None:
                    e["rep"] = (t.file, t.header_line, cited)
                if calyr is not None and e["calyr"] is None:
                    e["calyr"] = calyr
                if units and not e["units"]:
                    e["units"] = units
    index = {}
    for key, e in agg.items():
        consistent = len(e["vectors"]) == 1
        # canonical vector = the modal (most-reported) one; consistent flags no disagreement
        vec = list(max(e["vectors"].items(), key=lambda kv: kv[1])[0])
        rep_file, rep_hdr, rep_cited = e["rep"]
        index[key] = Canon(category=e["display"], year=key[1], vector=vec, consistent=consistent,
                            n_sources=e["n"], src_file=rep_file, src_header_line=rep_hdr,
                            cited_cells=rep_cited, calyr_total=e["calyr"], units=e["units"])
    return index


# --------------------------------------------------------------------------------------------
# Templates over canonical entries. Truth by construction: answer computed from canonical vector.
# --------------------------------------------------------------------------------------------

def _units_phrase(cc: Canon) -> str:
    return f"in {cc.units}" if cc.units else "in the table's reported units"


def _single_input(cc: Canon) -> dict:
    return {"role": "year", "year": cc.year, "file": cc.src_file,
            "table_header_line": cc.src_header_line, "cited_cells": cc.cited_cells,
            "subtotal_sum": round(sum(cc.vector), 4)}


def t_year_sum(cc: Canon) -> dict:
    ans = round(sum(cc.vector), 4)
    return {
        "template": "year_sum",
        "question": (f'Using only the reported values for all individual calendar months in '
                     f'{cc.year}, what is the total sum of the monthly values of "{cc.category}" '
                     f'({_units_phrase(cc)})?'),
        "answer": ans,
        "golden_path": {"operation": "sum(12 monthly values)", "category": cc.category,
                        "years": [cc.year], "units": cc.units, "inputs": [_single_input(cc)]},
        "xcheck_calyr_total": cc.calyr_total,
        "xcheck_abs_diff": (None if cc.calyr_total is None else round(abs(cc.calyr_total - ans), 4)),
    }


def t_year_mean(cc: Canon) -> dict:
    ans = round(sum(cc.vector) / 12.0, 2)
    return {
        "template": "year_mean",
        "question": (f'Using only the reported values for all individual calendar months in '
                     f'{cc.year}, what is the arithmetic mean of the monthly values of '
                     f'"{cc.category}", rounded to the nearest hundredth ({_units_phrase(cc)})?'),
        "answer": ans,
        "golden_path": {"operation": "mean(12 monthly values)", "category": cc.category,
                        "years": [cc.year], "units": cc.units, "inputs": [_single_input(cc)]},
    }


def t_halfyear_mean(cc: Canon) -> dict:
    """Mean over July..December (a consecutive month sub-range)."""
    sub = cc.vector[6:12]
    ans = round(sum(sub) / len(sub), 2)
    return {
        "template": "halfyear_mean",
        "question": (f'Using only the reported monthly values from July {cc.year} to December '
                     f'{cc.year} inclusive, what is the arithmetic mean of the monthly values of '
                     f'"{cc.category}", rounded to the nearest hundredth ({_units_phrase(cc)})?'),
        "answer": ans,
        "golden_path": {"operation": "mean(Jul..Dec monthly values)", "category": cc.category,
                        "years": [cc.year], "units": cc.units,
                        "inputs": [{**_single_input(cc), "months_used": MONTHS[6:12]}]},
    }


def t_year_geomean(cc: Canon) -> dict | None:
    """Geometric mean over the 12 months (only when all values are strictly positive)."""
    if any(v <= 0 for v in cc.vector):
        return None
    prod = 1.0
    for v in cc.vector:
        prod *= v
    ans = round(prod ** (1.0 / 12.0), 2)
    return {
        "template": "year_geomean",
        "question": (f'Using only the reported values for all individual calendar months in '
                     f'{cc.year}, what is the geometric mean of the monthly values of '
                     f'"{cc.category}", rounded to the nearest hundredth ({_units_phrase(cc)})?'),
        "answer": ans,
        "golden_path": {"operation": "geomean(12 monthly values)", "category": cc.category,
                        "years": [cc.year], "units": cc.units, "inputs": [_single_input(cc)]},
    }


def _two_input(cc: Canon, role: str) -> dict:
    return {"role": role, "year": cc.year, "file": cc.src_file,
            "table_header_line": cc.src_header_line, "cited_cells": cc.cited_cells,
            "subtotal_sum": round(sum(cc.vector), 4)}


def t_two_year_diff(a: Canon, b: Canon) -> dict:
    """Absolute difference of the two years' 12-month totals (a earlier, b later)."""
    sa, sb = round(sum(a.vector), 4), round(sum(b.vector), 4)
    ans = round(abs(sb - sa), 4)
    units = f"in {a.units}" if a.units else "in the table's reported units"
    return {
        "template": "two_year_diff",
        "question": (f'Using only the reported values for all individual calendar months in '
                     f'{a.year} and all individual calendar months in {b.year}, what is the '
                     f'absolute difference between the total sum of the monthly values of '
                     f'"{a.category}" in those two years ({units})?'),
        "answer": ans,
        "golden_path": {"operation": "abs(sum(year_b) - sum(year_a))", "category": a.category,
                        "years": [a.year, b.year], "units": a.units,
                        "inputs": [_two_input(a, "year_a"), _two_input(b, "year_b")]},
    }


def t_two_year_pct(a: Canon, b: Canon) -> dict | None:
    """Absolute percent change from year a to year b of the 12-month totals."""
    sa, sb = sum(a.vector), sum(b.vector)
    if sa == 0:
        return None
    ans = round(abs((sb - sa) / sa * 100.0), 2)
    return {
        "template": "two_year_pct_change",
        "question": (f'Using only the reported values for all individual calendar months in '
                     f'{a.year} and all individual calendar months in {b.year}, what is the '
                     f'absolute percent change of the total sum of the monthly values of '
                     f'"{a.category}" from {a.year} to {b.year}, rounded to the nearest '
                     f'hundredth and reported as a percent (e.g. 12.34%)?'),
        "answer": ans,
        "golden_path": {"operation": "abs((sum(b)-sum(a))/sum(a)*100)", "category": a.category,
                        "years": [a.year, b.year], "units": a.units,
                        "inputs": [_two_input(a, "year_a"), _two_input(b, "year_b")]},
    }


def generate_candidates(index: dict[tuple[str, str], Canon]) -> list[dict]:
    """All template candidates over the CONSISTENT canonical entries (well-posed only)."""
    cands: list[dict] = []
    canon = [c for c in index.values() if c.consistent]
    for cc in canon:
        cands.append(t_year_sum(cc))
        cands.append(t_year_mean(cc))
        cands.append(t_halfyear_mean(cc))
        g = t_year_geomean(cc)
        if g:
            cands.append(g)
    # two-year: pair the two nearest consistent years for each category
    by_cat: dict[str, list[Canon]] = {}
    for cc in canon:
        by_cat.setdefault(_norm_cat(cc.category), []).append(cc)
    for entries in by_cat.values():
        entries.sort(key=lambda c: c.year)
        for i in range(len(entries) - 1):
            a, b = entries[i], entries[i + 1]
            cands.append(t_two_year_diff(a, b))
            p = t_two_year_pct(a, b)
            if p:
                cands.append(p)
    for c in cands:                                   # stamp shared provenance
        gp = c["golden_path"]
        c["source_files"] = sorted({inp["file"] for inp in gp["inputs"]})
        c["n_sources"] = None
    return cands


def _decade(gp: dict) -> str:
    y = gp["years"][0]
    return y[:3] + "0s"


def curate(cands: list[dict], n: int, per_cat_cap: int = 5) -> list[dict]:
    """Balanced selection: round-robin across templates, cap questions per category, prefer
    spread across decades. Deterministic."""
    from collections import defaultdict
    by_t: dict[str, list] = defaultdict(list)
    for c in cands:
        by_t[c["template"]].append(c)
    for t in by_t:
        by_t[t].sort(key=lambda c: (c["golden_path"]["category"],
                                    str(c["golden_path"]["years"]), c["source_files"]))
    order = sorted(by_t)
    out, cat_count, idx = [], defaultdict(int), {t: 0 for t in order}
    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for t in order:
            i = idx[t]
            while i < len(by_t[t]):
                cand = by_t[t][i]
                i += 1
                cat = cand["golden_path"]["category"]
                if cat_count[cat] < per_cat_cap:
                    out.append(cand)
                    cat_count[cat] += 1
                    progressed = True
                    break
            idx[t] = i
            if len(out) >= n:
                break
    return out


def _corpus_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "treasury_bulletin_*.txt"))))
        else:
            files.append(p)
    return files


def _print_sample(tr: dict) -> None:
    print("\n" + "=" * 80)
    print(f"[{tr['template']}] Q: {tr['question']}")
    print("ANSWER (by construction):", tr["answer"])
    gp = tr["golden_path"]
    print(f"operation: {gp['operation']}  category: {gp['category']!r}  years: {gp['years']}  "
          f"sources: {tr['source_files']}")
    for inp in gp["inputs"]:
        vals = [cc["value"] for cc in inp["cited_cells"]]
        print(f"  {inp['role']} {inp['year']} ({inp['file']} @line {inp['table_header_line']}): "
              f"sum={inp['subtotal_sum']}  months={len(vals)}")
    if tr.get("xcheck_calyr_total") is not None:
        print(f"  in-table Cal.-yr. cross-check: {tr['xcheck_calyr_total']}  "
              f"|diff|={tr['xcheck_abs_diff']}"
              + ("  <-- EXACT" if tr["xcheck_abs_diff"] == 0 else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["explore", "emit", "curate"])
    ap.add_argument("paths", nargs="+", help="corpus dir or .txt file(s)")
    ap.add_argument("--out", default="/tmp/hard_gen/pilot.jsonl")
    ap.add_argument("-n", "--num", type=int, default=150, help="curate: target question count")
    ap.add_argument("--per-cat-cap", type=int, default=5)
    args = ap.parse_args()

    files = _corpus_files(args.paths)
    index = build_canonical_index(files)
    consistent = sum(1 for c in index.values() if c.consistent)
    cands = generate_candidates(index)
    from collections import Counter
    tmpl_mix = Counter(c["template"] for c in cands)
    print(f"files={len(files)}  canonical (category,year)={len(index)}  consistent={consistent}  "
          f"candidates={len(cands)}  mix={dict(tmpl_mix)}")

    if args.mode == "explore":
        for tr in cands[:8]:
            _print_sample(tr)
        return 0

    selected = curate(cands, args.num, args.per_cat_cap) if args.mode == "curate" else cands
    if args.mode == "curate":
        sel_mix = Counter(c["template"] for c in selected)
        yrs = sorted({c["golden_path"]["years"][0] for c in selected})
        print(f"curated {len(selected)} questions  mix={dict(sel_mix)}  "
              f"years={yrs[0]}..{yrs[-1]} ({len(set(yrs))} distinct)")
        for tr in selected[:6]:
            _print_sample(tr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for tr in selected:
            fh.write(json.dumps(tr) + "\n")
    print(f"wrote {len(selected)} triples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
