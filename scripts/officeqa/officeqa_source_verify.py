#!/usr/bin/env python3
"""Independent source verification for OfficeQA hard-question inputs (Phase-2 spike).

Extraction PROPOSES an input series (concept, edition, year, months); this module CERTIFIES it
against the ORIGINAL bulletin source and requires the actor-visible corpus to agree.

Why this exists (see memory/officeqa-truth-by-construction-flaws.md): the parsed Markdown corpus
mislabels columns. In "Receipts by Principal Sources" the source header drops a "Total internal
revenue" column, so every label right of it shifts -- the corpus calls 3,257 (=Total internal
revenue) "Customs" when real Customs = 37. BOTH our parsers (parse_html_tables, pd.read_html)
inherit the shift, and even the source HTML's own colspan geometry is inconsistent with the data.
The ONE reliable signal is that the data COLUMN ORDER is canonical, which accounting identities
confirm. So verification = reconcile the source's numeric rows against a per-family accounting
schema, pin the concept by its canonical position, then require the actor's Markdown to match.

Source of truth = the original `<table>` HTML embedded in the HF `databricks/officeqa`
`treasury_bulletins_parsed/jsons/*.json` (UPSTREAM of the Markdown conversion). Two families are
implemented for the spike; more are added by writing a schema, not new code.

verify_series(...) -> {status: verified|conflict|unresolved|..., value_by_month, ...}
  - verified   : identities reconcile, concept pinned, actor Markdown agrees.
  - conflict   : source verified but the actor-visible corpus disagrees -> QUARANTINE (Customs case).
  - unresolved : identities do not reconcile / concept not pinnable / coverage incomplete.
This is a source check for the small set of series we actually use; NOT a universal table engine.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
from scripts.officeqa import hard_question_gen as g  # noqa: E402

TOL = 1.0  # accounting identities are integer millions; allow +/-1 for published-subtotal rounding


# --------------------------------------------------------------------------- family schemas
# An identity is (target_idx, [component_idxs]) meaning value[target] == sum(value[components]).
# A diff-identity is (target_idx, base_idx, [subtrahend_idxs]) meaning target == base - sum(subs).
# Indices are 0-based over the N data columns (excluding the row label).
FAMILY_SCHEMAS = {
    "receipts_by_principal_sources": {
        "caption": re.compile(r"receipts by principal sources", re.I),
        "n_cols": 12,
        # canonical order (confirmed by identities), NOT by the broken header geometry
        "concepts": ["Withheld by employers", "Other income and profits taxes",
                     "Total income and profits taxes", "Employment taxes",
                     "Miscellaneous internal revenue", "Total internal revenue",
                     "Customs", "Other receipts", "Gross receipts",
                     "Appropriations to Federal Old-Age and Survivors Insurance Trust Fund",
                     "Refunds of receipts", "Net receipts"],
        "identities": [(2, [0, 1]), (5, [2, 3, 4]), (8, [5, 6, 7])],
        "diff_identities": [(11, 8, [9, 10])],
        "flow_leaves": [0, 1, 3, 4, 6, 7],          # standalone summable dollar-flow leaves
        "aggregates": [2, 5, 8, 11],                # subtotals/totals -> denominators only
    },
    "analysis_of_general_expenditures": {
        "caption": re.compile(r"analysis of general expenditures", re.I),
        "header_reliable": True,                    # single-row header; map concept by label
        "total_col": 0,                             # data col 0 == 'Total' == sum of the rest
    },
}


# --------------------------------------------------------------------------- html + json helpers
def _cells(tr: str) -> list[str]:
    return [re.sub("<.*?>", "", x).strip() for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)]


def _table_rows(html: str) -> list[list[str]]:
    return [_cells(tr) for tr in re.findall(r"<tr>.*?</tr>", html, flags=re.S)]


def _num(s: str):
    """Strict-ish numeric parse; dash/blank -> 0.0 (nil in these tables), else parse_number/None."""
    if s is None:
        return None
    if s.strip() in ("", "-", "--", "---", "----"):
        return 0.0
    return g.parse_number(s)


def find_table_html(doc: dict, caption_re: re.Pattern) -> str | None:
    """Return the HTML of the table element whose preceding section_header matches caption_re."""
    els = doc["elements"]
    for i, e in enumerate(els):
        if e.get("type") == "section_header" and caption_re.search(str(e.get("content", ""))):
            for e2 in els[i + 1:i + 4]:
                if e2.get("type") == "table":
                    return e2.get("content", "")
    return None


def _monthly_from_rows(rows: list[list[str]], n_cols: int, year: str, col: int) -> dict:
    """Walk data rows tracking the current calendar year; return {month_name: value} at `col`."""
    out, cur = {}, None
    for c in rows:
        if len(c) != n_cols + 1:
            continue
        mn, yr = g.month_of(c[0])
        if yr:
            cur = yr
        if mn and cur == year:
            v = _num(c[col + 1])
            if v is not None:
                out[g.MONTHS[mn - 1]] = v
    return out


def _tol(n_terms: int) -> float:
    """Rounding bound for a sum/diff of n rounded-integer operands plus a rounded result:
    each contributes up to 0.5, so the displayed relation can drift up to 0.5*(n+1)."""
    return max(1.0, 0.5 * (n_terms + 1))


def _identity_residuals(v: list[float], schema: dict) -> list[tuple[set, float, float]]:
    """Per-identity (columns_touched, |residual|, tolerance) for one numeric row."""
    out = []
    for tgt, comps in schema.get("identities", []):
        out.append(({tgt, *comps}, abs(v[tgt] - sum(v[i] for i in comps)), _tol(len(comps))))
    for tgt, base, subs in schema.get("diff_identities", []):
        out.append(({tgt, base, *subs}, abs(v[tgt] - (v[base] - sum(v[i] for i in subs))), _tol(1 + len(subs))))
    return out


def _row_ok(v: list[float], schema: dict, concept_idx: int | None = None) -> bool:
    """All identities hold within rounding tolerance; if concept_idx given, only those touching it."""
    for touched, resid, tol in _identity_residuals(v, schema):
        if concept_idx is not None and concept_idx not in touched:
            continue
        if resid > tol:
            return False
    return True


def _iter_numeric_rows(rows: list[list[str]], n_cols: int):
    """Yield (calendar_year, month_name|None, values[]) for each fully-numeric data row,
    carrying the current calendar year across bare-month rows."""
    cur = None
    for c in rows:
        if len(c) != n_cols + 1:
            continue
        vals = [_num(x) for x in c[1:]]
        if any(v is None for v in vals):
            continue
        mn, yr = g.month_of(c[0])
        if yr:
            cur = yr
        yield cur, (g.MONTHS[mn - 1] if mn else None), vals


# --------------------------------------------------------------------------- verification
def _leaf(s: str) -> str:
    """Normalized header-path leaf, the way the generator selects categories."""
    return g._norm_cat(g._clean_cat(s)).split(">")[-1].strip()


def verify_series(json_path: str, family_key: str, year: str, concept: str,
                  months: list[str] | None = None, caption_re=None) -> dict:
    """Certify one (concept, edition, year[, months]) series against the original source table.

    family_key is a key in FAMILY_SCHEMAS (bespoke, e.g. receipts' broken multi-level header) OR
    the literal 'total_components' for the GENERIC expenditure verifier: auto-detect the 'Total'
    column and require Total == sum(components). Generic mode needs caption_re to locate the table.
    """
    res = {"family": family_key, "year": year, "concept": concept, "edition": os.path.basename(json_path),
           "status": "unresolved", "value_by_month": {}, "detail": ""}
    schema = FAMILY_SCHEMAS.get(family_key)
    generic = family_key == "total_components"
    if schema is None and not generic:
        res["detail"] = f"no schema for family {family_key}"; return res
    cap = schema["caption"] if schema else caption_re
    if cap is None:
        res["detail"] = "generic mode needs caption_re to locate the table"; return res
    try:
        with open(json_path) as fh:
            doc = json.load(fh)["document"]
    except Exception as e:
        res["detail"] = f"json load failed: {e}"; return res
    html = find_table_html(doc, cap)
    if not html:
        res["detail"] = "family table not found in edition"; return res
    rows = _table_rows(html)
    want = months or g.MONTHS

    # resolve concept -> data column index, and the accounting schema used to reconcile it
    if generic or (schema and schema.get("header_reliable")):
        hdr = next((r for r in rows if r and not g.month_of(r[0])[0]
                    and not re.search(r"^\s*(19|20)\d\d", r[0])), rows[0])
        n_cols = len(hdr) - 1
        cidx = next((i - 1 for i in range(1, len(hdr)) if _leaf(hdr[i]) == _leaf(concept)), None)
        if cidx is None:
            res["detail"] = f"concept '{concept}' not in header"; return res
        if schema and "total_col" in schema:
            tc = schema["total_col"]
        else:
            tc = next((i - 1 for i in range(1, len(hdr)) if _leaf(hdr[i]) == "total"), None)
        if tc is None:
            res["detail"] = "no Total column detected"; return res
        if tc == cidx:
            res["detail"] = "concept is the Total column (aggregate, not a summable leaf)"; return res
        idschema = {"identities": [(tc, [i for i in range(n_cols) if i != tc])], "diff_identities": []}
    else:
        n_cols = schema["n_cols"]
        try:
            cidx = schema["concepts"].index(concept)
        except ValueError:
            res["detail"] = f"concept '{concept}' not in schema concepts"; return res
        res["canonical_index"] = cidx
        idschema = schema

    allrows = list(_iter_numeric_rows(rows, n_cols))
    if not allrows:
        res["detail"] = "no fully-numeric data rows"; return res
    frac = sum(1 for _y, _m, v in allrows if _row_ok(v, idschema)) / len(allrows)
    res["identities_frac"] = round(frac, 3)
    tgt = {m: v for (y, m, v) in allrows if y == year and m in want}
    res["value_by_month"] = {m: tgt[m][cidx] for m in tgt}

    # (1) enough rows reconcile to CONFIRM the canonical column order. Multi-term exact-sum
    # identities are near-impossible to satisfy under a wrong column assignment, so a solid
    # minority confirms order; the per-target-row check below is the real per-series guarantee.
    if frac < 0.4:
        res["detail"] = f"column order not confirmed (identities reconcile {frac:.0%} of rows)"; return res
    # (2) the target series is fully covered
    missing = [m for m in want if m not in tgt]
    if missing:
        res["detail"] = f"incomplete coverage: {len(want) - len(missing)}/{len(want)} months"; return res
    # (3) every target row reconciles on the identities TOUCHING this concept (no local OCR damage)
    bad = [m for m in want if not _row_ok(tgt[m], idschema, concept_idx=cidx)]
    if bad:
        res["detail"] = f"{len(bad)} target row(s) fail the concept identity (damage): {bad[:3]}"; return res
    res["status"] = "verified"
    res["vector"] = [tgt[m][cidx] for m in want]
    return res


def actor_view_vector(clean_dir: str, edition: str, caption_re, year: str,
                      concept: str, months: list[str] | None = None) -> dict:
    """What the ACTOR-visible clean Markdown corpus reports for this series (may be corrupted)."""
    path = os.path.join(clean_dir, edition)
    out = {}
    if not os.path.exists(path):
        return out
    cleaf = _leaf(concept)
    for t in g.extract_tables(path):
        if not (t.caption and caption_re.search(t.caption)):
            continue
        lc = g._month_label_col(t)
        if lc is None:
            continue
        for c in g._numeric_category_cols(t, lc):
            # match on the header-path LEAF, the same way the generator/is_good_category selects
            if _leaf(t.headers[c].strip()) != cleaf:
                continue
            cur = None
            for _ln, cells in t.rows:
                if lc >= len(cells):
                    continue
                mn, yr = g.month_of(cells[lc])
                if yr:
                    cur = yr
                if mn and cur == year and c < len(cells):
                    v = g.parse_number(cells[c])
                    if v is not None:
                        out[g.MONTHS[mn - 1]] = v
    return out


def cross_check(json_path: str, clean_dir: str, family_key: str, year: str, concept: str,
                months: list[str] | None = None, caption_re=None) -> dict:
    """verify_series + require the actor-visible Markdown to agree. conflict -> quarantine."""
    v = verify_series(json_path, family_key, year, concept, months, caption_re)
    if v["status"] != "verified":
        return v
    schema = FAMILY_SCHEMAS.get(family_key)
    cre = schema["caption"] if schema else caption_re
    av = actor_view_vector(clean_dir, v["edition"].replace(".json", ".txt"), cre, year, concept, months)
    want = months or g.MONTHS
    disagree = [(m, v["value_by_month"][m], av.get(m)) for m in want
                if av.get(m) is None or abs(av[m] - v["value_by_month"][m]) > TOL]
    v["actor_agrees"] = not disagree
    if disagree:
        v["status"] = "conflict"
        v["detail"] = (f"actor Markdown disagrees on {len(disagree)}/{len(want)} months "
                       f"(e.g. {disagree[0][0]}: source={disagree[0][1]} actor={disagree[0][2]})")
    return v


# --------------------------------------------------------------------------- source-first enumeration
@dataclass
class VerifiedSeries:
    """A source-verified monthly series for one (concept, year, edition) in a flow table, plus the
    table's Total column (the natural denominator for share/ratio questions)."""
    concept: str
    year: str
    edition: str            # json basename, e.g. treasury_bulletin_1949_02.json
    caption: str            # core caption (no 'Table N.-'/units/footnotes)
    family: str
    vector: list            # 12 monthly component values, Jan..Dec
    total_vector: list      # 12 monthly Total (denominator) values, Jan..Dec
    concept_idx: int
    total_idx: int
    actor_agrees: bool      # the clean corpus reports the same 12 values (else quarantine)


def core_caption(cap: str) -> str:
    """Strip a 'Table <id>.-' prefix (numeric '5.-' OR alphanumeric 'FFO-3. -'), '(In ...)' units,
    and trailing footnotes so JSON and clean-corpus captions align across eras."""
    c = re.sub(r"^\s*Table\s+[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*\.?\s*[-.]\s*", "", cap or "", flags=re.I)
    c = re.sub(r"\s*[-–]?\s*\(\s*cont(?:inued)?\b[^)]*\).*$", "", c, flags=re.I)   # drop '(Continued from Table …)'
    c = re.sub(r"\s*\(in .*$", "", c, flags=re.I).strip()
    return re.sub(r"\s+\d{1,2}/.*$", "", c).strip()


_CLEAN_CACHE: dict = {}


def _clean_tables(clean_dir: str, edition_txt: str):
    key = (clean_dir, edition_txt)
    if key not in _CLEAN_CACHE:
        p = os.path.join(clean_dir, edition_txt)
        _CLEAN_CACHE[key] = g.extract_tables(p) if os.path.exists(p) else []
    return _CLEAN_CACHE[key]


def _actor_reports(clean_dir: str, edition_txt: str, core: str, year: str, concept: str, vec: list) -> bool:
    """True iff the actor-visible clean Markdown reports the SAME 12 monthly values for this series."""
    cleaf = _leaf(concept)
    cre = re.compile(re.escape(core), re.I)
    for t in _clean_tables(clean_dir, edition_txt):
        if not (t.caption and cre.search(t.caption)):
            continue
        lc = g._month_label_col(t)
        if lc is None:
            continue
        for c in g._numeric_category_cols(t, lc):
            if _leaf(t.headers[c].strip()) != cleaf:
                continue
            cur, av = None, {}
            for _ln, cells in t.rows:
                if lc >= len(cells):
                    continue
                mn, yr = g.month_of(cells[lc])
                if yr:
                    cur = yr
                if mn and cur == year and c < len(cells):
                    vv = g.parse_number(cells[c])
                    if vv is not None:
                        av[g.MONTHS[mn - 1]] = vv
            if len(av) >= 12:
                return all(abs(av.get(m, 1e9) - vec[k]) <= TOL for k, m in enumerate(g.MONTHS))
    return False


def parse_grid(html: str):
    """Expand an HTML table (colspan/rowspan) into a full grid; return (col_headers, data_rows).
    col_headers[j] = '>'-joined header path for column j (col 0 = the row-label column). data_rows =
    [(label, [raw_cell,...])] for rows whose col 0 is a year/month. This handles the multi-row
    headers (e.g. 'Defense Department' -> Military/Civil) that a naive first-row flatten mis-sizes."""
    grid, pending = [], {}
    for tr in re.findall(r"<tr>.*?</tr>", html, flags=re.S):
        row, c = {}, 0

        def place(c):
            while c in pending:
                txt, rem, ish = pending[c]
                row[c] = (txt, ish)
                if rem - 1 > 0:
                    pending[c] = (txt, rem - 1, ish)
                else:
                    del pending[c]
                c += 1
            return c

        for tag, attr, txt in re.findall(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", tr, flags=re.S):
            c = place(c)
            text = re.sub("<.*?>", "", txt).replace("\\|", "|").strip()
            ish = tag == "th"
            cs = re.search(r'colspan="(\d+)"', attr)
            rs = re.search(r'rowspan="(\d+)"', attr)
            cs, rs = (int(cs.group(1)) if cs else 1), (int(rs.group(1)) if rs else 1)
            for _ in range(cs):
                row[c] = (text, ish)
                if rs > 1:
                    pending[c] = (text, rs - 1, ish)
                c += 1
        place(c)
        grid.append(row)
    if not grid:
        return [], []
    ncols = max((max(r) + 1 if r else 0) for r in grid)

    def cell(r, j):
        return r.get(j, ("", False))

    def is_data(r):
        t = cell(r, 0)[0]
        return bool(g.month_of(t)[0]) or bool(re.match(r"^\s*(19|20)\d\d\b", t))

    header_rows, data = [], []
    for r in grid:
        if is_data(r):
            data.append(r)
        elif not data:                       # header = the rows before the first data row
            header_rows.append(r)
    col_headers = []
    for j in range(ncols):
        seq = []
        for hr in header_rows:
            t = cell(hr, j)[0]
            if t and (not seq or seq[-1] != t):
                seq.append(t)
        col_headers.append(" > ".join(seq))
    data_rows = [(cell(r, 0)[0], [cell(r, j)[0] for j in range(1, ncols)]) for r in data]
    return col_headers, data_rows


def _detect_total_col(col_headers: list, numeric_rows: list, n_cols: int):
    """The Total column: a header-leaf 'total' if present, else the column equal to sum(others) for
    >=60% of rows (identity-based, robust to header damage). None if neither -> table not verifiable."""
    for j in range(n_cols):
        if _leaf(col_headers[j + 1]) == "total":
            return j
    if not numeric_rows:
        return None
    best, best_frac = None, 0.0
    for t in range(n_cols):
        comps = [k for k in range(n_cols) if k != t]
        ok = sum(1 for v in numeric_rows if abs(v[t] - sum(v[k] for k in comps)) <= _tol(len(comps)))
        if ok / len(numeric_rows) > best_frac:
            best, best_frac = t, ok / len(numeric_rows)
    return best if best_frac >= 0.6 else None


def enumerate_verified_series(json_path: str, clean_dir: str, flow_only: bool = True) -> list:
    """SOURCE-FIRST: enumerate every (concept, year) series in an edition's flow tables that is
    source-verified by an in-table Total=sum identity, tagging whether the actor corpus agrees.
    Uses the colspan/rowspan grid parser so multi-row-header tables size correctly."""
    out = []
    try:
        with open(json_path) as fh:
            els = json.load(fh)["document"]["elements"]
    except Exception:
        return out
    edition = os.path.basename(json_path)
    edition_txt = edition.replace(".json", ".txt")

    def prevcap(i):
        for k in range(i - 1, -1, -1):
            if els[k].get("type") == "section_header":
                return els[k].get("content", "")
        return ""

    for i, e in enumerate(els):
        if e.get("type") != "table":
            continue
        cap = core_caption(prevcap(i))
        if flow_only and not g._FLOW_CAPTION_RE.search(cap):   # flows only: no stocks/debt/holdings
            continue
        cols, drows = parse_grid(e["content"])
        if len(cols) < 3:
            continue
        n_cols = len(cols) - 1
        yv, allvals, cur = defaultdict(dict), [], None
        for lab, vals in drows:
            if len(vals) != n_cols:
                continue
            fv = [_num(x) for x in vals]
            if any(v is None for v in fv):
                continue
            mn, yr = g.month_of(lab)
            if yr:
                cur = yr
            allvals.append(fv)
            if mn and cur:
                yv[cur][g.MONTHS[mn - 1]] = fv
        if not allvals:
            continue
        tcol = _detect_total_col(cols, allvals, n_cols)
        if tcol is None:
            continue
        idschema = {"identities": [(tcol, [k for k in range(n_cols) if k != tcol])], "diff_identities": []}
        if sum(1 for v in allvals if _row_ok(v, idschema)) / len(allvals) < 0.4:
            continue
        for cidx in range(n_cols):
            if cidx == tcol or not g.is_good_category(g._clean_cat(cols[cidx + 1])):
                continue
            concept = g._clean_cat(cols[cidx + 1])
            for year, mv in yv.items():
                if not all(m in mv for m in g.MONTHS):
                    continue
                if not all(_row_ok(mv[m], idschema, concept_idx=cidx) for m in g.MONTHS):
                    continue
                vec = [mv[m][cidx] for m in g.MONTHS]
                tvec = [mv[m][tcol] for m in g.MONTHS]
                agrees = _actor_reports(clean_dir, edition_txt, cap, year, concept, vec)
                out.append(VerifiedSeries(concept, year, edition, cap, "total_components",
                                          vec, tvec, cidx, tcol, agrees))
    return out


# --------------------------------------------------------------------------- annual (fiscal-year) reassembly
@dataclass
class AnnualCell:
    """A source-verified fiscal-year figure for one (concept, fiscal_year, edition) in a reassembled
    flow table, with the table's Total for that year (denominator) and actor-corpus agreement."""
    concept: str
    fiscal_year: str
    edition: str
    caption: str
    value: float
    total: float
    actor_agrees: bool


def _bare_year(lab: str) -> str | None:
    """A bare fiscal-year row label ('1970'), NOT a monthly row ('1970-June') or estimate."""
    if g.month_of(lab)[0] is not None:
        return None
    m = re.match(r"^\s*((?:19|20)\d\d)\s*$", (lab or "").strip())
    return m.group(1) if m else None


def _actor_year_value(clean_dir: str, edition_txt: str, concept: str, fy: str):
    """Value the clean corpus reports for `concept` at bare fiscal-year `fy`, matched by the
    concept-leaf COLUMN (caption-independent: 1970s clean captions aren't recognized by the parser)."""
    cleaf = _leaf(concept)
    for t in _clean_tables(clean_dir, edition_txt):
        cols = [c for c in range(len(t.headers)) if _leaf(t.headers[c]) == cleaf]
        if not cols:
            continue
        c = cols[0]
        for _ln, cells in t.rows:
            if cells and _bare_year(cells[0]) == fy and c < len(cells):
                v = g.parse_number(cells[c])
                if v is not None:
                    return v
    return None


def enumerate_annual_series(json_path: str, clean_dir: str, flow_only: bool = True) -> list:
    """Reassemble each flow table's split page-elements (matched by caption), verify Total=sum(all
    components) per bare fiscal-year row, and yield AnnualCell per (verified concept, fiscal-year),
    tagged with actor-corpus agreement. This is the 1970s+ path (annual history, wide split tables)."""
    out = []
    try:
        with open(json_path) as fh:
            els = json.load(fh)["document"]["elements"]
    except Exception:
        return out
    edition = os.path.basename(json_path)
    edition_txt = edition.replace(".json", ".txt")

    def prevcap(i):
        for k in range(i - 1, -1, -1):
            if els[k].get("type") == "section_header":
                return els[k].get("content", "")
        return ""

    groups, order = defaultdict(list), []
    for i, e in enumerate(els):
        if e.get("type") != "table":
            continue
        cap = core_caption(prevcap(i))
        if flow_only and not g._FLOW_CAPTION_RE.search(cap):
            continue
        if cap not in groups:
            order.append(cap)
        groups[cap].append(e["content"])

    for cap in order:
        comp, total, anns = [], None, []          # comp: (half_idx, col, concept); total: (half_idx, col)
        for hi, html in enumerate(groups[cap]):
            cols, drows = parse_grid(html)
            a = {}
            for lab, vals in drows:
                y = _bare_year(lab)
                if not y or len(vals) != len(cols) - 1:
                    continue
                fv = [_num(x) for x in vals]
                if all(v is not None for v in fv):
                    a[y] = fv
            anns.append(a)
            for c in range(len(cols) - 1):
                if _leaf(cols[c + 1]) == "total":
                    if total is None:
                        total = (hi, c)
                elif g.is_good_category(g._clean_cat(cols[c + 1])):
                    comp.append((hi, c, g._clean_cat(cols[c + 1])))
        if total is None or len(comp) < 2:
            continue
        thi, tci = total
        halves = {hi for hi, _c, _n in comp} | {thi}
        years = set(anns[thi])
        for hi in halves:
            years &= set(anns[hi])
        if len(years) < 2:
            continue
        good = {y for y in years
                if abs(sum(anns[hi][y][c] for hi, c, _n in comp) - anns[thi][y][tci]) <= _tol(len(comp))}
        if len(good) < 2 or len(good) / len(years) < 0.5:      # reassembly must reconcile (>=2 clean FYs)
            continue
        for hi, c, concept in comp:
            for y in good:
                val, tot = anns[hi][y][c], anns[thi][y][tci]
                av = _actor_year_value(clean_dir, edition_txt, concept, y)
                out.append(AnnualCell(concept, y, edition, cap, val, tot,
                                      av is not None and abs(av - val) <= _tol(1)))
    return out


# --------------------------------------------------------------------------- spike demo
# The selection spans the classes the reviewer asked for: a clean receipts leaf, the Customs
# misalignment, and the genuine cross-edition revision. (edition-json, family, year, concept)
SPIKE_SERIES = [
    # (edition-json, family, year, concept, months|None) — spans the classes the reviewer asked for
    ("treasury_bulletin_1951_02.json", "receipts_by_principal_sources", "1950", "Employment taxes", None),                # verify (actor cols pre-shift, agree)
    ("treasury_bulletin_1951_02.json", "receipts_by_principal_sources", "1950", "Miscellaneous internal revenue", None),  # verify
    ("treasury_bulletin_1950_03.json", "receipts_by_principal_sources", "1950", "Customs", ["January"]),                  # CONFLICT: source 37 vs actor 3257
    ("treasury_bulletin_1951_02.json", "receipts_by_principal_sources", "1950", "Customs", None),                         # unresolved (July OCR) -> honest exclude
    ("treasury_bulletin_1941_10.json", "analysis_of_general_expenditures", "1941", "Aid to Agriculture", ["January"]),    # verify -> 129
    ("treasury_bulletin_1942_01.json", "analysis_of_general_expenditures", "1941", "Aid to agriculture", ["January"]),    # verify -> 121 (REVISION)
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/oqa_src", help="dir of parsed *.json editions")
    ap.add_argument("--clean", default="/tmp/oqa_corpus/unzipped", help="actor-visible Markdown corpus dir")
    a = ap.parse_args()
    print(f"# source-verification spike: {len(SPIKE_SERIES)} series\n# src={a.src} clean={a.clean}\n")
    for ed, fam, yr, concept, months in SPIKE_SERIES:
        jp = os.path.join(a.src, ed)
        if not os.path.exists(jp):
            print(f"  SKIP (missing edition json): {ed}"); continue
        r = cross_check(jp, a.clean, fam, yr, concept, months)
        vec = r.get("vector")
        s = round(sum(vec), 2) if vec else None
        mlabel = f" {months}" if months else ""
        val = f" value={vec[0]}" if (vec and months and len(months) == 1) else ""
        print(f"[{r['status'].upper():10}] {concept!r:34} {yr}{mlabel} @ {ed[:28]}")
        print(f"       id_ok={r.get('identities_frac')} cov={len(r['value_by_month'])} sum={s}{val}"
              f"{'' if not r.get('detail') else ' | ' + r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
