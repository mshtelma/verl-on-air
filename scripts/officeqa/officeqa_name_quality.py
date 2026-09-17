#!/usr/bin/env python3
"""Name-quality filter + probe-cohort selection for the Phase-2 verified bank.

The source-verified bank (officeqa_pipeline.py) proves each question's GOLD is arithmetically
unique under its named vintage, but the concept NAMES are OCR'd table headers and a large fraction
are garbled -- blank leaves, footnote contamination ("Aid to agriculture 1/31"), OCR line-break
splits ("Agri- cultural Depart-ment"), mangled tokens ("Heavy Department" for Navy/War), and dozens
of near-duplicate spellings of one concept ("Veterans' Admire-tretion" vs "Veterans' Administration").
A garbled name verifies numerically but is unusable as an actor-visible PROMPT.

This is a PROMPT-QUALITY gate, deliberately high-precision / low-recall: the bank has ~8k verified
main questions against a ~400-950 target, so we can afford to keep only unambiguously clean concept
names and discard everything else. Two mechanisms, both reproducible:

  1. heuristic hard-rejects  -- blank, digit/fraction/symbol contamination, hyphen+space line
     breaks, and bare short all-caps acronym leaves (too vague/garbled);
  2. a curated CANONICAL allowlist of clean concept leaves, matched on a normalized form
     (lowercased, apostrophes/periods/commas dropped, '&'->'and', punctuation->space) so clean
     spelling/spacing variants collapse to one entry while OCR-mangled variants simply miss it.

`select_cohort` then picks a diverse probe cohort: caps questions per concept, balances the two
compositional templates, and spreads answer magnitude, deterministically (ordered by qid).

CPU-only; reads/writes JSONL. No network, no GPU, launches nothing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

# ------------------------------------------------------------------ normalization + allowlist
_FRAC = set("½¼¾⅓⅔⅛⅜⅝⅞•·°")


def _leaf(concept: str) -> str:
    """Most-specific concept segment: the text after the last '>' hierarchy separator."""
    return concept.split(">")[-1].strip()


def _norm(s: str) -> str:
    """Collapse clean spelling/spacing variants to one key: lowercase, drop apostrophes/periods/
    commas, '&'->'and', all other non-alphanumerics -> single spaces, collapse whitespace."""
    s = s.lower().replace("&", " and ")
    s = s.replace("'", "").replace("’", "").replace(".", "").replace(",", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Curated clean canonical concept leaves (readable raw forms; normalized at import). Only
# unambiguously clean, specific names -- vague singletons ("Other", "Miscellaneous", "Civil",
# "Interest") and OCR garble ("Heavy Department") are intentionally excluded.
_CANON_RAW = [
    # cabinet / executive departments
    "Agriculture Department", "War Department", "Navy Department", "Treasury Department",
    "Commerce Department", "Justice Department", "Labor Department", "State Department",
    "Interior Department", "Post Office Department", "Transportation Department", "Energy Department",
    "Defense Department Military", "Defense Department Civil", "Executive Office of the President",
    "Department of the Army", "Department of the Navy", "Department of the Air Force",
    "Department of Agriculture", "Department of Commerce", "Department of State",
    "Housing & Urban Development Department", "Health, Education, & Welfare Department",
    # agencies / commissions / corporations
    "Veterans' Administration", "Federal Security Agency", "Federal Works Agency",
    "National Housing Agency", "Housing and Home Finance Agency", "Public Housing Administration",
    "General Services Administration", "General Services Admin.", "Environmental Protection Agency",
    "Atomic Energy Commission", "Civilian Conservation Corps", "Reconstruction Finance Corporation",
    "Panama Canal Company", "Commodity Credit Corporation", "Commodity Credit Corp.",
    "Export-Import Bank", "Export-Import Bank of Washington", "Small Business Admin.",
    "Small Business Administration", "National Aeronautics and Space Administration",
    "National Aeronautics and Space Admin.", "United States Maritime Commission",
    "United States Postal Service", "Rural Electrification Administration",
    "Work Projects Administration and National Youth Administration",
    "Selective Service (administrative)", "Funds appropriated to the President",
    "Other independent agencies", "Legislative branch", "The Judiciary", "District of Columbia",
    "Secretary of Defense",
    # trust funds / accounts
    "Federal Old-Age and Survivors Insurance Trust Fund", "Unemployment Trust Fund",
    "Government Life Insurance Fund", "National Service Life Insurance Fund",
    "Railroad Retirement Account", "Federal Disability Insurance Trust Fund", "Highway Trust Fund",
    "Government employees' retirement funds", "Government employees' retirement fund",
    "Other trust accounts", "Other trust funds and accounts",
    # taxes / receipts
    "Interest on the public debt", "Interest on public debt", "Estate and gift taxes",
    "Alcoholic beverage taxes", "Tobacco taxes", "Stamp taxes", "Capital stock tax",
    "Manufacturers' and retailers' excise taxes", "Miscellaneous taxes",
    "Miscellaneous internal revenue", "Income and profit taxes", "Income and profits taxes",
    "Fines, penalties and forfeitures", "Fees for permits and licenses",
    "Fees and other charges for services, etc.", "Sale of Government property", "Sale of products",
    "Realization upon loans and investments", "Recoveries and refunds",
    "Dividends and other earnings", "Royalties", "Rents", "Seigniorage",
    # functions / programs
    "Social security program", "Public works", "National defense",
    "National defense and related activities", "Aid to agriculture", "Agricultural Aid",
    "Interest on uninvested trust funds", "Veterans' services and benefits", "Natural resources",
    "Strategic and critical materials", "Federal contribution to District of Columbia",
    "Transportation and communication", "Finance, commerce, and industry",
]
CANON = {_norm(c) for c in _CANON_RAW}


def heuristic_reject(concept: str) -> str | None:
    """Return a reject reason (str) for a garbled/contaminated concept, or None if it passes."""
    c = (concept or "").strip()
    if not c or not _leaf(c):
        return "blank"
    if any(ch.isdigit() for ch in c):
        return "digit"                       # footnote/revision marker: "1/31", "2½/", "18'"
    if any(ch in _FRAC for ch in c):
        return "fraction/symbol"
    if "$" in c:
        return "symbol"
    if "- " in c or " -" in c:
        return "hyphen-space"                # OCR line break: "Depart- ment", "Washing- ton"
    if re.fullmatch(r"[A-Z]{2,6}\.?", _leaf(c)):
        return "bare-acronym"                # "VPA", "UERRA", "WPA" leaf -- too vague/garbled
    return None


def is_clean(concept: str) -> bool:
    """A concept is prompt-clean iff it passes the heuristic AND its leaf is a canonical name."""
    return heuristic_reject(concept) is None and _norm(_leaf(concept)) in CANON


# ------------------------------------------------------------------ cohort selection
def _spread(items: list, k: int) -> list:
    """Evenly sample k items across a sorted list (endpoints + interior) for magnitude spread."""
    if k >= len(items):
        return items
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
    return [items[i] for i in idx]


def select_cohort(mains: list, target: int, per_concept: int, balance_templates: bool,
                  min_abs: float = 0.0) -> list:
    """Pick <=target clean main questions with concept/template/magnitude diversity, deterministically.

    Round-robins across distinct concept leaves (so no single concept dominates), takes up to
    `per_concept` magnitude-spread questions from each, and -- when balance_templates -- interleaves
    the two compositional families so the rarer monthly `quarter_share_gap` is represented.
    `min_abs` drops near-zero answers (a <~1pp share change rounds within source noise and is a
    low-value analytical target regardless of difficulty) so probe GPU is spent on meaningful tasks."""
    clean = [q for q in mains
             if is_clean(q["golden_path"]["concept"]) and abs(q["answer"]) >= min_abs]
    by_concept: dict[tuple, list] = defaultdict(list)
    for q in clean:
        key = (q["template"], _norm(_leaf(q["golden_path"]["concept"])))
        by_concept[key].append(q)
    # within each (template, concept): dedup captions/year-pairs by qid order, spread by |answer|
    picks: dict[tuple, list] = {}
    for key, qs in by_concept.items():
        qs = sorted(qs, key=lambda q: abs(q["answer"]))
        picks[key] = _spread(qs, per_concept)
    # round-robin across concept groups (optionally template-first) for even coverage
    groups = sorted(picks.keys())
    if balance_templates:
        groups.sort(key=lambda k: (0, k) if k[0] == "quarter_share_gap" else (1, k))
    out, ptr, exhausted = [], defaultdict(int), 0
    while len(out) < target and exhausted < len(groups):
        exhausted = 0
        for key in groups:
            if len(out) >= target:
                break
            if ptr[key] < len(picks[key]):
                out.append(picks[key][ptr[key]])
                ptr[key] += 1
            else:
                exhausted += 1
    return sorted(out, key=lambda q: q["qid"])


# ------------------------------------------------------------------ driver
def _report(mains: list) -> None:
    from collections import Counter
    rej = Counter()
    clean = []
    for q in mains:
        c = q["golden_path"]["concept"]
        r = heuristic_reject(c)
        if r:
            rej[r] += 1
        elif _norm(_leaf(c)) not in CANON:
            rej["not-canonical"] += 1
        else:
            clean.append(q)
    print(f"main questions: {len(mains)}")
    print(f"  rejected: {dict(rej)} (total {sum(rej.values())})")
    print(f"  CLEAN: {len(clean)}")
    cc = Counter(_norm(_leaf(q['golden_path']['concept'])) for q in clean)
    print(f"  distinct clean concepts: {len(cc)}")
    print(f"  by template: {dict(Counter(q['template'] for q in clean))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/oqa_pipeline/questions_full.jsonl")
    ap.add_argument("--clean-out", default="/tmp/oqa_pipeline/clean_main.jsonl")
    ap.add_argument("--cohort-out", default=None, help="also write a selected probe cohort here")
    ap.add_argument("--target", type=int, default=480)
    ap.add_argument("--per-concept", type=int, default=8)
    ap.add_argument("--min-abs", type=float, default=0.0, help="drop |answer| below this (pp)")
    ap.add_argument("--balance-templates", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    mains = [q for q in (json.loads(l) for l in open(a.inp)) if q["pool_role"] == "main"]
    if a.report:
        _report(mains)
    clean = [q for q in mains if is_clean(q["golden_path"]["concept"])]
    with open(a.clean_out, "w") as fh:
        for q in clean:
            fh.write(json.dumps(q) + "\n")
    print(f"wrote {len(clean)} clean main questions -> {a.clean_out}")
    if a.cohort_out:
        cohort = select_cohort(mains, a.target, a.per_concept, a.balance_templates, a.min_abs)
        with open(a.cohort_out, "w") as fh:
            for q in cohort:
                fh.write(json.dumps(q) + "\n")
        from collections import Counter
        print(f"wrote {len(cohort)} cohort questions -> {a.cohort_out}")
        print(f"  cohort templates: {dict(Counter(q['template'] for q in cohort))}")
        print(f"  cohort distinct concepts: {len(set(_norm(_leaf(q['golden_path']['concept'])) for q in cohort))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
