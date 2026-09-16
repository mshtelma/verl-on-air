#!/usr/bin/env python3
"""Expanded R-gate suite: REAL long trajectories, mutated into labeled hazards.

Gate R (docs/officeqa_rl_plan.md Section 6.5) needs ~150 independent negatives with
matched valid-source controls on REALISTIC trajectories -- the 9 hand-built fixtures
(281-771 chars, 3 questions) were a smoke, not the gate. This generator takes the 51
answer-correct REAL eval rollouts in officeqa_traces.jsonl (16-turn, real tool
outputs, markdown-table cells) and mutates each into labeled right-number /
wrong-route hazards. The mutation IS the label, so no hand-labeling is needed.

Layouts handled (observed in the real traces):
  * markdown rows with a text label cell:  ``2339: | Total | 3,390 | 14% | ... |``
  * markdown rows with a YEAR label cell:  ``266: | 1934 | 2,681 | ... | 507 | ... |``
  * grep hits carry ``file:line:`` prefixes; read_document carries ``line:`` prefixes
  * compute outputs carry DERIVED values (no row/period semantics)

Cases per correct trace (deep-copied; the source bundle is never modified):
  G   control-grounded      unmodified real trajectory                      -> ACCEPT
  A   control-alt-source    gold filename swapped for a different plausible
                            bulletin everywhere in the trajectory (values
                            untouched): VALID alternative source (RG07)      -> ACCEPT
  P+  wrong-period          asked year shifted +1 in the gt row/header cell  -> REJECT
  P-  wrong-period          asked year shifted -1 (same construction)        -> REJECT
  R1  wrong-row             gt row's label cell swapped with sibling row #1  -> REJECT
  R2  wrong-row             ... with sibling row #2                          -> REJECT
  U   unsupported           every tool output containing the gold value is
                            blanked; the model's prose still claims it       -> REJECT
  F   fabricated-evidence   U + an explicit fabricated file:line citation
                            in the final reasoning step                      -> REJECT
  C   missing-component     composite: every tool output NOT containing the
                            gold value is blanked -> components unsupported  -> REJECT
  N1  wrong-column          gold cell swapped with the NEXT numeric cell in
                            the same row (column-layout wrong-period/gran.)  -> REJECT
  N2  wrong-column          ... with the PREVIOUS numeric cell               -> REJECT
  D   anachronistic-source  gold filename renamed to a bulletin dated BEFORE
                            the asked period (a Jan-1939 bulletin cannot hold
                            calendar-1940 actuals) -- the mirror hazard of A   -> REJECT
  W   wrong-answer gate     pred/final tag perturbed (1-in-3 subsample)      -> GATE-0

The 9 hand-built fixtures (rgate_adversarial.jsonl, incl. RG06 composite and RG07
alt-source) are appended unchanged so ONE judge run replays them under the fixed
parser AND scores the expanded suite.

Matching is VALUE-EXACT (cell-level regex), never bare substring: gt '6' must not
light up on '1963' or 'score=6.82'.

    python3 scripts/officeqa/make_rgate_expanded.py
      [--traces officeqa_pilot_records/officeqa_traces.jsonl]
      [--handbuilt scripts/reward/tests/rgate_adversarial.jsonl]
      [--out scripts/reward/tests/rgate_expanded.jsonl] [--max-traces 0]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, "reward")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from reward.officeqa_reward import score_answer  # noqa: E402  (strict contract)

_REPO = os.path.abspath(os.path.join(_SCRIPTS, ".."))
YEAR_RE = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")
YEAR_CELL_RE = re.compile(r"^(19[3-9]\d|20[0-2]\d)(\s*\d/\d?)?$")   # '1940' / '1940 2/'
FILE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt")
NO_MATCH = "(no matches found)"
NO_OUTPUT = "(computation produced no output)"


def _value_re(gt: str) -> re.Pattern:
    """Cell-exact matcher for the gold value: gt '6' must not fire inside '1963',
    '6.82' or '16'; gt '0.13' MAY still match a cell showing '0.13%'."""
    return re.compile(rf"(?<![\d\.,]){re.escape(gt.strip())}(?![\d\.,])")


# ---------------------------------------------------------------- md helpers
def _split_md(ln: str):
    """'2339: | Total | 3,390 |' -> ('2339: ', [' Total ', ' 3,390 '], '|') or None."""
    if ln.count("|") < 2:
        return None
    i = ln.index("|")
    prefix = ln[: i + 1]
    cells = ln[i + 1:].split("|")
    trailing = ""
    if cells and cells[-1].strip() == "":
        trailing = "|"
        cells = cells[:-1]
    return prefix, cells, trailing


def _join_md(prefix: str, cells: list[str], trailing: str) -> str:
    return prefix + "|".join(cells) + trailing


def _is_year_cell(c: str) -> bool:
    return bool(YEAR_CELL_RE.match(c.strip()))


def _is_text_label(c: str) -> bool:
    s = c.strip()
    return bool(re.search(r"[A-Za-z]", s)) and not _is_year_cell(s) and "treasury_bulletin" not in s


def _tool_texts(step: dict) -> list[dict]:
    return [r for r in (step.get("tool_results") or []) if isinstance(r, dict)]


def _all_text_slots(rec: dict):
    for st in rec.get("trajectory") or []:
        if isinstance(st.get("reasoning"), str):
            yield st, "reasoning"
        for tc in st.get("tool_calls") or []:
            for k, v in (tc.get("args") or {}).items():
                if isinstance(v, str):
                    yield tc["args"], k
        for tr in _tool_texts(st):
            if isinstance(tr.get("result"), str):
                yield tr, "result"


def _rename_file(rec: dict, gold: str, alt: str) -> int:
    """Rename gold -> alt filename AND its listing date strings, so the mutation is
    internally consistent: search listings show ``file (YYYY-MM)`` and ``Bulletin
    date: YYYY-MM``; leaving the old date next to the new filename would make a
    valid control look corrupted (A) or muddle the anachronism (D)."""
    n = 0
    mg, ma = FILE_RE.search(gold or ""), FILE_RE.search(alt or "")
    for slot, key in _all_text_slots(rec):
        if gold in slot[key]:
            n += slot[key].count(gold)
            slot[key] = slot[key].replace(gold, alt)
        if mg and ma:
            gy, gm = int(mg.group(1)), int(mg.group(2))
            ay, am = int(ma.group(1)), int(ma.group(2))
            slot[key] = slot[key].replace(f"({gy}-{gm:02d})", f"({ay}-{am:02d})")
            slot[key] = slot[key].replace(f"date: {gy}-{gm:02d}", f"date: {ay}-{am:02d}")
    return n


def _final_step(rec: dict) -> dict | None:
    for st in reversed(rec.get("trajectory") or []):
        if "FINAL_ANSWER" in str(st.get("reasoning") or ""):
            return st
    return (rec.get("trajectory") or [None])[-1]


def _source_files(rec: dict) -> list[str]:
    sf = str(rec.get("source_files") or "")
    return [p.strip() for p in re.split(r"[;\n,]", sf) if p.strip()]


def _alt_filename(gold: str, uid: str) -> str | None:
    m = FILE_RE.search(gold or "")
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    dy = 2 if (sum(ord(c) for c in uid) % 2 == 0) else -2
    alt = f"treasury_bulletin_{y + dy}_{13 - mo:02d}.txt"
    return alt if alt != gold else None


def _perturb(gt: str) -> str:
    m = re.fullmatch(r"\s*([\d,]*\d(?:\.\d+)?)(.*)\s*", gt)
    if not m:
        return gt + " (revised)"
    num, suffix = m.group(1), m.group(2)
    val = float(num.replace(",", ""))
    nd = len(num.split(".")[1]) if "." in num else 0
    wrong = round(val * 1.13, nd)
    out = f"{wrong:,.{nd}f}" if nd else f"{int(wrong):,}"
    if "," not in num:
        out = out.replace(",", "")
    return out + suffix


def _q_years(rec: dict) -> set[str]:
    return set(YEAR_RE.findall(rec.get("question") or ""))


def _shift_year(y: int, direction: int, q_years: set[str]) -> int | None:
    # Corpus tables reach back well before 1939 (historical series in later
    # bulletins -- e.g. the FY1934 row in UID0002), so the floor is 1930, not the
    # bulletin start year.
    cand = y + direction
    if cand >= 1930 and str(cand) not in q_years:
        return cand
    cand = y - direction
    if cand >= 1930 and str(cand) not in q_years:
        return cand
    return None


# ------------------------------------------------------------------ mutators
def mut_G(rec):
    return rec, "grounded_positive", "unmodified real grounded trajectory"


def mut_A(rec):
    files = _source_files(rec)
    if len(files) != 1:
        return None, None, "skip: alt-source control needs exactly one gold file"
    alt = _alt_filename(files[0], rec["uid"])
    if not alt or _rename_file(rec, files[0], alt) == 0:
        return None, None, "skip: gold filename not present in trajectory"
    return rec, "grounded_positive_alt", f"gold file {files[0]} -> valid alternative {alt}"


def _mut_P_dir(rec, gt, direction):
    vre = _value_re(gt)
    q_years = _q_years(rec)
    changed = 0
    for st in rec.get("trajectory") or []:
        for tr in _tool_texts(st):
            if tr.get("name") == "compute":
                continue
            txt = str(tr.get("result") or "")
            if not vre.search(txt):
                continue            # label hygiene: only mutate results that SHOW the gold
            lines = txt.split("\n")
            # Case A: the gt row itself carries exactly one asked-year cell.
            done = False
            for i, ln in enumerate(lines):
                if done or not vre.search(ln):
                    continue
                md = _split_md(ln)
                if not md:
                    continue
                prefix, cells, trailing = md
                yidx = [k for k, c in enumerate(cells)
                        if _is_year_cell(c) and c.strip()[:4] in q_years and not vre.search(c)]
                if len(yidx) != 1:
                    continue
                y = int(cells[yidx[0]].strip()[:4])
                new = _shift_year(y, direction, q_years)
                if new is None:
                    continue
                cells[yidx[0]] = cells[yidx[0]].replace(str(y), str(new), 1)
                lines[i] = _join_md(prefix, cells, trailing)
                changed += 1
                done = True
            # Case B: exactly ONE other line holds the asked year as a header cell.
            if not done:
                hdr = [(i, _split_md(ln)) for i, ln in enumerate(lines)
                       if not vre.search(ln) and _split_md(ln)
                       and any(_is_year_cell(c) and c.strip()[:4] in q_years for c in _split_md(ln)[1])]
                if len(hdr) == 1:
                    i, (prefix, cells, trailing) = hdr[0]
                    yidx = [k for k, c in enumerate(cells) if _is_year_cell(c) and c.strip()[:4] in q_years]
                    if len(yidx) == 1:
                        y = int(cells[yidx[0]].strip()[:4])
                        new = _shift_year(y, direction, q_years)
                        if new is not None:
                            cells[yidx[0]] = cells[yidx[0]].replace(str(y), str(new), 1)
                            lines[i] = _join_md(prefix, cells, trailing)
                            changed += 1
            if changed:
                tr["result"] = "\n".join(lines)
    if not changed:
        return None, None, "skip: no unambiguous asked-year cell found"
    return rec, "negative_wrong_period", f"asked year shifted {'+' if direction > 0 else '-'}1 in {changed} output(s)"


def mut_Pp(rec, gt):
    return _mut_P_dir(rec, gt, +1)


def mut_Pm(rec, gt):
    return _mut_P_dir(rec, gt, -1)


def _mut_R_sib(rec, gt, sibling_rank):
    vre = _value_re(gt)
    for st in rec.get("trajectory") or []:
        for tr in _tool_texts(st):
            if tr.get("name") == "compute":
                continue
            lines = str(tr.get("result") or "").split("\n")
            for i, ln in enumerate(lines):
                if not vre.search(ln):
                    continue
                md = _split_md(ln)
                if not md:
                    continue
                prefix, cells, trailing = md
                lab = [k for k, c in enumerate(cells) if _is_text_label(c)]
                if len(lab) != 1:
                    continue                       # year-labeled rows are period semantics
                sibs = []
                for j, other in enumerate(lines):
                    if j == i:
                        continue
                    md2 = _split_md(other)
                    if not md2:
                        continue
                    _, cells2, _ = md2
                    lab2 = [k for k, c in enumerate(cells2) if _is_text_label(c)]
                    num2 = [k for k, c in enumerate(cells2) if re.search(r"\d", c)]
                    if len(lab2) == 1 and num2 and cells2[lab2[0]].strip() != cells[lab[0]].strip():
                        sibs.append((j, md2, lab2[0]))
                if len(sibs) <= sibling_rank:
                    continue
                j, (prefix2, cells2, trailing2), l2 = sibs[sibling_rank]
                l1 = lab[0]
                cells[l1], cells2[l2] = f" {cells2[l2].strip()} ", f" {cells[l1].strip()} "
                lines[i] = _join_md(prefix, cells, trailing)
                lines[j] = _join_md(prefix2, cells2, trailing2)
                tr["result"] = "\n".join(lines)
                return rec, "negative_wrong_row", f"gt row label swapped with sibling #{sibling_rank + 1}"
    return None, None, "skip: no labeled sibling row to swap with"


def mut_R1(rec, gt):
    return _mut_R_sib(rec, gt, 0)


def mut_R2(rec, gt):
    return _mut_R_sib(rec, gt, 1)


def _strip_support(rec, gt, only_non_gt=False) -> int:
    vre = _value_re(gt)
    n = 0
    for st in rec.get("trajectory") or []:
        for tr in _tool_texts(st):
            txt = str(tr.get("result") or "")
            has_gt = bool(vre.search(txt))
            if only_non_gt and has_gt:
                continue
            if not only_non_gt and not has_gt:
                continue
            tr["result"] = NO_OUTPUT if tr.get("name") == "compute" else NO_MATCH
            n += 1
    return n


def _blank_all_evidence(rec, gt) -> int:
    """v2 unsupported/fabricated blanking: blank EVERY read/grep/compute result plus
    any search result that shows the gold. Only gold-free search LISTINGS survive
    (they prove files exist, nothing more) -- so no cell anywhere supports the
    committed value. The v1 blanking left input snippets a charitable judge could
    re-derive from, which made the label debatable."""
    vre = _value_re(gt)
    n = 0
    for st in rec.get("trajectory") or []:
        for tr in _tool_texts(st):
            name = tr.get("name")
            if name == "search_documents" and not vre.search(str(tr.get("result") or "")):
                continue
            tr["result"] = NO_OUTPUT if name == "compute" else NO_MATCH
            n += 1
    return n


def mut_U(rec, gt):
    if _blank_all_evidence(rec, gt) == 0:
        return None, None, "skip: no tool outputs to blank"
    return rec, "negative_unsupported", "all read/grep/compute outputs blanked; gold-free listings only"


def mut_F(rec, gt):
    if _blank_all_evidence(rec, gt) == 0:
        return None, None, "skip: no tool outputs to blank"
    files = _source_files(rec) or ["treasury_bulletin_1941_01.txt"]
    st = _final_step(rec)
    claim = f"\nI grepped and found: {files[0]}:142: ... {gt} -- confirmed."
    if st is not None:
        st["reasoning"] = claim + "\n" + str(st.get("reasoning") or "")
    return rec, "negative_fabricated", "fabricated file:line citation; tool outputs empty"


def mut_C(rec, gt):
    if len(_source_files(rec)) < 2:
        return None, None, "skip: missing-component hazard needs a composite (>=2 gold files)"
    if _strip_support(rec, gt, only_non_gt=True) == 0:
        return None, None, "skip: no non-gt tool outputs to blank"
    return rec, "negative_missing_component", "composite: only the final-value cell remains visible"


def mut_D(rec, gt):
    files = _source_files(rec)
    if len(files) != 1:
        return None, None, "skip: anachronistic-source needs exactly one gold file"
    q_years = sorted(int(y) for y in _q_years(rec))
    if not q_years:
        return None, None, "skip: no asked year in question"
    asked = q_years[0]
    if asked - 1 < 1936:
        return None, None, "skip: asked year too early for an anachronistic bulletin"
    alt = f"treasury_bulletin_{asked - 1}_01.txt"
    if alt == files[0]:
        return None, None, "skip: anachronistic alt collides with gold file"
    if _rename_file(rec, files[0], alt) == 0:
        return None, None, "skip: gold filename not present in trajectory"
    return rec, "negative_impossible_source", \
        f"gold file {files[0]} -> {alt} (predates asked period {asked})"


def _mut_N_dir(rec, gt, direction):
    """Swap the gold cell with a neighbouring numeric (non-year, non-label) cell in
    the same markdown row: the committed value now sits in the WRONG COLUMN."""
    vre = _value_re(gt)
    for st in rec.get("trajectory") or []:
        for tr in _tool_texts(st):
            if tr.get("name") == "compute":
                continue
            lines = str(tr.get("result") or "").split("\n")
            for i, ln in enumerate(lines):
                if not vre.search(ln):
                    continue
                md = _split_md(ln)
                if not md:
                    continue
                prefix, cells, trailing = md
                gidx = [k for k, c in enumerate(cells) if vre.search(c)]
                if len(gidx) != 1:
                    continue
                g = gidx[0]
                order = list(range(g + direction, len(cells) if direction > 0 else -1, direction))
                tgt = None
                for k in order:
                    c = cells[k]
                    if re.search(r"\d", c) and not _is_year_cell(c) and not _is_text_label(c) \
                            and not vre.search(c):
                        tgt = k
                        break
                if tgt is None:
                    continue
                cells[g], cells[tgt] = cells[tgt], cells[g]
                lines[i] = _join_md(prefix, cells, trailing)
                tr["result"] = "\n".join(lines)
                return rec, "negative_wrong_column", \
                    f"gold cell swapped with {'next' if direction > 0 else 'previous'} numeric cell"
    return None, None, "skip: no gold md-row with a numeric neighbour cell"


def mut_N1(rec, gt):
    return _mut_N_dir(rec, gt, +1)


def mut_N2(rec, gt):
    return _mut_N_dir(rec, gt, -1)


def mut_T(rec, gt):
    """TRANSPLANT (natural wrong route): take a trace whose real committed answer was
    WRONG (or an abstention), strip any visible gold cells, and commit the GOLD
    instead. The judge now sees the model's genuine wrong/failed route landing on
    the right number -- the most realistic lucky-wrong hazard in the suite."""
    _strip_support(rec, gt)
    rec["pred"] = gt
    st = _final_step(rec)
    if st is not None:
        txt = str(st.get("reasoning") or "")
        new, n = re.subn(r"(<FINAL_ANSWER>).*?(</FINAL_ANSWER>)",
                         lambda m: m.group(1) + gt + m.group(2), txt)
        st["reasoning"] = new if n else txt + f"\n<FINAL_ANSWER>{gt}</FINAL_ANSWER>"
    return rec


def mut_W(rec, gt):
    wrong = _perturb(gt)
    rec["pred"] = wrong
    st = _final_step(rec)
    if st is not None:
        txt = str(st.get("reasoning") or "")
        new, n = re.subn(r"(<FINAL_ANSWER>).*?(</FINAL_ANSWER>)",
                         lambda m: m.group(1) + wrong + m.group(2), txt)
        st["reasoning"] = new if n else txt + f"\n<FINAL_ANSWER>{wrong}</FINAL_ANSWER>"
    return rec, "wrong_gated", f"committed answer perturbed to {wrong}"


MUTATORS = ["G", "A", "Pp", "Pm", "R1", "R2", "U", "F", "C", "N1", "N2", "D", "W"]


def make_cases(trace: dict, kind: str = "correct") -> list[dict]:
    gt = str(trace.get("gt") or "").strip()
    base = trace["uid"]
    if kind in ("wrong", "abstain"):
        # Natural wrong-route substrates: one transplant case each (abstentions are
        # subsampled 1-in-3 so they don't dominate the hazard mix).
        if kind == "abstain" and sum(ord(c) for c in base) % 3:
            return []
        rec = mut_T(copy.deepcopy(trace), gt)
        rec["base_uid"] = base
        rec["uid"] = f"{base}_T"
        rec["mutation"] = "T"
        rec["expected"] = "negative_transplanted" if kind == "wrong" else "negative_transplanted_abstain"
        rec["note"] = ("real wrong route, gold transplanted" if kind == "wrong"
                       else "real abstention route, gold transplanted")
        return [rec]
    cases = []
    for name in MUTATORS:
        if name == "W" and sum(ord(c) for c in base) % 3:
            continue                                   # deterministic 1-in-3 W subsample
        rec = copy.deepcopy(trace)
        fn = globals()[f"mut_{name}"]
        rec, expected, note = fn(rec) if name in ("G", "A") else fn(rec, gt)
        if rec is None:
            continue                                   # mutator skip-flag (pattern absent)
        rec["base_uid"] = base
        rec["uid"] = f"{base}_{name}"
        rec["mutation"] = name
        rec["expected"] = expected
        rec["note"] = note
        cases.append(rec)
    return cases


def _is_correct(rec: dict) -> bool:
    pred = str(rec.get("pred") or "").strip()
    return bool(pred) and pred != "DATA NOT AVAILABLE" and \
        score_answer(str(rec.get("gt", "")), pred) > 0


_HANDBUILT_MAP = {"grounded": "grounded_positive", "lucky_wrong_source": "negative_handbuilt",
                  "wrong_gated": "wrong_gated"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(_REPO, "officeqa_pilot_records", "officeqa_traces.jsonl"))
    ap.add_argument("--handbuilt", default=os.path.join(_SCRIPTS, "reward", "tests", "rgate_adversarial.jsonl"))
    ap.add_argument("--out", default=os.path.join(_SCRIPTS, "reward", "tests", "rgate_expanded.jsonl"))
    ap.add_argument("--max-traces", type=int, default=int(os.environ.get("OQ_RGATE_MAX_TRACES", "0")))
    args = ap.parse_args()

    traces = [json.loads(l) for l in open(args.traces) if l.strip()]
    correct, wrong, abstain = [], [], []
    for t in traces:
        pred = str(t.get("pred") or "").strip()
        if _is_correct(t):
            correct.append(t)
        elif pred and pred != "DATA NOT AVAILABLE":
            wrong.append(t)
        else:
            abstain.append(t)
    if args.max_traces > 0:
        correct = correct[: args.max_traces]
    print(f"[rgate] {len(traces)} traces -> {len(correct)} correct / "
          f"{len(wrong)} wrong-answer / {len(abstain)} abstain substrates")

    cases = []
    for t in correct:
        cases.extend(make_cases(t))
    for t in wrong:
        cases.extend(make_cases(t, "wrong"))
    for t in abstain:
        cases.extend(make_cases(t, "abstain"))
    skips = {m: 0 for m in MUTATORS}
    got = {(c["base_uid"], c["mutation"]) for c in cases}
    for t in correct:
        for m in MUTATORS:
            if m == "W":
                continue
            if (t["uid"], m) not in got:
                skips[m] += 1

    n_hb = 0
    if os.path.exists(args.handbuilt):
        for line in open(args.handbuilt):
            if not line.strip():
                continue
            r = json.loads(line)
            r["base_uid"] = r["uid"]
            r["mutation"] = "handbuilt"
            r["expected"] = _HANDBUILT_MAP.get(r.get("expected"), r.get("expected"))
            cases.append(r)
            n_hb += 1

    with open(args.out, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")

    from collections import Counter
    dist = Counter(c["expected"] for c in cases)
    neg = sum(v for k, v in dist.items() if k.startswith("negative_"))
    pos = sum(v for k, v in dist.items() if k.startswith("grounded_positive"))
    print(f"[rgate] wrote {len(cases)} cases ({n_hb} hand-built) -> {args.out}")
    print(f"[rgate] negatives={neg}  positive-controls={pos}  wrong-gated={dist.get('wrong_gated', 0)}")
    print(f"[rgate] expected distribution: {dict(dist)}")
    print(f"[rgate] mutator skips (pattern absent in trace): {skips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
