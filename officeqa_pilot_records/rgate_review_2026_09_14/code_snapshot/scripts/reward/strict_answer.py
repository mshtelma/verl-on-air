#!/usr/bin/env python3
"""STRICT typed answer check -- the TRAINING gate for the OfficeQA grounded reward.

This is deliberately SEPARATE from ``officeqa_reward.score_answer`` (the fuzzy,
benchmark-facing scorer). A critical review confirmed the fuzzy scorer accepts, at
tolerance 0, a family of malformed answers that let a policy collect reward without
being right:

    gold "507"          <- "500 or 507 or 509"   (candidate/disjunction list)
    gold "34.4, 0.391"  <- "0.391, 34.4"          (reversed list -- order ignored)
    gold "1, 1"         <- "1"                     (arity ignored -- one value for two)
    gold "543 million"  <- "543 billion"           (explicit unit ignored)
    gold "1"            <- "1e9"                    (sci-notation split into [1, 9])

Under GRPO those are pure reward hacks. The training gate must be STRICT: the
committed answer has to match the gold in ARITY, ORDER, VALUE (exact, Decimal), and
EXPLICIT UNIT, with no candidate lists and correct scientific-notation parsing.

Design choices (see docs/officeqa_rl_plan.md Section 6.4.A):
  * We grade the ONE committed answer string (already extracted from <FINAL_ANSWER>
    by the caller), never tool output or reasoning tags.
  * Numbers compare EXACTLY (Decimal); tolerance 0. The benchmark's 1%/5% bands are
    reported separately for the official metric -- they are not a training target.
  * Thousands separators are normalised ("2,602" == "2602").
  * A UNIT-WORD mismatch is rejected ONLY when BOTH sides carry an explicit unit and
    they differ ("543 billion" vs "543 million"). If the gold string omits the unit
    (OfficeQA gold is often the bare base number, scale implied by the question) we
    do NOT punish a prediction for adding/omitting one -- that matches the benchmark
    and avoids false rejections, while still killing the explicit-unit exploit.
  * TEXT / DATE answers require normalised EXACT equality (not substring), so an
    answer buried in prose does not get substring credit.

Public API: ``strict_correct(gold: str, pred: str) -> StrictResult``.
``StrictResult.valid`` is the boolean the reward gate uses; ``.reason`` is loggable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# One numeric token: optional sign, digits with optional thousands separators, optional
# fraction, optional scientific-notation exponent. Crucially the exponent is part of the
# SAME token so "1e9" parses as 1_000_000_000, not as the two numbers [1, 9].
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Unit words -> canonical name (we compare NAMES, not multipliers: the base number is
# what the gold carries, so we only guard against an explicit, differing scale word).
_UNIT_WORDS = {
    "trillion": "trillion", "trillions": "trillion",
    "billion": "billion", "billions": "billion",
    "million": "million", "millions": "million",
    "thousand": "thousand", "thousands": "thousand",
}
_UNIT_RE = re.compile(r"\b(" + "|".join(_UNIT_WORDS) + r")\b", re.IGNORECASE)
# Disjunction markers that turn an answer into a hedged candidate list.
_DISJUNCTION_RE = re.compile(r"\bor\b|/", re.IGNORECASE)
# A "word" made of letters -- used to detect text/date answers (excludes unit words).
_ALPHA_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class StrictResult:
    valid: bool
    reason: str
    kind: str = ""            # "numeric" | "list" | "text" | "empty"


@dataclass
class _Num:
    value: Decimal
    unit: str | None          # canonical unit word following the number, else None
    percent: bool


def _to_decimal(tok: str) -> Decimal | None:
    try:
        return Decimal(tok.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _strip_units_words(text: str) -> str:
    """Remove unit words + numbers + punctuation for the text/date comparison."""
    t = _UNIT_RE.sub(" ", text)
    t = _NUM_RE.sub(" ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def _parse_numbers(text: str) -> list[_Num]:
    """Ordered numeric tokens with the unit word (if any) that immediately follows,
    and a percent flag. Order is preserved so list order can be enforced."""
    t = text.replace("−", "-").replace("−", "-")
    out: list[_Num] = []
    for m in _NUM_RE.finditer(t):
        raw = m.group()
        if raw in {"-", "+"}:
            continue
        val = _to_decimal(raw)
        if val is None:
            continue
        tail = t[m.end():m.end() + 16]           # small window after the number
        um = _UNIT_RE.match(tail.strip())
        unit = _UNIT_WORDS[um.group(1).lower()] if um else None
        percent = tail.lstrip().startswith("%") or "%" in raw
        out.append(_Num(value=val, unit=unit, percent=percent))
    return out


def _has_text(text: str) -> bool:
    """True if the answer carries meaningful letters beyond unit words (a date/name)."""
    return bool(_ALPHA_RE.search(_UNIT_RE.sub(" ", text)))


def _norm_text(text: str) -> str:
    t = text.strip().strip('"').strip("'").lower()
    t = re.sub(r"\([^)]*\)", " ", t)             # drop parentheticals: "(OASI)" etc.
    t = re.sub(r"[^\w\s.\-/]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- fully-anchored numeric-answer grammar (full-input validation, not extraction) ---
# ONE numeric value, ANCHORED end-to-end: valid thousands grouping (groups of exactly 3)
# OR plain digits OR a leading-decimal, with optional fraction and scientific exponent.
# Anchoring is the point: "not 507" / "approximately 507" cannot pass as a number, and a
# leading ".5" parses as 0.5 (the old `\d`-first regex silently read it as 5).
_NUM_ELEMENT_RE = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|)(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_CURRENCY_RE = re.compile(r"^[\$€£]\s*")
_TRAILING_UNIT_RE = re.compile(r"\s*\b(trillion|billion|million|thousand)s?\b\.?$", re.IGNORECASE)
# List separators. Attempt A keeps a thousands value ("2,602") intact -- a comma is a list
# separator only when FOLLOWED by whitespace (", "), plus ';' and ' and '. Attempt B adds a
# BARE comma so "1,2" -> [1,2]; it is reached only AFTER the whole string failed to parse as
# ONE number, which every valid thousands value ("2,602" -> 2602) already did.
_SPLIT_A_RE = re.compile(r";\s*|\s+and\s+|,\s+")
_SPLIT_B_RE = re.compile(r"[;,]\s*|\s+and\s+")


def _strip_wrappers(s: str) -> str:
    """Strip surrounding quotes and one layer of enclosing brackets: "[1, 2]" -> "1, 2"."""
    s = s.strip().strip('"').strip("'").strip()
    while len(s) >= 2 and (s[0], s[-1]) in (("[", "]"), ("(", ")"), ("{", "}")):
        s = s[1:-1].strip()
    return s


def _num_element(tok: str) -> _Num | None:
    """Parse exactly ONE clean numeric value (optional currency/unit/percent). Returns
    None if `tok` is not entirely a well-formed number -- no extraction from prose."""
    t = tok.strip()
    if not t:
        return None
    t = _CURRENCY_RE.sub("", t).strip()
    percent = t.endswith("%")
    if percent:
        t = t[:-1].strip()
    unit = None
    um = _TRAILING_UNIT_RE.search(t)
    if um:
        unit = _UNIT_WORDS[um.group(1).lower()]
        t = t[:um.start()].strip()
    t = t.replace("−", "-")                 # unicode minus
    if t in ("", "+", "-", ".", "+.", "-."):
        return None
    if not _NUM_ELEMENT_RE.fullmatch(t):
        return None
    val = _to_decimal(t)
    if val is None:
        return None
    return _Num(value=val, unit=unit, percent=percent)


def _parse_answer(s: str) -> list[_Num] | None:
    """A CLEAN scalar or list numeric answer -> ordered [_Num]; anything else -> None.
    None means 'not a well-formed numeric answer', which the gate treats as a mismatch --
    the review's requirement that surrounding prose and malformed lists never pass."""
    core = _strip_wrappers(s)
    if not core:
        return None
    one = _num_element(core)                     # whole string is ONE number? (incl. thousands)
    if one is not None:
        return [one]
    for splitter in (_SPLIT_A_RE, _SPLIT_B_RE):
        parts = [p for p in splitter.split(core) if p.strip()]
        if len(parts) >= 2:
            nums = [_num_element(p) for p in parts]
            if all(n is not None for n in nums):
                return nums  # type: ignore[return-value]
    return None


def strict_correct(gold: str, pred: str) -> StrictResult:
    """Strict typed equality of the committed ``pred`` against ``gold``.

    Returns a StrictResult; ``.valid`` is the reward gate. Never raises.
    """
    gold_s = str(gold or "").strip()
    pred_s = str(pred or "").strip()
    if not pred_s:
        return StrictResult(False, "empty prediction", "empty")
    if not gold_s:
        return StrictResult(False, "empty gold", "empty")

    gold_is_text = _has_text(gold_s)

    # --- TEXT / DATE answers: normalised exact equality (no substring credit) -------
    if gold_is_text:
        gnums, pnums = _parse_numbers(gold_s), _parse_numbers(pred_s)
        gtext, ptext = _strip_units_words(gold_s), _strip_units_words(pred_s)
        if gtext != ptext:
            return StrictResult(False, f"text mismatch: {gtext!r} != {ptext!r}", "text")
        if len(gnums) != len(pnums):
            return StrictResult(False, f"number-count mismatch in text answer "
                                       f"({len(pnums)} vs {len(gnums)})", "text")
        for i, (g, p) in enumerate(zip(gnums, pnums)):
            if g.value != p.value:
                return StrictResult(False, f"number {i} mismatch: {p.value} != {g.value}", "text")
        return StrictResult(True, "exact text/date match", "text")

    # --- NUMERIC / LIST answers (full-input validation via the anchored grammar) -----
    gnums = _parse_answer(gold_s)
    if gnums is None:
        # gold is neither text nor a clean numeric answer -> exact normalised equality.
        return (StrictResult(True, "exact match", "text")
                if _norm_text(gold_s) == _norm_text(pred_s)
                else StrictResult(False, "unparseable gold; text mismatch", "text"))

    # Candidate/disjunction lists ("500 or 507", "500/507") are a hedge, never committed.
    if _DISJUNCTION_RE.search(pred_s):
        return StrictResult(False, "prediction is a candidate/disjunction list", "numeric")

    # The prediction must be a CLEAN numeric answer end-to-end -- no number lifted out of
    # surrounding prose ("not 507"), no malformed number, no leading-decimal misread.
    pnums = _parse_answer(pred_s)
    if pnums is None:
        return StrictResult(False, "prediction is not a clean numeric answer "
                                   "(stray text or malformed number)", "numeric")
    if len(pnums) != len(gnums):
        return StrictResult(False, f"arity mismatch: {len(pnums)} value(s) vs "
                                   f"{len(gnums)} expected", "list" if len(gnums) > 1 else "numeric")

    for i, (g, p) in enumerate(zip(gnums, pnums)):           # ORDER enforced
        if g.value != p.value:
            return StrictResult(False, f"value {i} mismatch: {p.value} != {g.value}", "numeric")
        if g.unit and p.unit and g.unit != p.unit:           # explicit-unit exploit
            return StrictResult(False, f"unit {i} mismatch: {p.unit} != {g.unit}", "numeric")

    kind = "list" if len(gnums) > 1 else "numeric"
    return StrictResult(True, "exact numeric match", kind)


if __name__ == "__main__":
    # The confirmed exploits must all be REJECTED; the sanity cases ACCEPTED.
    exploits = [
        ("507", "500 or 507 or 509"),
        ("34.4, 0.391", "0.391, 34.4"),
        ("1, 1", "1"),
        ("543 million", "543 billion"),
        ("1", "1e9"),
    ]
    sane_ok = [
        ("2,602", "2602"),
        ("34.4, 0.391", "34.4, 0.391"),
        ("543 million", "543"),
        ("March 1977", "March 1977"),
        ("1", "1"),
    ]
    sane_bad = [
        ("507", "508"),
        ("March 1977", "April 1977"),
        ("2,602", "2,603"),
    ]
    print("== exploits (must be INVALID) ==")
    for g, p in exploits:
        r = strict_correct(g, p); print(f"  {'FAIL' if r.valid else 'ok  '} gold={g!r:16s} pred={p!r:20s} -> {r.valid} ({r.reason})")
    print("== sane correct (must be VALID) ==")
    for g, p in sane_ok:
        r = strict_correct(g, p); print(f"  {'ok  ' if r.valid else 'FAIL'} gold={g!r:16s} pred={p!r:20s} -> {r.valid} ({r.reason})")
    print("== sane wrong (must be INVALID) ==")
    for g, p in sane_bad:
        r = strict_correct(g, p); print(f"  {'FAIL' if r.valid else 'ok  '} gold={g!r:16s} pred={p!r:20s} -> {r.valid} ({r.reason})")
