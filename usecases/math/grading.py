#!/usr/bin/env python3
"""The one final-answer extractor and the one grader for math: the training rule score
(reward.py) and the MATH-500 / AIME eval (eval.py) both use these, so "correct" means the same
thing in both.

extract_answer(text) -> the content of the LAST \\boxed{...} (or \\fbox{...}, brace-balanced),
    else the LAST explicit `#### <number>`, else None. Never a bare number: the last number in a
    truncated chain of thought is not an answer, and a later box always beats an earlier ####.

grade(pred, gold) -> verl's prime_math.grade_answer at the pinned commit: the MATH (mathd)
    normalizer, then sympy on the normalized forms. Its numeric policy is EXACT after
    normalization and symbolic simplification --
        0.75 == \\frac{3}{4}, 2\\sqrt{2} == \\sqrt{8}, x^2+1 == 1+x^2, (1,2) == (1, 2)
        0 != 0.00001, 50 != 0.5 (no percent rescaling), \\frac{2}{4} != \\frac{1}{2}
    (MATH answers are in lowest terms) -- with one tolerance: a value within 1e-7 of an integer
    is that integer. See usecases/math/tests/test_grading.py for the labelled edge cases.

Model output is untrusted. prime_math.grade_answer refuses to hand anything with more than two
unknown letters to sympy's eval-based parser; prime_math.math_equal is NOT used, because it
passes raw predictions to that parser and, for matrices, to Python's eval(). Each sympy
comparison runs in verl's forked child process with a 10 s limit, and predictions longer than
MAX_PRED_CHARS are not graded (no MATH answer is that long), so a grade is bounded. It never
raises: an answer it cannot grade is not correct.
"""
from __future__ import annotations

import re
from typing import Any

MAX_PRED_CHARS = 400
_HASH_RE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")


def last_boxed(s: str) -> str | None:
    """Content of the LAST \\boxed{...}/\\fbox{...}, brace-balanced (so \\boxed{\\frac{1}{2}} ->
    `\\frac{1}{2}`, not `\\frac{1`). None if absent or unclosed."""
    key = "\\boxed"
    i = s.rfind(key)
    if i < 0:
        key = "\\fbox"
        i = s.rfind(key)
        if i < 0:
            return None
    j = i + len(key)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    depth, start = 0, j
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
        j += 1
    return None


def extract_answer(text: str) -> str | None:
    boxed = last_boxed(text or "")
    if boxed is not None:
        return boxed.strip()
    hits = _HASH_RE.findall(text or "")
    return hits[-1].replace(",", "") if hits else None


def grade(pred: str | None, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    pred, gold = str(pred).strip(), str(gold).strip()
    if not pred or not gold or len(pred) > MAX_PRED_CHARS:
        return False
    from verl.utils.reward_score import prime_math   # verl's own grader, at the pinned commit
    try:
        return bool(prime_math.grade_answer(pred, gold))
    except Exception:  # noqa: BLE001 - an answer that cannot be graded is not correct
        return False
