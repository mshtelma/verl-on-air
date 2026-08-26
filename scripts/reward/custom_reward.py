#!/usr/bin/env python3
"""PHASE 2 HOOK — custom reward function.

Not wired in by default. The pipeline currently uses verl's built-in rule-based
scorer, dispatched on the parquet's `data_source` column
(verl/utils/reward_score/__init__.py). To switch to this file:

    CUSTOM_REWARD_PATH=/app/scripts/reward/custom_reward.py \
    CUSTOM_REWARD_NAME=compute_score \
    bash scripts/run_grpo_megatron.sh

or set it in the air YAML's `env_variables:`. The launcher forwards it as
`custom_reward_function.path` / `.name`.

CONTRACT (verified against verl/workers/reward_manager/naive.py:131)
--------------------------------------------------------------------
verl calls this with KEYWORD arguments only:

    compute_score(data_source=str, solution_str=str, ground_truth=Any,
                  extra_info=dict) -> float | dict

Return either:
  * a float — the scalar reward, or
  * a dict with a "score" key — every other key is collected into
    `reward_extra_info` and logged as its own MLflow metric. Returning a dict is
    strongly preferred: it lets you watch accuracy and format-compliance move
    independently, which is the only way to tell "learning to answer" apart
    from "learning to obey the output format".

Signature must be tolerant: accept **kwargs so a future verl release that adds
an argument does not break the run mid-training.
"""

from __future__ import annotations

import re
from typing import Any

# Same contract geo3k's reward enforces; keep prompt instructions in sync with
# scripts/prep_geo3k.py or format reward is unreachable.
#
# MEASURED REWARD SURFACE (run this file directly to reproduce):
#
#   response                                     score  acc  fmt
#   <think>..</think> ... \boxed{42}   correct     1.00   1    1
#   <think>..</think> ... \boxed{7}    wrong       0.10   0    1
#   "the answer is \boxed{42}"         correct     0.90   1    0
#   "42"                               correct     0.00   0    0   <-- !!
#   "no idea"                          wrong       0.00   0    0
#
# NOTE the 4th row. The accuracy term is GATED ON \boxed{} EXTRACTION:
# extract_boxed_content() returns nothing for an unboxed response, so a
# factually correct answer with no \boxed{} scores ZERO, not 0.9. Emitting the
# box is therefore a prerequisite for earning *any* reward, not merely a 10%
# style bonus. Consequence for GRPO: a model that already boxes reliably lives
# in the {0.10, 1.00} regime (good variance), whereas a base model that does not
# box yet sits at a flat 0.00 across the whole group -> zero advantage -> no
# gradient at all. This is the single most important thing to check with
# scripts/baseline_eval.py before choosing a checkpoint.
_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)

FORMAT_WEIGHT = 0.1
ACCURACY_WEIGHT = 0.9


def _format_reward(solution_str: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(solution_str) else 0.0


def _accuracy_reward(solution_str: str, ground_truth: Any) -> float:
    from mathruler.grader import extract_boxed_content, grade_answer

    answer = extract_boxed_content(solution_str)
    return 1.0 if grade_answer(answer, str(ground_truth)) else 0.0


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Default implementation: a dict-returning clone of verl's geo3k scorer.

    Behaviourally identical to the built-in, but reports the components
    separately so MLflow shows them as distinct curves. Replace the body with
    your task's logic in phase 2.
    """
    acc = _accuracy_reward(solution_str, ground_truth)
    fmt = _format_reward(solution_str)
    score = ACCURACY_WEIGHT * acc + FORMAT_WEIGHT * fmt

    return {
        "score": score,       # <- the reward GRPO actually optimises
        "accuracy": acc,      # <- logged separately
        "format": fmt,        # <- logged separately
        "response_chars": float(len(solution_str)),
    }


# ---------------------------------------------------------------------------
# Local sanity check:  python3 scripts/reward/custom_reward.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        (r"<think>angles sum to 180</think> therefore \boxed{42}", "42", "correct + full format"),
        (r"<think>hmm</think> so \boxed{7}", "42", "wrong + full format"),
        (r"the answer is \boxed{42}", "42", "correct + boxed, no <think>"),
        ("42", "42", "correct but UNboxed"),
        ("no idea", "42", "wrong, unformatted"),
    ]
    for response, gt, label in cases:
        out = compute_score(data_source="hiyouga/geometry3k",
                            solution_str=response, ground_truth=gt)
        print(f"{label:24s} -> score={out['score']:.2f} "
              f"acc={out['accuracy']:.0f} fmt={out['format']:.0f}")
