#!/usr/bin/env python3
"""Rule-based EM reward for the agentic search/RAG tool-agent (MuSiQue multi-hop questions).

This is the Search-R1 outcome reward, ported to verl's ``compute_score`` contract. It is
DELIBERATELY simple and FAIL-CLOSED, and it needs NO LLM judge -- the whole judge-fragility
surface of a judge-based reward (outages, rate limits, rendezvous, non-parse) is gone:

  * The model's final answer must be wrapped in ``<answer> ... </answer>`` -- the last such block
    THE MODEL WROTE. No answer block -> 0.0 (format gate), exactly like Search-R1.
  * The extracted answer is normalised SQuAD-style (lowercase, drop articles/punctuation, fix
    whitespace) and compared to the gold answer list. ``em`` = normalised exact match against any
    gold; ``cover_em`` (subem) = a gold appears as a substring of the prediction; ``f1`` = token
    overlap (logged).
  * The OPTIMISED scalar is ``em`` by default (honest, matches Search-R1); set QA_REWARD_METRIC=
    cover_em for a denser early signal. A correct answer always scores 1.0.
  * OPTIONAL retrieval-recall shaping (QA_RETRIEVAL_BONUS, default 0.0 = off, pure Search-R1): a
    small additive credit for a WRONG answer when a gold answer string appeared in text a TOOL
    returned. QA_FORMAT_SCORE + QA_RETRIEVAL_BONUS must stay in [0, 1), so a wrong answer can never
    score as high as a correct one (checked at import).

WHO WROTE WHAT. ``solution_str`` is the whole decoded episode -- assistant turns, tool responses and
chat-template text -- so the reward cannot tell from the string alone whether an ``<answer>`` was
committed by the model or sits inside a tool response, nor whether "<tool_response>" text is real
or typed by the model. The role-span agent loop (engine/train/role_span_agent_loop.py, registered
as `tool_agent` by the launchers) records it from verl's response_mask and passes it as
``extra_info["role_spans"]``: the answer is read from ASSISTANT spans only, retrieval credit from
TOOL spans only (engine/lib/role_spans.py). Without valid spans the sample scores 0 with
``provenance_ok=0`` and the run is aborted -- a misconfigured agent loop must not train on a
guessed attribution. eval.py scores its structured transcript with the same ``score_segments``.

verl wiring: naive reward manager (no rate limit needed -- pure CPU),
  reward.custom_reward_function.path=usecases/agentic-search/reward.py
  reward.custom_reward_function.name=compute_score

``ground_truth`` is the gold answer(s) from ``reward_model.ground_truth`` -- accepted as a list, a
``{"target": [...]}`` dict, or a string.

Pure stdlib -> imports and runs on CPU with no verl (see usecases/agentic-search/tests/test_reward.py).
"""
from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path
from typing import Any

# engine/lib, located relative to this file (works in the job's code snapshot and in a checkout)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import role_spans  # noqa: E402
import run_control  # noqa: E402

# Optimised metric: "em" (strict, default) or "cover_em" (substring, denser early signal).
_REWARD_METRIC = os.environ.get("QA_REWARD_METRIC", "em").strip().lower()
# Small positive credit for producing a well-formed <answer> that is nonetheless wrong. Default 0
# (Search-R1 uses 0). A tiny value (e.g. 0.05) can help the model first learn the output format.
_FORMAT_SCORE = float(os.environ.get("QA_FORMAT_SCORE", "0.0"))
# Additive credit for a WRONG answer whose trajectory surfaced a gold answer in a retrieved passage.
# Default 0.0 = off (pure Search-R1 EM, backward-compatible). Keep < 1.0 so a correct answer wins.
_RETRIEVAL_BONUS = float(os.environ.get("QA_RETRIEVAL_BONUS", "0.0"))

if not (0.0 <= _FORMAT_SCORE and 0.0 <= _RETRIEVAL_BONUS and _FORMAT_SCORE + _RETRIEVAL_BONUS < 1.0):
    raise ValueError(f"QA_FORMAT_SCORE ({_FORMAT_SCORE}) + QA_RETRIEVAL_BONUS ({_RETRIEVAL_BONUS}) must be "
                     ">= 0 and < 1: a wrong answer must never score as high as a correct one")

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# SQuAD-style normalisation (identical to Search-R1 / official SQuAD/HotpotQA eval).
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def extract_answer(solution_str: str) -> str | None:
    """Return the text of the LAST ``<answer>...</answer>`` block, or None if there is none.

    Using the last block means the model's final committed answer wins even if it emitted an
    ``<answer>`` mid-reasoning (Search-R1 takes the last match)."""
    matches = _ANSWER_RE.findall(solution_str or "")
    if not matches:
        return None
    return matches[-1].strip()


def _gold_list(ground_truth: Any) -> list[str]:
    """Coerce reward_model.ground_truth into a flat list of gold answer strings."""
    if ground_truth is None:
        return []
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("answers", []))
    if isinstance(ground_truth, str):
        return [ground_truth]
    try:
        return [str(g) for g in ground_truth if str(g).strip()]
    except TypeError:
        return [str(ground_truth)]


def em_check(pred_norm: str, golds_norm: list[str]) -> int:
    return int(any(pred_norm == g for g in golds_norm))


def cover_em_check(pred_norm: str, golds_norm: list[str]) -> int:
    """SubEM: a (normalised) gold appears as a substring of the (normalised) prediction."""
    return int(any(g and g in pred_norm for g in golds_norm))


def _f1(pred_norm: str, golds_norm: list[str]) -> float:
    pred_toks = pred_norm.split()
    best = 0.0
    for g in golds_norm:
        gold_toks = g.split()
        if not pred_toks or not gold_toks:
            best = max(best, float(pred_toks == gold_toks))
            continue
        common: dict[str, int] = {}
        for t in pred_toks:
            if t in gold_toks:
                common[t] = min(pred_toks.count(t), gold_toks.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _is_sublist(hay: list[str], needle: list[str]) -> bool:
    """True if ``needle`` occurs as a CONTIGUOUS run in ``hay`` (token-level; avoids the false
    positives a raw substring test gives for short golds, e.g. 'us' inside 'must')."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return False
    first = needle[0]
    for i in range(n - m + 1):
        if hay[i] == first and hay[i:i + m] == needle:
            return True
    return False


def gold_in_retrieval(tool_texts: list[str], golds_norm: list[str]) -> bool:
    """Did a gold answer appear (as a contiguous normalised token run) in text the TOOLS returned?
    The same 'answer-string recall' the offline diagnostic measures, computed at train time."""
    if not golds_norm:
        return False
    hay = normalize_answer("\n".join(tool_texts)).split()
    return any(_is_sublist(hay, g.split()) for g in golds_norm if g)


def _last_answer(assistant_texts: list[str]) -> str | None:
    """The last <answer> block the model wrote (searched per assistant segment, latest first)."""
    for text in reversed(assistant_texts):
        found = extract_answer(text)
        if found is not None:
            return found
    return None


def score_segments(assistant_texts: list[str], tool_texts: list[str], ground_truth: Any) -> dict[str, float]:
    """The reward, given the episode already split into what the model wrote and what tools
    returned. Shared by training (compute_score) and eval.py. ``score`` = 1.0 for a correct answer;
    otherwise QA_FORMAT_SCORE (for a wrong answer) + QA_RETRIEVAL_BONUS if a tool surfaced a gold."""
    golds = _gold_list(ground_truth)
    golds_norm = [normalize_answer(g) for g in golds]
    # always MEASURED (logged as the retrieval-recall curve); only REWARDED when the bonus is on
    retrieved = gold_in_retrieval(tool_texts, golds_norm)
    retr_bonus = _RETRIEVAL_BONUS if retrieved else 0.0
    n_tool_calls = float(sum(t.count("<tool_call>") for t in assistant_texts))

    pred = _last_answer(assistant_texts)
    if pred is None:
        # No <answer> block: format gate. Still credit having SURFACED the gold via retrieval, so a
        # rollout that searched well but failed to answer beats one that did neither.
        return {"score": float(retr_bonus), "em": 0.0, "cover_em": 0.0, "f1": 0.0, "has_answer": 0.0,
                "gold_retrieved": float(retrieved), "num_gold": float(len(golds)), "n_tool_calls": n_tool_calls}

    pred_norm = normalize_answer(pred)
    em = em_check(pred_norm, golds_norm)
    cover_em = cover_em_check(pred_norm, golds_norm)
    hit = cover_em if _REWARD_METRIC == "cover_em" else em
    return {
        "score": 1.0 if hit else float(_FORMAT_SCORE + retr_bonus),
        "em": float(em), "cover_em": float(cover_em), "f1": float(_f1(pred_norm, golds_norm)),
        "has_answer": 1.0, "gold_retrieved": float(retrieved), "num_gold": float(len(golds)),
        "n_tool_calls": n_tool_calls,
    }


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl reward entrypoint. Returns a dict; GRPO optimises ``score``; the rest are logged."""
    try:
        assistant, tool = role_spans.split(solution_str or "", (extra_info or {}).get("role_spans"))
    except role_spans.ProvenanceError as e:
        run_control.request_abort(f"search reward cannot tell what the model wrote: {e}",
                                  "usecases/agentic-search/reward.py")
        out = score_segments([], [], ground_truth)
        return {**out, "score": 0.0, "provenance_ok": 0.0}
    return {**score_segments(assistant, tool, ground_truth), "provenance_ok": 1.0}


if __name__ == "__main__":  # tiny smoke
    traj = "reasoning... <answer> Barack Obama </answer>"
    spans = {"role_spans": [["assistant", 0, len(traj)]]}
    print(compute_score(solution_str=traj, ground_truth=["Barack Obama", "Obama"], extra_info=spans))
    print(score_segments(["no answer here"], [], ["x"]))
