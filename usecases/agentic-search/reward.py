#!/usr/bin/env python3
"""Rule-based EM reward for the agentic search/RAG (NQ + HotpotQA) tool-agent.

This is the Search-R1 outcome reward, ported to verl's ``compute_score`` contract. It is
DELIBERATELY simple and FAIL-CLOSED, and it needs NO LLM judge -- the whole judge-fragility
surface we fought on OfficeQA (outages, rate limits, rendezvous, non-parse) is gone:

  * The model's final answer must be wrapped in ``<answer> ... </answer>`` (the last such block
    in the trajectory). No answer block -> 0.0 (format gate), exactly like Search-R1.
  * The extracted answer is normalised SQuAD-style (lowercase, drop articles/punctuation, fix
    whitespace) and compared to the gold answer list. ``em`` = normalised exact match against any
    gold; ``cover_em`` (subem) = a gold appears as a substring of the prediction; ``f1`` = token
    overlap (logged, useful for HotpotQA).
  * The OPTIMISED scalar is ``em`` by default (honest, matches Search-R1); set QA_REWARD_METRIC=
    cover_em for a denser early signal. A correct answer always scores 1.0, so standard GRPO advantage
    normalisation applies (UNLIKE the graded OfficeQA reward, this does NOT require
    norm_adv_by_std_in_grpo=False).
  * OPTIONAL retrieval-recall shaping (QA_RETRIEVAL_BONUS, default 0.0 = off, pure Search-R1): add a
    small additive credit to a WRONG answer when a gold answer string was SURFACED in a retrieved
    passage -- detected only inside the ``<tool_response>`` spans of the trajectory, so the model
    cannot farm it by echoing the gold in its own reasoning. This counteracts the recall regression
    seen under pure-EM GRPO (rewarding only the final answer let the policy trade search thoroughness
    for shorter lucky rollouts -> retrieval recall fell 79%->72% while EM rose, capping the gain) and
    it densifies the signal on hard prompts where every sampled answer is wrong (an all-wrong GRPO
    group otherwise has zero advantage). A correct answer still scores exactly 1.0 (>= any wrong+bonus)
    so the total stays in [0,1] and the reward is unchanged when the bonus is off.

verl wiring: naive reward manager (no rate limit needed -- pure CPU),
  reward.custom_reward_function.path=usecases/agentic-search/reward.py
  reward.custom_reward_function.name=compute_score

``solution_str`` is the full decoded trajectory (system+user+assistant turns+tool responses). This
is VERIFIED for the fully-async ToolAgentLoop: tool-response tokens are appended into the response
span (prompt_ids += response_ids; response_mask += [0]*len -- masked out of the LOSS but still inside
the decoded response), so the reward manager's decode includes the retrieved passages. We read the
final ``<answer>`` for EM, and the ``<tool_response>`` spans for the optional retrieval bonus.
``ground_truth`` is the gold answer(s) from ``reward_model.ground_truth`` -- accepted as a list, a
``{"target": [...]}`` dict, or a string.

Pure stdlib -> imports and runs on CPU with no verl (see usecases/agentic-search/tests/test_reward.py).
"""
from __future__ import annotations

import os
import re
import string
from typing import Any

# Optimised metric: "em" (strict, default) or "cover_em" (substring, denser early signal).
_REWARD_METRIC = os.environ.get("QA_REWARD_METRIC", "em").strip().lower()
# Small positive credit for producing a well-formed <answer> that is nonetheless wrong. Default 0
# (Search-R1 uses 0). A tiny value (e.g. 0.05) can help the model first learn the output format.
_FORMAT_SCORE = float(os.environ.get("QA_FORMAT_SCORE", "0.0"))
# Additive credit for a WRONG answer whose trajectory surfaced a gold answer in a retrieved passage.
# Default 0.0 = off (pure Search-R1 EM, backward-compatible). Keep < 1.0 so a correct answer wins.
_RETRIEVAL_BONUS = float(os.environ.get("QA_RETRIEVAL_BONUS", "0.0"))

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.IGNORECASE | re.DOTALL)


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


def _num_tool_calls(solution_str: str) -> int:
    """Best-effort count of tool invocations in the decoded trajectory (hermes/qwen3 markers).
    Diagnostic only -- never affects the score."""
    s = solution_str or ""
    return s.count("<tool_call>") or s.count("<tool_response>") or len(re.findall(r"\bsearch\b\s*\(", s))


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


def _retrieved_text(solution_str: str) -> str:
    """The text the TOOLS returned (retrieved passages), for the retrieval-recall check.

    Primary: the concatenated content of every ``<tool_response>...</tool_response>`` block -- how the
    qwen3/hermes chat templates wrap tool results in the decoded trajectory. Those wrappers are
    ordinary tokens (not tokenizer specials), so they survive ``skip_special_tokens`` in the reward
    manager's decode. Scoping to them isolates RETRIEVED text from what the model wrote, so the bonus
    cannot be earned by echoing the gold in reasoning.
    Fallback (no such markers -- template variance / stripped): the whole trajectory with the
    ``<answer>`` block(s) removed, so at minimum the model's committed answer can't count as retrieval.
    """
    s = solution_str or ""
    spans = _TOOLRESP_RE.findall(s)
    if spans:
        return "\n".join(spans)
    return _ANSWER_RE.sub(" ", s)


def gold_in_retrieval(solution_str: str, golds_norm: list[str]) -> bool:
    """Did any gold answer appear (as a contiguous normalised token run) in the retrieved passages?
    This is exactly the 'answer-string recall' the offline diagnostic measures, computed at train time.
    """
    if not golds_norm:
        return False
    hay = normalize_answer(_retrieved_text(solution_str)).split()
    return any(_is_sublist(hay, g.split()) for g in golds_norm if g)


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl reward entrypoint. Returns a dict; GRPO optimises ``score``; the rest are logged.

    ``score`` = 1.0 for a correct answer; otherwise QA_FORMAT_SCORE plus QA_RETRIEVAL_BONUS when the
    trajectory surfaced a gold answer in a retrieved passage. Correct always wins and score stays in
    [0,1]. With QA_RETRIEVAL_BONUS=0 (default) this is exactly the old pure-EM reward."""
    golds = _gold_list(ground_truth)
    golds_norm = [normalize_answer(g) for g in golds]

    # Retrieval-recall shaping. Only scanned when enabled, so output is byte-identical when off.
    retrieved = gold_in_retrieval(solution_str, golds_norm) if (_RETRIEVAL_BONUS and golds_norm) else False
    retr_bonus = _RETRIEVAL_BONUS if retrieved else 0.0

    pred = extract_answer(solution_str)
    if pred is None:
        # No <answer> block: format gate. Still credit having SURFACED the gold via retrieval, so a
        # rollout that searched well but failed to answer beats one that did neither.
        return {
            "score": float(retr_bonus), "em": 0.0, "cover_em": 0.0, "f1": 0.0,
            "has_answer": 0.0, "gold_retrieved": float(retrieved), "num_gold": float(len(golds)),
            "n_tool_calls": float(_num_tool_calls(solution_str)),
        }

    pred_norm = normalize_answer(pred)
    em = em_check(pred_norm, golds_norm)
    cover_em = cover_em_check(pred_norm, golds_norm)
    f1 = _f1(pred_norm, golds_norm)

    hit = cover_em if _REWARD_METRIC == "cover_em" else em
    score = 1.0 if hit else (_FORMAT_SCORE + retr_bonus)

    return {
        "score": float(score),
        "em": float(em),
        "cover_em": float(cover_em),
        "f1": float(f1),
        "has_answer": 1.0,
        "gold_retrieved": float(retrieved),
        "num_gold": float(len(golds)),
        "n_tool_calls": float(_num_tool_calls(solution_str)),
    }


if __name__ == "__main__":  # tiny smoke
    traj = "reasoning... <answer> Barack Obama </answer>"
    print(compute_score(solution_str=traj, ground_truth=["Barack Obama", "Obama"]))
    print(compute_score(solution_str="no answer here", ground_truth=["x"]))
