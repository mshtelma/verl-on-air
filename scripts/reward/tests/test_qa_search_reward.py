#!/usr/bin/env python3
"""CPU unit tests for the agentic-search EM reward (scripts/reward/qa_search_reward.py).

Run: PYTHONPATH=scripts python3 -m pytest scripts/reward/tests/test_qa_search_reward.py -q
"""
from __future__ import annotations

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REWARD_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _REWARD_DIR not in sys.path:
    sys.path.insert(0, _REWARD_DIR)

import qa_search_reward as R  # noqa: E402


def _score(traj, gt, **env):
    """Reload the module under a metric env override so the module-level knob is picked up."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        importlib.reload(R)
        return R.compute_score(solution_str=traj, ground_truth=gt)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(R)


# --- normalisation -------------------------------------------------------
def test_normalize_articles_punct_case():
    assert R.normalize_answer("The White House.") == "white house"
    assert R.normalize_answer("  a  CAT ") == "cat"


# --- extraction ----------------------------------------------------------
def test_extract_last_answer_block_wins():
    traj = "<answer> wrong </answer> more <answer> Paris </answer>"
    assert R.extract_answer(traj) == "Paris"


def test_extract_none_when_absent():
    assert R.extract_answer("no tags here") is None


# --- exact match ---------------------------------------------------------
def test_exact_match_scores_one():
    out = R.compute_score(solution_str="x <answer>Barack Obama</answer>", ground_truth=["Barack Obama"])
    assert out["score"] == 1.0 and out["em"] == 1.0 and out["has_answer"] == 1.0


def test_case_and_punct_insensitive_match():
    out = R.compute_score(solution_str="<answer>the U.S.A.</answer>", ground_truth=["USA"])
    assert out["em"] == 1.0


def test_multiple_golds_any_hit():
    out = R.compute_score(solution_str="<answer>Obama</answer>", ground_truth=["Barack Obama", "Obama"])
    assert out["em"] == 1.0


# --- ground_truth shapes -------------------------------------------------
def test_ground_truth_dict_target():
    out = R.compute_score(solution_str="<answer>Paris</answer>", ground_truth={"target": ["Paris"]})
    assert out["em"] == 1.0


def test_ground_truth_plain_string():
    out = R.compute_score(solution_str="<answer>Paris</answer>", ground_truth="Paris")
    assert out["em"] == 1.0


# --- format gate ---------------------------------------------------------
def test_no_answer_block_is_zero():
    out = R.compute_score(solution_str="I think it is Paris", ground_truth=["Paris"])
    assert out["score"] == 0.0 and out["has_answer"] == 0.0


def test_wrong_answer_is_zero():
    out = R.compute_score(solution_str="<answer>London</answer>", ground_truth=["Paris"])
    assert out["score"] == 0.0 and out["em"] == 0.0


# --- cover_em (subem) ----------------------------------------------------
def test_cover_em_substring_but_not_exact():
    # gold is a substring of a wordier prediction: em=0, cover_em=1
    out = R.compute_score(
        solution_str="<answer>The capital is Paris, France</answer>", ground_truth=["Paris"]
    )
    assert out["em"] == 0.0 and out["cover_em"] == 1.0


def test_metric_switch_to_cover_em_scores_one():
    out = _score("<answer>The capital is Paris, France</answer>", ["Paris"], QA_REWARD_METRIC="cover_em")
    assert out["score"] == 1.0


# --- f1 ------------------------------------------------------------------
def test_f1_partial_overlap():
    out = R.compute_score(solution_str="<answer>Barack Hussein Obama</answer>", ground_truth=["Barack Obama"])
    assert 0.0 < out["f1"] < 1.0 and out["em"] == 0.0


# --- retrieval-recall bonus (QA_RETRIEVAL_BONUS) -------------------------
_TR = "<tool_response>\n[0] Gong (band)  score=0.9\n    Miquette Giraudy performed with the band.\n</tool_response>"


def test_retrieval_bonus_off_by_default():
    # Gold IS in the tool response, but with the bonus off the reward is the old pure-EM 0.0.
    traj = _TR + "\n<answer>Melody Green</answer>"
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.0)
    assert out["score"] == 0.0 and out["gold_retrieved"] == 0.0


def test_retrieval_bonus_wrong_but_gold_surfaced():
    # Wrong final answer, but the search surfaced the gold -> small positive credit.
    traj = _TR + "\n<answer>Melody Green</answer>"
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.2)
    assert out["em"] == 0.0 and out["gold_retrieved"] == 1.0 and abs(out["score"] - 0.2) < 1e-9


def test_retrieval_bonus_not_farmable_via_reasoning():
    # Gold appears ONLY in the model's own reasoning; the tool_response does NOT contain it.
    # Scoping detection to <tool_response> spans must reject this -> no bonus (anti-gaming).
    traj = ("<tool_response>\n[0] Unrelated  score=0.7\n    Something else entirely.\n</tool_response>"
            "\nI'm pretty sure the answer is Miquette Giraudy.\n<answer>Melody Green</answer>")
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.2)
    assert out["gold_retrieved"] == 0.0 and out["score"] == 0.0


def test_retrieval_bonus_correct_answer_still_exactly_one():
    # A correct answer scores exactly 1.0 (not 1.0+bonus) -> reward stays bounded in [0,1].
    traj = _TR + "\n<answer>Miquette Giraudy</answer>"
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.2)
    assert out["em"] == 1.0 and out["score"] == 1.0 and out["gold_retrieved"] == 1.0


def test_retrieval_bonus_no_answer_but_surfaced():
    # No <answer> block, but the search surfaced the gold -> credit the search behaviour.
    traj = _TR + "\nlet me think about this some more"
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.2)
    assert out["has_answer"] == 0.0 and out["gold_retrieved"] == 1.0 and abs(out["score"] - 0.2) < 1e-9


def test_retrieval_bonus_fallback_when_no_toolresponse_markers():
    # No <tool_response> markers at all -> fallback scans the trajectory minus the <answer> block.
    traj = "the search results mention Miquette Giraudy as the spouse\n<answer>Melody Green</answer>"
    out = _score(traj, ["Miquette Giraudy"], QA_RETRIEVAL_BONUS=0.2)
    assert out["gold_retrieved"] == 1.0 and abs(out["score"] - 0.2) < 1e-9


def test_gold_in_retrieval_is_token_contiguous():
    # Short gold must not match inside a longer word; multi-token gold must be contiguous & in order.
    assert R.gold_in_retrieval("<tool_response>you must go now</tool_response>", ["us"]) is False
    assert R.gold_in_retrieval("<tool_response>born in New York City</tool_response>", ["new york"]) is True
    assert R.gold_in_retrieval("<tool_response>York and New things</tool_response>", ["new york"]) is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
