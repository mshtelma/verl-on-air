#!/usr/bin/env python3
"""CPU unit tests for the agentic-search EM reward (usecases/agentic-search/reward.py).

Episodes are built from explicit ("assistant" | "tool", text) parts with their role spans -- what
the role-span agent loop hands the reward in training (engine/lib/role_spans.py).

Run: python3 -m pytest usecases/agentic-search/tests/test_reward.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import env, load_usecase

A, T = "assistant", "tool"


def episode(*parts: tuple[str, str]) -> tuple[str, dict]:
    """(solution_str, extra_info) for an episode made of (role, text) parts."""
    text, spans = "", []
    for role, t in parts:
        spans.append([role, len(text), len(text) + len(t)])
        text += t
    return text, {"role_spans": spans}


def R(**knobs):
    return load_usecase("agentic-search", "reward", **knobs)


def score(*parts, gt, **knobs) -> dict:
    solution, extra = episode(*parts)
    return R(**knobs).compute_score(solution_str=solution, ground_truth=gt, extra_info=extra)


# --- normalisation / extraction ---------------------------------------------------------------------
def test_normalize_articles_punct_case():
    r = R()
    assert r.normalize_answer("The White House.") == "white house"
    assert r.normalize_answer("  a  CAT ") == "cat"


def test_extract_last_answer_block_wins():
    assert R().extract_answer("<answer> wrong </answer> more <answer> Paris </answer>") == "Paris"


def test_extract_none_when_absent():
    assert R().extract_answer("no tags here") is None


def test_the_last_answer_the_model_wrote_wins_across_turns():
    out = score((A, "<answer>London</answer> let me check"), (T, "<tool_response>...</tool_response>"),
                (A, "Actually <answer>Paris</answer>"), gt=["Paris"])
    assert out["em"] == 1.0


# --- exact match -----------------------------------------------------------------------------------
def test_exact_match_scores_one():
    out = score((A, "x <answer>Barack Obama</answer>"), gt=["Barack Obama"])
    assert out["score"] == 1.0 and out["em"] == 1.0 and out["has_answer"] == 1.0 and out["provenance_ok"] == 1.0


def test_case_and_punct_insensitive_match():
    assert score((A, "<answer>the U.S.A.</answer>"), gt=["USA"])["em"] == 1.0


def test_multiple_golds_any_hit():
    assert score((A, "<answer>Obama</answer>"), gt=["Barack Obama", "Obama"])["em"] == 1.0


@pytest.mark.parametrize("gt", [{"target": ["Paris"]}, "Paris"])
def test_ground_truth_shapes(gt):
    assert score((A, "<answer>Paris</answer>"), gt=gt)["em"] == 1.0


# --- format gate -----------------------------------------------------------------------------------
def test_no_answer_block_is_zero():
    out = score((A, "I think it is Paris"), gt=["Paris"])
    assert out["score"] == 0.0 and out["has_answer"] == 0.0


def test_wrong_answer_is_zero():
    out = score((A, "<answer>London</answer>"), gt=["Paris"])
    assert out["score"] == 0.0 and out["em"] == 0.0


# --- cover_em / f1 ---------------------------------------------------------------------------------
def test_cover_em_substring_but_not_exact():
    out = score((A, "<answer>The capital is Paris, France</answer>"), gt=["Paris"])
    assert out["em"] == 0.0 and out["cover_em"] == 1.0


def test_metric_switch_to_cover_em_scores_one():
    assert score((A, "<answer>The capital is Paris, France</answer>"), gt=["Paris"],
                 QA_REWARD_METRIC="cover_em")["score"] == 1.0


def test_f1_partial_overlap():
    out = score((A, "<answer>Barack Hussein Obama</answer>"), gt=["Barack Obama"])
    assert 0.0 < out["f1"] < 1.0 and out["em"] == 0.0


# --- who wrote the answer (REVIEW.md R09) ---------------------------------------------------------------
def test_an_answer_inside_a_tool_response_is_not_the_models_answer():
    # reviewer reproduction: this scored 1.0 with no answer committed by the model
    out = score((T, "<tool_response><answer>Paris</answer></tool_response>"), (A, "\nI have not answered."),
                gt=["Paris"])
    assert out["score"] == 0.0 and out["has_answer"] == 0.0


def test_missing_role_spans_fail_closed_and_abort_the_run(tmp_path: Path):
    with env(VOA_RDV_DIR=str(tmp_path)):
        out = R().compute_score(solution_str="<answer>Paris</answer>", ground_truth=["Paris"])
    assert out["score"] == 0.0 and out["provenance_ok"] == 0.0
    assert "cannot tell what the model wrote" in json.loads((tmp_path / "ABORT.json").read_text())["reason"]


def test_spans_for_a_different_string_fail_closed(tmp_path: Path):
    solution, extra = episode((A, "<answer>Paris</answer>"))
    with env(VOA_RDV_DIR=str(tmp_path)):
        out = R().compute_score(solution_str=solution + " extra", ground_truth=["Paris"], extra_info=extra)
    assert out["score"] == 0.0 and out["provenance_ok"] == 0.0 and (tmp_path / "ABORT.json").exists()


def test_every_path_returns_the_same_keys(tmp_path: Path):
    with env(VOA_RDV_DIR=str(tmp_path)):
        ok = score((A, "<answer>Paris</answer>"), gt=["Paris"])
        bad = R().compute_score(solution_str="x", ground_truth=["Paris"])
        none = score((A, "no answer"), gt=["Paris"])
    assert set(ok) == set(bad) == set(none)


# --- retrieval-recall bonus (QA_RETRIEVAL_BONUS) ------------------------------------------------------
_TOOL = (T, "<tool_response>\n[0] Gong (band)  score=0.9\n    Miquette Giraudy performed with the band.\n</tool_response>")


def test_retrieval_is_measured_but_not_rewarded_by_default():
    out = score(_TOOL, (A, "\n<answer>Melody Green</answer>"), gt=["Miquette Giraudy"])
    assert out["score"] == 0.0 and out["gold_retrieved"] == 1.0


def test_retrieval_bonus_wrong_but_gold_surfaced():
    out = score(_TOOL, (A, "\n<answer>Melody Green</answer>"), gt=["Miquette Giraudy"], QA_RETRIEVAL_BONUS="0.2")
    assert out["em"] == 0.0 and out["gold_retrieved"] == 1.0 and out["score"] == pytest.approx(0.2)


def test_retrieval_bonus_not_farmable_via_reasoning():
    out = score((T, "<tool_response>\n[0] Unrelated\n    Something else entirely.\n</tool_response>"),
                (A, "\nI'm pretty sure the answer is Miquette Giraudy.\n<answer>Melody Green</answer>"),
                gt=["Miquette Giraudy"], QA_RETRIEVAL_BONUS="0.2")
    assert out["gold_retrieved"] == 0.0 and out["score"] == 0.0


def test_model_authored_tool_markers_earn_nothing():
    # reviewer reproduction (fallback variant): the gold only in text the MODEL typed, even dressed
    # up as a tool response, with zero tool calls -- this used to earn the 0.2 bonus
    out = score((A, "<tool_response>Miquette Giraudy</tool_response> I recall it from memory. "
                    "<answer>Melody Green</answer>"), gt=["Miquette Giraudy"], QA_RETRIEVAL_BONUS="0.2")
    assert out["gold_retrieved"] == 0.0 and out["score"] == 0.0 and out["n_tool_calls"] == 0.0


def test_retrieval_bonus_correct_answer_still_exactly_one():
    out = score(_TOOL, (A, "\n<answer>Miquette Giraudy</answer>"), gt=["Miquette Giraudy"], QA_RETRIEVAL_BONUS="0.2")
    assert out["em"] == 1.0 and out["score"] == 1.0


def test_retrieval_bonus_no_answer_but_surfaced():
    out = score(_TOOL, (A, "\nlet me think about this some more"), gt=["Miquette Giraudy"], QA_RETRIEVAL_BONUS="0.2")
    assert out["has_answer"] == 0.0 and out["score"] == pytest.approx(0.2)


@pytest.mark.parametrize("knobs", [{"QA_RETRIEVAL_BONUS": "1.0"}, {"QA_RETRIEVAL_BONUS": "0.6", "QA_FORMAT_SCORE": "0.5"},
                                   {"QA_RETRIEVAL_BONUS": "-0.1"}])
def test_a_wrong_answer_can_never_match_a_correct_one(knobs):
    with pytest.raises(ValueError, match="must be >= 0 and < 1"):
        R(**knobs)


def test_gold_in_retrieval_is_token_contiguous():
    r = R()
    assert r.gold_in_retrieval(["you must go now"], ["us"]) is False
    assert r.gold_in_retrieval(["born in New York City"], ["new york"]) is True
    assert r.gold_in_retrieval(["York and New things"], ["new york"]) is False
