import copy

import pytest

from metrics import compare, pass_at_k, summarize


def test_pass_at_k_matches_miles_formula():
    assert pass_at_k(4, 1, 1) == 0.25
    assert pass_at_k(4, 1, 3) == 0.75
    assert pass_at_k(4, 2, 3) == 1
    assert pass_at_k(4, 0, 3) == 0
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)


def test_infrastructure_failures_are_excluded_from_pass_rates():
    rows = [{"group": "q", "status": "infra_harness", "reward": 0}]
    result = summarize(rows, 4)
    assert result["fully_scored_questions"] == 0 and result["pass_at_1"] is None


def evaluation(rewards):
    return {"valid": True, "eval_policy": {"samples_per_question": 4}, "question_ids_sha256": "fixed",
            "dataset": {"manifest": "fixed"}, "code": {"git_sha": "same"},
            "results": [{"id": f"{uid}/{sample}", "uid": uid, "sample": sample, "group": uid,
                         "split": "heldout", "status": "scored", "reward": value,
                         "metrics": {"answer_correct": value}}
                        for uid, values in rewards.items() for sample, value in enumerate(values)]}


def test_paired_interval_keeps_correlated_samples_in_their_question():
    before = evaluation({"q1": [0, 0, 0, 0], "q2": [1, 1, 1, 1]})
    after = evaluation({"q1": [1, 1, 1, 1], "q2": [0, 0, 0, 0]})
    paired = compare(before, after, bootstrap_samples=1000)
    result = paired["splits"]["heldout"]["metrics"]["pass_at_1"]
    assert result["delta"] == 0 and result["delta_ci95"] == [-1, 1]
    assert result["improved_questions"] == result["regressed_questions"] == 1
    assert paired["bootstrap_unit"] == "question"


@pytest.mark.parametrize("change", ["eval_policy", "question_ids_sha256", "dataset", "code", "missing", "infra"])
def test_pairing_rejects_changed_evaluation_contract(change):
    before = evaluation({"q1": [0, 1, 0, 0]})
    after = copy.deepcopy(before)
    if change in {"eval_policy", "dataset", "code"}:
        after[change]["changed"] = True
        if change == "code":
            after[change]["git_sha"] = "different"
    elif change == "question_ids_sha256":
        after[change] = "different"
    elif change == "missing":
        after["results"].pop()
    else:
        after["results"][0]["status"] = "infra_harness"
    with pytest.raises(ValueError):
        compare(before, after)
