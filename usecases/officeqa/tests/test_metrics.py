import pytest

from metrics import pass_at_k, summarize


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
