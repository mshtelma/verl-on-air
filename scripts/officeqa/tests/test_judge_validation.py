#!/usr/bin/env python3
"""CPU tests for the judge-validation harness + fixtures. No model/network/GPU.

Run standalone:
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/officeqa/tests/test_judge_validation.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "..")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import path_report as pr                    # noqa: E402
import judge_validation as jv               # noqa: E402
import judge_validation_fixtures as jvf     # noqa: E402


def test_fixtures_are_wellformed():
    for fx in jvf.FIXTURES:
        report = pr.parse_report(fx["terminal_text"])
        assert report.ok, f"{fx['episode_id']} report not valid: {report.errors}"
        ledger = pr.EpisodeLedger.from_record(fx)
        ref = pr.check_references(report, ledger)
        assert ref.valid, f"{fx['episode_id']} semantic fixture must have resolvable refs: {ref.codes}"
        # a SEMANTIC fixture must PASS the deterministic value gate (its claimed values are present);
        # if a value is absent it is a deterministic case and belongs in DETERMINISTIC_FIXTURES.
        values = pr.check_claimed_values(report, ledger)
        assert values.valid, (f"{fx['episode_id']} semantic fixture must pass the value gate; a claimed "
                              f"value is absent from the trace -> move it to DETERMINISTIC_FIXTURES: {values.codes}")
        assert fx["expected_path_status"] in (pr.SUPPORTED, pr.UNSUPPORTED)


def test_deterministic_fixtures_caught_pre_judge():
    # each deterministic fixture must be rejected by SOME deterministic gate (references OR values)
    # before the judge is ever asked.
    for fx in jvf.DETERMINISTIC_FIXTURES:
        report = pr.parse_report(fx["terminal_text"])
        ledger = pr.EpisodeLedger.from_record(fx)
        ref = pr.check_references(report, ledger)
        values = pr.check_claimed_values(report, ledger)
        assert (not ref.valid) or (not values.valid), \
            f"{fx['episode_id']} deterministic fixture was NOT caught pre-judge (ref+value both valid)"


def test_fixture_family_coverage():
    fams = {f["family"] for f in jvf.all_fixtures()}   # families span semantic + deterministic
    # the failure modes that matter must all be represented
    for needed in ("correct_lookup", "correct_compute", "valid_alternative", "valid_shortcut",
                   "recovered_route", "sufficient_snippet",
                   "wrong_value_claimed_faithfulness", "wrong_value_claimed_material",
                   "wrong_row_near_label", "wrong_period", "wrong_column", "wrong_year",
                   "wrong_units", "unsupported_comparison", "compute_output_not_document", "wrong_source",
                   "incomplete_aggregate", "code_does_not_compute",
                   "value_not_in_bytes", "nonexistent_ref", "undelivered_observation", "impossible_chronology"):
        assert needed in fams, f"missing family {needed}"


def test_oracle_judge_passes_cleanly():
    fixtures = jvf.all_fixtures()
    _res, summary = jv.run(fixtures, jv._oracle_judge(fixtures), max_retries=1)
    assert summary["pass"] and summary["false_accepts"] == 0 and summary["false_rejects"] == 0
    assert summary["unknowns"] == 0 and summary["deterministic_ok"]


def test_naive_always_supported_flags_false_accepts():
    # a judge that rubber-stamps everything must be caught: every must-reject case is a false accept.
    n_reject = sum(1 for f in jvf.FIXTURES if f["expected_path_status"] == pr.UNSUPPORTED)
    _res, summary = jv.run(jvf.all_fixtures(),
                           lambda _r: '{"path_status":"supported","path_score":1.0,"issues":[]}')
    assert not summary["pass"]
    assert summary["false_accepts"] == n_reject


def test_naive_always_unsupported_flags_false_rejects():
    n_accept = sum(1 for f in jvf.FIXTURES if f["expected_path_status"] == pr.SUPPORTED)
    _res, summary = jv.run(jvf.all_fixtures(),
                           lambda _r: '{"path_status":"unsupported","path_score":0,"issues":["x"]}')
    assert not summary["pass"]
    assert summary["false_rejects"] == n_accept


def test_unknown_judge_fails_smoke():
    # an all-UNKNOWN judge is useless and must NOT read as PASS.
    _res, summary = jv.run(jvf.all_fixtures(), lambda _r: "garbage, no verdict")
    assert not summary["pass"] and summary["unknowns"] >= len(jvf.FIXTURES)


def test_judge_never_sees_labels_or_gold():
    # the request built for the judge must not carry the construction label / family / answer flag
    fx = jvf.FIXTURES[0]
    report = pr.parse_report(fx["terminal_text"])
    req = pr.build_support_request(fx["question"], fx["question_requirements"], report,
                                   pr.EpisodeLedger.from_record(fx))
    for banned in ("expected_path_status", "family", "rationale", "answer_correct"):
        assert banned not in req


def test_wrong_row_uses_real_adjacent_row():
    # the wrong_row fixture cites the real WPA Dec value (105), grounding the trap in real bytes
    fx = next(f for f in jvf.FIXTURES if f["episode_id"] == "wrong_row")
    assert "105" in fx["observations"][0]["delivered_text"]
    assert "Works Projects Administration" in fx["observations"][0]["delivered_text"]


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed" + ("" if not failed else f", {len(failed)} FAILED: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
