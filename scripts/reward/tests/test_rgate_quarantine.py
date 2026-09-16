#!/usr/bin/env python3
"""Contract tests for the Gate-R machinery added 2026-09-13:

  * reward/quarantine.py            -- whole-group unknown quarantine (pure core)
  * _site/sitecustomize.py          -- GRPO advantage-estimator patch (duck-typed)
  * officeqa/make_rgate_expanded.py -- real-trace mutation generator invariants
  * officeqa/rgate_run.py           -- gate statistics (Clopper-Pearson) + evaluation
  * judge_prompt alt-source clause  -- the RG07 policy decision

Run standalone (no pytest, no GPU, no judge):
    PYTHONPATH=scripts:scripts/reward:scripts/officeqa python3 scripts/reward/tests/test_rgate_quarantine.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, ".."),
           os.path.join(_HERE, "..", "..", "officeqa")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grounding                                    # noqa: E402
import judge_prompt                                 # noqa: E402
import officeqa_grounded_reward as rw                # noqa: E402
import quarantine                                   # noqa: E402
import make_rgate_expanded as gx                     # noqa: E402
import rgate_run                                    # noqa: E402


def _load_sitecustomize():
    path = os.path.join(_HERE, "..", "..", "_site", "sitecustomize.py")
    spec = importlib.util.spec_from_file_location("oq_sitecustomize_test", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- quarantine
def test_quarantine_row_mask_whole_group():
    rewards = [1.0, 0.0, 0.5, quarantine.OQ_UNKNOWN_SENTINEL, 0.25, quarantine.OQ_UNKNOWN_SENTINEL]
    index = ["a", "a", "b", "b", "c", "c"]
    mask = quarantine.quarantine_row_mask(rewards, index)
    # groups b and c each carry a sentinel -> whole groups quarantined; a untouched
    assert mask == [False, False, True, True, True, True]


def test_quarantine_no_sentinel_is_noop():
    rewards = [0.0, 1.0, 0.5]
    index = ["a", "a", "b"]
    assert not any(quarantine.quarantine_row_mask(rewards, index))
    adv = [[r] * 4 for r in rewards]
    a2, r2, n = quarantine.apply_quarantine(adv, adv, rewards, index)
    assert a2 == adv and r2 == adv and n == 0


def test_quarantine_all_sentinel_group_zeroes_everything():
    s = quarantine.OQ_UNKNOWN_SENTINEL
    adv, ret, n = quarantine.apply_quarantine([[1.0], [2.0]], [[1.0], [2.0]], [s, s], ["g", "g"])
    assert adv == [[0.0], [0.0]] and ret == [[0.0], [0.0]] and n == 1


def test_unknown_score_env_toggle():
    assert quarantine.unknown_score({}) == 0.0
    assert quarantine.unknown_score({"OQ_REWARD_QUARANTINE": "1"}) == quarantine.OQ_UNKNOWN_SENTINEL


def test_reward_unknown_emits_sentinel_under_quarantine():
    steps = [{"tool_calls": [{"name": "grep_documents", "args": {"pattern": "x", "file_name": "treasury_bulletin_1941_01.txt"}}],
              "tool_results": [{"name": "grep_documents", "result": "treasury_bulletin_1941_01.txt:142: National defense 2,602"}]}]
    report = grounding.grounding_report(steps, {"source_files": "treasury_bulletin_1941_01.txt"})
    old = os.environ.get("OQ_REWARD_QUARANTINE")
    try:
        os.environ["OQ_REWARD_QUARANTINE"] = "1"
        out = rw._assemble(True, report, None)      # judge unavailable -> unknown
        assert out["status"] == "unknown" and out["score"] == quarantine.OQ_UNKNOWN_SENTINEL, out
        assert out["verifier_ok"] == 0.0
    finally:
        if old is None:
            os.environ.pop("OQ_REWARD_QUARANTINE", None)
        else:
            os.environ["OQ_REWARD_QUARANTINE"] = old
    out = rw._assemble(True, report, None)
    assert out["score"] == 0.0                        # legacy offline behavior preserved


# --------------------------------------------------------------- sitecustomize
def _fake_grpo(token_level_rewards, response_mask, index, **kw):
    scores = [sum(r) for r in token_level_rewards]
    groups = {}
    for s, i in zip(scores, index):
        groups.setdefault(i, []).append(s)
    adv = []
    for s, i in zip(scores, index):
        m = sum(groups[i]) / len(groups[i])
        adv.append([s - m] * len(token_level_rewards[0]))
    return adv, [row[:] for row in adv]


def test_sitecustomize_wrapper_zeroes_sentinel_groups_on_lists():
    sc = _load_sitecustomize()
    wrapped = sc.make_quarantine_wrapper(_fake_grpo, quarantine)
    s = quarantine.OQ_UNKNOWN_SENTINEL
    rewards = [[1.0], [0.0], [0.6], [s], [0.2]]
    index = ["a", "a", "b", "b", "c"]
    adv, ret = wrapped(rewards, [[1]] * 5, index)
    # group a: mean 0.5 -> adv [0.5, -0.5]; group b zeroed (sentinel); c singleton -> 0
    assert adv[0] == [0.5] and adv[1] == [-0.5]
    assert adv[2] == [0.0] and adv[3] == [0.0] and ret[2] == [0.0]
    assert adv[4] == [0.0]


def test_sitecustomize_import_hook_patches_fake_verl(tmp_path=None):
    import pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    pkg = tmp / "verl" / "trainer" / "ppo"
    pkg.mkdir(parents=True)
    for d in ("verl", "verl/trainer", "verl/trainer/ppo"):
        (tmp / d / "__init__.py").write_text("")
    (pkg / "core_algos.py").write_text(
        "ADV_ESTIMATOR_REGISTRY = {}\n"
        "def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, **kw):\n"
        "    ADV_ESTIMATOR_REGISTRY.setdefault('grpo', compute_grpo_outcome_advantage)\n"
        "    scores = [sum(r) for r in token_level_rewards]\n"
        "    return [[s] for s in scores], [[s] for s in scores]\n"
        "ADV_ESTIMATOR_REGISTRY['grpo'] = compute_grpo_outcome_advantage\n")
    old_env = os.environ.get("OQ_REWARD_QUARANTINE")
    os.environ["OQ_REWARD_QUARANTINE"] = "1"
    sc = _load_sitecustomize()                      # installs the meta-path hook
    sys.path.insert(0, str(tmp))
    try:
        import verl.trainer.ppo.core_algos as ca    # hook must fire HERE
        s = quarantine.OQ_UNKNOWN_SENTINEL
        adv, _ = ca.ADV_ESTIMATOR_REGISTRY["grpo"]([[0.5], [s]], [[1], [1]], ["g", "g"])
        assert adv == [[0.0], [0.0]], adv
        adv2, _ = ca.compute_grpo_outcome_advantage([[0.5], [0.7]], [[1], [1]], ["g", "g"])
        assert adv2 == [[0.5], [0.7]]               # untouched groups pass through
        assert ca.ADV_ESTIMATOR_REGISTRY["grpo"].__name__.startswith("quarantined_")
    finally:
        sys.path.remove(str(tmp))
        for m in [m for m in sys.modules if m == "verl" or m.startswith("verl.")]:
            sys.modules.pop(m, None)
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, sc._CoreAlgosHook)]
        if old_env is None:
            os.environ.pop("OQ_REWARD_QUARANTINE", None)
        else:
            os.environ["OQ_REWARD_QUARANTINE"] = old_env


def test_sitecustomize_inactive_without_env():
    os.environ.pop("OQ_REWARD_QUARANTINE", None)
    sc = _load_sitecustomize()
    assert not any(isinstance(f, sc._CoreAlgosHook) for f in sys.meta_path)


# -------------------------------------------------------------------- generator
def _mini_trace(uid, gt, pred, qyear="1945"):
    return {
        "uid": uid, "difficulty": "easy", "gt": gt, "pred": pred,
        "question": f"What were total expenditures in FY {qyear}?",
        "source_files": "treasury_bulletin_1945_09.txt",
        "trajectory": [
            {"turn": 0, "reasoning": "Searching.",
             "tool_calls": [{"name": "grep_documents", "args": {"pattern": "expenditures", "file_name": "treasury_bulletin_1945_09.txt"}}],
             "tool_results": [{"name": "grep_documents",
                               "result": "treasury_bulletin_1945_09.txt:266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |"}]},
            {"turn": 1, "reasoning": "Reading the table.",
             "tool_calls": [{"name": "read_document", "args": {"file_name": "treasury_bulletin_1945_09.txt", "start_line": 260}}],
             "tool_results": [{"name": "read_document",
                               "result": "266: | 1945 | 2,681 | 341 | 480 | 507 | 154 |\n267: | 1946 | 9,999 | 1 | 2 | 3 | 4 |"}]},
            {"turn": 2, "reasoning": f"The answer is {pred}.\n<FINAL_ANSWER>{pred}</FINAL_ANSWER>",
             "tool_calls": [], "tool_results": []},
        ],
    }


def test_generator_period_mutation_shifts_year():
    t = _mini_trace("T001", "507", "507")
    rec, exp, note = gx.mut_Pp(copy.deepcopy(t), "507")
    assert rec is not None and exp == "negative_wrong_period"
    out = "\n".join(str(tr.get("result")) for st in rec["trajectory"] for tr in st.get("tool_results") or [])
    assert "| 1946 | 2,681 | 341 | 480 | 507 |" in out and "| 1945 | 2,681 |" not in out


def test_generator_unsupported_blanks_gold_everywhere():
    t = _mini_trace("T001", "507", "507")
    rec, exp, _ = gx.mut_U(copy.deepcopy(t), "507")
    out = "\n".join(str(tr.get("result")) for st in rec["trajectory"] for tr in st.get("tool_results") or [])
    assert "507" not in out and "no matches found" in out and exp == "negative_unsupported"


def test_generator_alt_source_renames_file_and_dates():
    t = _mini_trace("T001", "507", "507")
    rec, exp, note = gx.mut_A(copy.deepcopy(t))
    assert rec is not None and exp == "grounded_positive_alt"
    txt = json.dumps(rec["trajectory"])
    assert "treasury_bulletin_1945_09.txt" not in txt
    import re
    m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})\.txt", txt)
    assert m and (m.group(1), m.group(2)) != ("1945", "09")
    # listing date strings must be consistent with the new filename (no stale dates)
    assert "(1945-09)" not in txt


def test_generator_anachronistic_source_predates_asked_period():
    t = _mini_trace("T001", "507", "507", qyear="1945")
    rec, exp, _ = gx.mut_D(copy.deepcopy(t), "507")
    assert rec is not None and exp == "negative_impossible_source"
    assert "treasury_bulletin_1944_01.txt" in json.dumps(rec["trajectory"])


def test_generator_transplant_rewrites_commitment_and_strips_gold():
    t = _mini_trace("T002", "100", "120")
    cases = gx.make_cases(t, "wrong")
    assert len(cases) == 1 and cases[0]["expected"] == "negative_transplanted"
    c = cases[0]
    assert c["pred"] == "100" and "<FINAL_ANSWER>100</FINAL_ANSWER>" in c["trajectory"][-1]["reasoning"]


def test_generator_source_traces_untouched():
    t = _mini_trace("T001", "507", "507")
    before = json.dumps(t, sort_keys=True)
    gx.make_cases(t)
    assert json.dumps(t, sort_keys=True) == before


def test_generator_value_regex_no_substring_false_positives():
    assert gx._value_re("6").search("| Total | 6 | 1963 |")
    assert not gx._value_re("6").search("score=6.82")
    assert not gx._value_re("6").search("1963-06")
    assert not gx._value_re("3%").search("| 73% |")
    assert gx._value_re("2,602").search("| 2,602 |")
    assert not gx._value_re("2,602").search("| 12,602 |")


# ------------------------------------------------------------------ gate math
def test_cp_upper_rule_of_three():
    v = rgate_run.cp_upper_95(0, 150)
    assert abs(v - 0.01977) < 0.0005, v          # 0/150 -> ~2% (the Gate-R sizing)
    assert rgate_run.cp_upper_95(0, 220) < 0.02  # expanded-suite sizing
    assert rgate_run.cp_upper_95(2, 150) > 0.02  # 2 misses no longer clears 2%


def test_evaluate_gates_pass_and_fail():
    records = [
        {"uid": "p1", "expected": "grounded_positive", "base_uid": "p1"},
        {"uid": "p2", "expected": "grounded_positive_alt", "base_uid": "p2"},
        {"uid": "n1", "expected": "negative_unsupported", "base_uid": "b1"},
        {"uid": "n2", "expected": "negative_transplanted", "base_uid": "b2"},
        {"uid": "w1", "expected": "wrong_gated", "base_uid": "w1"},
    ]
    scores = [
        {"uid": "p1", "reward": 1.0, "reward_status": "scored"},
        {"uid": "p2", "reward": 0.925, "reward_status": "scored"},
        {"uid": "n1", "reward": 0.0, "reward_status": "scored"},
        {"uid": "n2", "reward": 0.0, "reward_status": "unknown"},   # safe: quarantined
        {"uid": "w1", "reward": 0.0, "reward_status": "scored"},
    ]
    old = (rgate_run.MIN_TA, rgate_run.MAX_FA_UPPER, rgate_run.MAX_UNKNOWN)
    rgate_run.MIN_TA, rgate_run.MAX_FA_UPPER, rgate_run.MAX_UNKNOWN = 0.95, 0.99, 0.5
    try:
        rep = rgate_run.evaluate(records, scores)
        assert rep["overall_pass"], rep["gates"]
        assert rep["positives"]["true_accept_rate"] == 1.0
        assert rep["negatives"]["false_accepts"] == 0
        # a false accept must surface in both case- and base-level accounting
        scores[2] = {"uid": "n1", "reward": 0.95, "reward_status": "scored"}
        rep = rgate_run.evaluate(records, scores)
        assert rep["negatives"]["false_accepts"] == 1
        assert rep["negatives"]["false_accepts_base"] == 1
        # wrong-gate nonzero must fail the exact gate
        scores[4] = {"uid": "w1", "reward": 0.2, "reward_status": "scored"}
        rep = rgate_run.evaluate(records, scores)
        assert not rep["gates"]["wrong_gate_exact"]["pass"] and not rep["overall_pass"]
    finally:
        rgate_run.MIN_TA, rgate_run.MAX_FA_UPPER, rgate_run.MAX_UNKNOWN = old


# ---------------------------------------------------------------- prompt policy
def test_judge_prompt_has_alt_source_clause():
    assert "EQUALLY VALID alternative source" in judge_prompt.JUDGE_SYSTEM
    assert "VALID" in judge_prompt.JUDGE_SYSTEM and "ALTERNATIVE FILE" in judge_prompt.JUDGE_SYSTEM


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
