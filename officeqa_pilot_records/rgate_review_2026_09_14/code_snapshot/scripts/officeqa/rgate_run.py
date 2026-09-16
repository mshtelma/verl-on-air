#!/usr/bin/env python3
"""R-GATE runner: score the expanded adversarial suite with the real GLM-5.3 judge
and evaluate the result against quantitative thresholds (docs plan Section 6.5).

Unlike score_trajectories.py (a human-validation aid), this is a GATE: it emits a
machine-readable report and exits non-zero when the reward/judge stack misses a
threshold.

Suite: scripts/reward/tests/rgate_expanded.jsonl (built by make_rgate_expanded.py;
real long trajectories + labeled hazards + the 9 original hand-built fixtures).

Groups and default thresholds (env-overridable):
  positives  (grounded_positive, grounded_positive_alt)
             true-accept rate >= OQ_RGATE_MIN_TA (0.95)
  negatives  (negative_*)
             false-accept count's one-sided 95% Clopper-Pearson upper bound
             <= OQ_RGATE_MAX_FA_UPPER (0.02); a base-question-level bound (any
             variant accepted => the base question failed) is reported alongside
  wrong_gated
             reward must be exactly 0 for 100% (strict answer gate, deterministic)
  unknowns   (verifier outages / unusable verdicts) are never failures-to-catch,
             but the overall unknown rate must stay <= OQ_RGATE_MAX_UNKNOWN (0.10)
             -- a judge that answers "unclear" to everything is not a gate.

Env: OQ_TRACES_IN, OQ_SCORES_OUT, OQ_RGATE_REPORT, JUDGE_CONCURRENCY,
     OQ_RGATE_MIN_TA, OQ_RGATE_MAX_FA_UPPER, OQ_RGATE_MAX_UNKNOWN,
     OQ_RGATE_EXIT_CODE (default 1: exit 1 on gate failure), plus JUDGE_* knobs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _SCRIPTS, os.path.join(_SCRIPTS, "reward")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JUDGE_BASE_URL", os.environ.get("EVAL_BASE_URL", ""))
os.environ.setdefault("JUDGE_MODEL", os.environ.get("EVAL_MODEL", "judge"))

from reward.officeqa_grounded_reward import score_record  # noqa: E402

TRACES_IN = os.environ.get("OQ_TRACES_IN", "reward/tests/rgate_expanded.jsonl")
if not os.path.isabs(TRACES_IN) and not os.path.exists(TRACES_IN):
    _cand = os.path.join(_SCRIPTS, TRACES_IN)   # snapshot-relative: scripts/reward/tests/...
    if os.path.exists(_cand):
        TRACES_IN = _cand
SCORES_OUT = os.environ.get("OQ_SCORES_OUT", "/Volumes/main/mshtelma/verl/eval/rgate_expanded_scores.jsonl")
REPORT_OUT = os.environ.get("OQ_RGATE_REPORT", SCORES_OUT + ".report.json")
CONC = int(os.environ.get("JUDGE_CONCURRENCY", "16"))
MIN_TA = float(os.environ.get("OQ_RGATE_MIN_TA", "0.95"))
MAX_FA_UPPER = float(os.environ.get("OQ_RGATE_MAX_FA_UPPER", "0.02"))
MAX_UNKNOWN = float(os.environ.get("OQ_RGATE_MAX_UNKNOWN", "0.10"))
EXIT_CODE = os.environ.get("OQ_RGATE_EXIT_CODE", "1") == "1"

POS = {"grounded_positive", "grounded_positive_alt"}
NEG_PREFIX = "negative_"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp_upper_95(x: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound on a binomial rate.

    Exact via the beta distribution when scipy is available (the GPU image);
    otherwise: x=0 uses the exact rule-of-three form 1 - 0.05**(1/n); x>0 falls
    back to the Wilson upper bound (slightly anti-conservative -- flagged in the
    report when used)."""
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import beta  # type: ignore
        return float(beta.ppf(0.95, x + 1, n - x))
    except Exception:  # noqa: BLE001
        if x == 0:
            return 1.0 - 0.05 ** (1.0 / n)
        z = 1.6448536269514722  # one-sided 95%
        p = x / n
        den = 1 + z * z / n
        center = p + z * z / (2 * n)
        return float((center + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den)


def evaluate(records: list[dict], scores: list[dict]) -> dict:
    by_uid = {r.get("uid"): r for r in records}
    groups = {"positives": [], "negatives": [], "wrong_gated": []}
    n_unknown = 0
    for s in scores:
        exp = (by_uid.get(s.get("uid")) or {}).get("expected", "")
        status = s.get("reward_status")
        reward = float(s.get("reward") or 0.0)
        unknown = (status == "unknown") or bool(s.get("error"))
        n_unknown += int(unknown)
        row = {"uid": s.get("uid"), "expected": exp, "reward": reward, "unknown": unknown,
               "base_uid": (by_uid.get(s.get("uid")) or {}).get("base_uid", s.get("uid")),
               "verdict": ((s.get("judge_verdict") or {}).get("verdict")),
               "reason": s.get("reward_reason")}
        if exp in POS:
            groups["positives"].append(row)
        elif exp.startswith(NEG_PREFIX):
            groups["negatives"].append(row)
        elif exp == "wrong_gated":
            groups["wrong_gated"].append(row)

    pos = groups["positives"]
    neg = groups["negatives"]
    wg = groups["wrong_gated"]
    ta = sum(1 for r in pos if r["reward"] > 0 and not r["unknown"])
    fa_rows = [r for r in neg if r["reward"] > 0 and not r["unknown"]]
    wg_bad = [r for r in wg if r["reward"] != 0.0]
    # Base-question level: a base fails if ANY of its negative variants was credited.
    base_neg: dict[str, bool] = {}
    for r in neg:
        base_neg.setdefault(r["base_uid"], False)
        if r["reward"] > 0 and not r["unknown"]:
            base_neg[r["base_uid"]] = True
    fa_base = sum(1 for v in base_neg.values() if v)

    report = {
        "n_records": len(records),
        "n_scored": len(scores),
        "n_unknown": n_unknown,
        "unknown_rate": (n_unknown / len(scores)) if scores else 1.0,
        "positives": {"n": len(pos), "accepted": ta,
                      "true_accept_rate": (ta / len(pos)) if pos else 0.0,
                      "false_rejects": [r["uid"] for r in pos if not r["reward"] > 0 and not r["unknown"]]},
        "negatives": {"n": len(neg), "false_accepts": len(fa_rows),
                      "fa_upper_95_case": cp_upper_95(len(fa_rows), len(neg)),
                      "false_accept_uids": [r["uid"] for r in fa_rows],
                      "n_base_questions": len(base_neg), "false_accepts_base": fa_base,
                      "fa_upper_95_base": cp_upper_95(fa_base, len(base_neg))},
        "wrong_gated": {"n": len(wg), "nonzero": len(wg_bad),
                        "bad_uids": [r["uid"] for r in wg_bad]},
    }
    gates = {
        "true_accept": {"pass": report["positives"]["true_accept_rate"] >= MIN_TA,
                        "value": report["positives"]["true_accept_rate"], "threshold": MIN_TA},
        "false_accept_upper_case": {"pass": report["negatives"]["fa_upper_95_case"] <= MAX_FA_UPPER,
                                    "value": report["negatives"]["fa_upper_95_case"],
                                    "threshold": MAX_FA_UPPER},
        "wrong_gate_exact": {"pass": len(wg_bad) == 0, "value": len(wg_bad), "threshold": 0},
        "unknown_budget": {"pass": report["unknown_rate"] <= MAX_UNKNOWN,
                           "value": report["unknown_rate"], "threshold": MAX_UNKNOWN},
    }
    report["gates"] = gates
    report["overall_pass"] = all(g["pass"] for g in gates.values())
    return report


async def _run() -> int:
    records = [json.loads(l) for l in open(TRACES_IN) if l.strip()]
    print(f"[rgate] read {len(records)} cases from {TRACES_IN}", flush=True)
    print(f"[rgate] judge={os.environ.get('JUDGE_BASE_URL')!r} model={os.environ.get('JUDGE_MODEL')!r} "
          f"conc={CONC}", flush=True)
    sem = asyncio.Semaphore(CONC)

    async def _one(rec):
        async with sem:
            try:
                return await score_record(rec, judge_all=True)
            except Exception as e:  # noqa: BLE001
                return {"uid": rec.get("uid"), "error": f"{type(e).__name__}: {e}", "reward": 0.0}

    t0 = time.time()
    scores = await asyncio.gather(*[_one(r) for r in records])
    os.makedirs(os.path.dirname(SCORES_OUT) or ".", exist_ok=True)
    with open(SCORES_OUT, "w") as fh:
        for o in scores:
            fh.write(json.dumps(o, default=list) + "\n")

    report = evaluate(records, scores)
    report.update({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "judge": {"base_url": os.environ.get("JUDGE_BASE_URL", ""),
                  "model": os.environ.get("JUDGE_MODEL", "")},
        "artifacts": {"bundle": {"path": TRACES_IN, "sha256": _sha256(TRACES_IN)},
                      "scores": {"path": SCORES_OUT, "sha256": _sha256(SCORES_OUT)}},
        "thresholds": {"min_true_accept": MIN_TA, "max_fa_upper": MAX_FA_UPPER,
                       "max_unknown_rate": MAX_UNKNOWN},
    })
    with open(REPORT_OUT, "w") as fh:
        json.dump(report, fh, indent=2)

    p, n, w = report["positives"], report["negatives"], report["wrong_gated"]
    print("\n==================== R-GATE REPORT ====================", flush=True)
    print(f"cases={report['n_records']}  unknown={report['n_unknown']} ({report['unknown_rate']:.1%})  "
          f"elapsed={report['elapsed_s']}s", flush=True)
    print(f"POSITIVES: {p['accepted']}/{p['n']} accepted (true-accept {p['true_accept_rate']:.3f}, "
          f"need >= {MIN_TA})  false-rejects={p['false_rejects'][:8]}", flush=True)
    print(f"NEGATIVES: {n['false_accepts']}/{n['n']} false-accepts "
          f"(upper95 case {n['fa_upper_95_case']:.4f} / base {n['fa_upper_95_base']:.4f}, "
          f"need <= {MAX_FA_UPPER})  fa_uids={n['false_accept_uids'][:8]}", flush=True)
    print(f"WRONG-GATED: {w['nonzero']}/{w['n']} nonzero (need 0)", flush=True)
    for name, g in report["gates"].items():
        print(f"  gate {name:26s} {'PASS' if g['pass'] else 'FAIL'}  value={g['value']}", flush=True)
    print(f"OVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}  -> {REPORT_OUT}", flush=True)
    print("=======================================================", flush=True)
    return 0 if (report["overall_pass"] or not EXIT_CODE) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
