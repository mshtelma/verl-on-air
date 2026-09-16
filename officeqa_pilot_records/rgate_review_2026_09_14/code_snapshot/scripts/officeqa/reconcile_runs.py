#!/usr/bin/env python3
"""Reconcile two OfficeQA eval runs at the PER-UID level (Gate M; also the base->trained
pairing tool). Aggregate accuracy hides run-to-run churn: two temperature-0 runs can post
the same total while disagreeing on many individual questions (vLLM greedy is not
batch-deterministic). A credible base->RL delta therefore needs PAIRED per-UID stats, not
a difference of two headline numbers.

Accepts either shape:
  * air/71-style JSON: {"results": [{uid,pred,correct,gt,difficulty,scores}, ...]}
  * air/72-style JSONL bundle: one {uid,pred,correct,gt,difficulty,...} per line

Reports: per-UID prediction-identity and correctness agreement, each run's correct count
under the run-native scorer AND the byte-pinned upstream scorer (same scorer, both arms),
and the McNemar 2x2 table (b,c are the discordant pairs a paired test uses). Writes an
immutable manifest. Read-only.

    PYTHONPATH=scripts:scripts/reward python3 scripts/officeqa/reconcile_runs.py RUN_A RUN_B [--out m.json]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "reward")):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from officeqa_reward_upstream_pinned import score_answer as _pinned   # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(path: str) -> dict[str, dict]:
    """Load one eval artifact and VALIDATE it. Silent UID intersections/duplicates or a
    malformed one-row JSONL must not create a confidently wrong paired statistic."""
    txt = open(path).read()
    rows = []
    try:                                            # whole-file JSON (air/71-style)
        d = json.loads(txt)
        if isinstance(d, dict) and "results" in d:
            rows = d["results"] or []
        elif isinstance(d, dict) and "uid" in d:   # a ONE-record JSONL line is valid JSON
            rows = [d]
        elif isinstance(d, list):
            rows = d
        else:
            raise ValueError(f"{path}: whole-file JSON has no `results` rows and is not one result record")
    except json.JSONDecodeError:                    # JSONL bundle (air/72-style)
        try:
            rows = [json.loads(l) for l in txt.splitlines() if l.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: neither whole-file JSON nor valid JSONL: {e}") from e
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: no result rows found")
    out: dict[str, dict] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not r.get("uid"):
            raise ValueError(f"{path}: row {i} is not a dict with a uid")
        uid = str(r["uid"])
        if uid in out:
            raise ValueError(f"{path}: duplicate uid {uid!r}")
        out[uid] = r
    return out


def _pinned_ok(rec: dict) -> bool:
    p = rec.get("pred") or ""
    try:
        return bool(p) and _pinned(str(rec.get("gt", "")), p, 0.0) > 0
    except Exception:  # noqa: BLE001
        return False


def reconcile(path_a: str, path_b: str) -> dict:
    A, B = load_run(path_a), load_run(path_b)
    common = sorted(set(A) & set(B))
    if set(A) != set(B):
        missing_b = sorted(set(A) - set(B))[:20]
        missing_a = sorted(set(B) - set(A))[:20]
        raise ValueError(
            "runs do not cover the same UID set; refusing to silently intersect "
            f"({len(set(A) - set(B))} only in A, {len(set(B) - set(A))} only in B; "
            f"examples only_a={missing_b}, only_b={missing_a})")
    gold_mismatch = [u for u in common if str(A[u].get("gt", "")) != str(B[u].get("gt", ""))]
    diff_mismatch = [u for u in common if str(A[u].get("difficulty", "")) != str(B[u].get("difficulty", ""))]
    if gold_mismatch:
        raise ValueError(f"gold answers differ for {len(gold_mismatch)} UIDs; refusing to pair "
                         f"(examples {gold_mismatch[:20]})")
    if diff_mismatch:
        raise ValueError(f"difficulty labels differ for {len(diff_mismatch)} UIDs; refusing to pair "
                         f"(examples {diff_mismatch[:20]})")
    pred_same = corr_same = 0
    diffs = []
    a = b = c = d = 0            # McNemar on PINNED scorer: a=both ok, b=A ok only, c=B ok only, d=both wrong
    for u in common:
        pa, pb = (A[u].get("pred") or "").strip(), (B[u].get("pred") or "").strip()
        pred_same += pa == pb
        oa, ob = _pinned_ok(A[u]), _pinned_ok(B[u])
        corr_same += (oa == ob)
        a, b, c, d = a + (oa and ob), b + (oa and not ob), c + (ob and not oa), d + (not oa and not ob)
        if pa != pb:
            diffs.append({"uid": u, "difficulty": A[u].get("difficulty"),
                          "pred_a": A[u].get("pred"), "pred_b": B[u].get("pred"),
                          "pinned_a": oa, "pinned_b": ob})
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_a": {"path": os.path.abspath(path_a), "sha256": _sha256(path_a), "n": len(A)},
        "run_b": {"path": os.path.abspath(path_b), "sha256": _sha256(path_b), "n": len(B)},
        "common_uids": len(common),
        "pred_identical": pred_same,
        "correct_agrees_pinned": corr_same,
        "prediction_divergences": len(diffs),
        "pinned_correct": {"run_a": a + b, "run_b": a + c},
        "mcnemar": {"both_correct": a, "a_only": b, "b_only": c, "both_wrong": d,
                    "discordant": b + c,
                    "note": "b,c are the discordant pairs; a paired McNemar test uses these, "
                            "not the difference of the two totals."},
        "divergence_cases": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("--out", default="")
    args = ap.parse_args()
    m = reconcile(args.run_a, args.run_b)
    print(f"common={m['common_uids']}  pred_identical={m['pred_identical']}  "
          f"correct_agrees={m['correct_agrees_pinned']}")
    print(f"pinned correct: A={m['pinned_correct']['run_a']}  B={m['pinned_correct']['run_b']}")
    mc = m["mcnemar"]
    print(f"McNemar: both_ok={mc['both_correct']} A_only={mc['a_only']} B_only={mc['b_only']} "
          f"both_wrong={mc['both_wrong']}  DISCORDANT={mc['discordant']}")
    print(f"prediction divergences: {m['prediction_divergences']}/{m['common_uids']} "
          f"({100.0*m['prediction_divergences']/max(1,m['common_uids']):.0f}% of preds changed between runs)")
    out = args.out or (args.run_a + ".reconcile.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
