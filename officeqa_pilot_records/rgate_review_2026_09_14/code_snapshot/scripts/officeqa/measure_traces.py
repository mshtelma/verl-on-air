#!/usr/bin/env python3
"""Honest measurement + scorer-drift reconciliation for an OfficeQA trace bundle.

Gate M (docs/officeqa_rl_plan.md Section 2.1, 7): a headline accuracy number is not a
faithful picture. This script computes, from a trace bundle JSONL, the metrics the plan
requires for EVERY arm (base / SFT / trained), so a "gain" cannot be a coverage or
scorer artifact:

  * COVERAGE: substantive answers vs explicit abstentions ("DATA NOT AVAILABLE") vs
    missing/empty finals -- per difficulty. Accuracy among attempted answers is a
    diagnostic, not the headline.
  * TERMINATION: how many trajectories hit the turn cap.
  * SCORER DRIFT: correctness under THREE scorers at exact tolerance --
      local  = scripts/reward/officeqa_reward.score_answer         (the legacy port)
      pinned = officeqa_reward_upstream_pinned.score_answer         (byte-pinned upstream)
      strict = strict_answer.strict_correct                        (the training gate)
    with the exact per-UID disagreements listed. Re-scoring both arms with the SAME
    pinned scorer is how you tell a model change from a scorer change (the plan's
    51->52 hard 13->14 example was scorer drift, not learning).
  * TOOL EFFICACY: grep calls vs grep calls that actually matched.

It emits a JSON manifest (input + scorer hashes + the full stat block) so the numbers
are reproducible and immutable. It reads only; it never edits the bundle or the scorers.

Usage:
    PYTHONPATH=scripts:scripts/reward python3 scripts/officeqa/measure_traces.py \
        officeqa_pilot_records/officeqa_traces.jsonl [--out <manifest.json>] [--max-turns 16]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "reward")):  # scripts/, scripts/reward/
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grounding                                             # noqa: E402
from strict_answer import strict_correct                     # noqa: E402
from officeqa_reward import score_answer as _score_local     # noqa: E402
from officeqa_reward_upstream_pinned import score_answer as _score_upstream  # noqa: E402

_PINNED_UPSTREAM_SHA = "0d91698c87df6d889339aac36f63ae0966607f169890b0bf8b472b26bfe8138f"
_ABSTAIN = ("data not available", "not available", "n/a", "na", "unknown",
            "cannot determine", "cannot be determined", "no data", "insufficient")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_pinned_scorer() -> str:
    p = os.path.join(_HERE, "..", "reward", "officeqa_reward_upstream_pinned.py")
    sha = _sha256(os.path.abspath(p))
    if sha != _PINNED_UPSTREAM_SHA:
        raise SystemExit(f"REFUSING TO RUN: pinned upstream scorer drifted\n"
                         f"  expected {_PINNED_UPSTREAM_SHA}\n  found    {sha}")
    return sha


def classify_coverage(pred: str) -> str:
    p = (pred or "").strip()
    if not p:
        return "missing"
    pl = p.lower().strip(". ").strip('"').strip("'")
    if pl in _ABSTAIN or pl.startswith("data not available") or pl.startswith("not available"):
        return "abstain"
    return "substantive"


def _ok(fn, gt: str, pred: str) -> bool:
    """Score at EXACT tolerance; empty/None pred or scorer error -> not correct."""
    if not pred:
        return False
    try:
        return fn(gt, pred, 0.0) > 0
    except Exception:  # noqa: BLE001
        return False


def grep_stats(trajectory) -> tuple[int, int]:
    """(#grep_documents calls, #grep results that actually matched a 'file:line:' hit)."""
    calls = hits = 0
    for st in trajectory or []:
        for c in (st.get("tool_calls") or []):
            if c.get("name") == "grep_documents":
                calls += 1
        for r in (st.get("tool_results") or []):
            if r.get("name") == "grep_documents" and grounding._GREP_HIT_RE.search(str(r.get("result") or "")):
                hits += 1
    return calls, hits


def _blank_bucket() -> dict:
    return {"n": 0, "substantive": 0, "abstain": 0, "missing": 0,
            "reached_cap": 0, "truncated": 0,   # DISTINCT signals: turns>=max vs the bundle's truncated flag
            "correct_local": 0, "correct_upstream": 0, "correct_strict": 0,
            "grep_calls": 0, "grep_hits": 0}


def measure(path: str, max_turns: int = 16) -> dict:
    _verify_pinned_scorer()
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    buckets: dict[str, dict] = {"full": _blank_bucket()}
    disagreements = []
    for r in records:
        diff = str(r.get("difficulty", "unknown")).lower()
        gt, pred = str(r.get("gt", "")), r.get("pred", "")
        cov = classify_coverage(pred)
        lo, uo, so = _ok(_score_local, gt, pred), _ok(_score_upstream, gt, pred), \
            (bool(pred) and strict_correct(gt, pred).valid)
        reached_cap = int(r.get("turns", 0) or 0) >= max_turns
        truncated = bool(r.get("truncated"))
        gcalls, ghits = grep_stats(r.get("trajectory"))

        for key in ("full", diff):
            b = buckets.setdefault(key, _blank_bucket())
            b["n"] += 1
            b[cov] += 1
            b["reached_cap"] += int(reached_cap)
            b["truncated"] += int(truncated)
            b["correct_local"] += int(lo)
            b["correct_upstream"] += int(uo)
            b["correct_strict"] += int(so)
            b["grep_calls"] += gcalls
            b["grep_hits"] += ghits

        if not (lo == uo == so):
            disagreements.append({"uid": r.get("uid"), "difficulty": diff, "gt": gt,
                                  "pred": pred, "local": lo, "upstream": uo, "strict": so})

    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bundle_path": os.path.abspath(path),
        "bundle_sha256": _sha256(path),
        "n_records": len(records),
        "max_turns": max_turns,
        "scorers": {
            "local": "scripts/reward/officeqa_reward.py::score_answer",
            "upstream_pinned": {"file": "scripts/reward/officeqa_reward_upstream_pinned.py",
                                "sha256": _PINNED_UPSTREAM_SHA},
            "strict": "scripts/reward/strict_answer.py::strict_correct",
        },
        "buckets": buckets,
        "scorer_disagreements": {"count": len(disagreements), "cases": disagreements},
    }


def _print_summary(m: dict) -> None:
    print(f"\nBundle: {m['bundle_path']}\n  sha256={m['bundle_sha256'][:16]}…  n={m['n_records']}  "
          f"max_turns={m['max_turns']}")
    order = [k for k in ("full", "hard", "easy") if k in m["buckets"]] + \
            [k for k in m["buckets"] if k not in ("full", "hard", "easy")]
    hdr = f"{'set':6s} {'n':>4} {'subst':>6} {'abst':>5} {'miss':>5} {'cap':>5} {'trunc':>5} " \
          f"{'local':>6} {'upstr':>6} {'strict':>6} {'grepHit%':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for k in order:
        b = m["buckets"][k]
        gh = (100.0 * b["grep_hits"] / b["grep_calls"]) if b["grep_calls"] else 0.0
        print(f"{k:6s} {b['n']:>4} {b['substantive']:>6} {b['abstain']:>5} {b['missing']:>5} "
              f"{b['reached_cap']:>5} {b['truncated']:>5} {b['correct_local']:>6} "
              f"{b['correct_upstream']:>6} {b['correct_strict']:>6} {gh:>7.1f}%")
    d = m["scorer_disagreements"]
    print(f"\nScorer disagreements (local vs upstream vs strict): {d['count']}")
    for c in d["cases"][:12]:
        print(f"  {c['uid']} [{c['difficulty']}] gt={c['gt']!r} pred={c['pred']!r:24.24} "
              f"-> local={int(c['local'])} upstream={int(c['upstream'])} strict={int(c['strict'])}")
    if d["count"] > 12:
        print(f"  … +{d['count'] - 12} more (see manifest)")
    print("\nNote: 'local'/'upstr'/'strict' are correct-counts at EXACT tolerance. Accuracy "
          "among substantive answers is a diagnostic; the headline must report coverage too.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-turns", type=int, default=16)
    args = ap.parse_args()
    m = measure(args.bundle, max_turns=args.max_turns)
    _print_summary(m)
    out = args.out or (args.bundle + ".measure.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nManifest written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
