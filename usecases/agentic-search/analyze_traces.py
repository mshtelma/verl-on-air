#!/usr/bin/env python3
"""Tier-0 offline diagnostic for the MuSiQue agentic-search eval traces.

Runs PURELY on the saved trajectory traces (musique_*_traces.jsonl) -- no GPU, no
serving. For each dev question it decides whether the gold answer was ever *surfaced*
in a retrieved passage, then buckets every EM miss into retrieval- vs reasoning- vs
format-bound. That tells us which lever actually moves 56% -> 60-70%:

  * RETRIEVAL_MISS  -> the fact never entered context. Policy/reward training can't fix
                       it; only better retrieval (top_k, reranker, bigger/other corpus).
  * REASONING_MISS  -> the fact WAS retrieved but the model answered the wrong entity.
                       -> coverage/curriculum/SFT-warmstart levers.
  * FORMAT (cover_em==1, em==0) -> right answer, wrong string (over-complete / alias).
                       -> reward shaping / answer-format bonus / alias fixes = free points.

"Answer-string recall" = fraction of questions where any gold answer appears as a
contiguous normalized token-run in the concatenated tool outputs. It is the EM CEILING
given the current retrieval: EM can't exceed recall unless the model answers from
parametric memory.

Usage:
  python usecases/agentic-search/analyze_traces.py /tmp/musique_diag/musique_*_traces.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

# Reuse the EXACT normalizer the reward/eval use, so "retrieved_gold" is consistent with EM.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from reward import normalize_answer  # noqa: E402


def _toks(s: str) -> list[str]:
    return normalize_answer(s).split()


def _contiguous_contains(hay: list[str], needle: list[str]) -> bool:
    """Is `needle` a contiguous sublist of `hay`? (token-level, avoids short-substring FPs)."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return False
    first = needle[0]
    for i in range(n - m + 1):
        if hay[i] == first and hay[i:i + m] == needle:
            return True
    return False


def _retrieved_text(rec: dict) -> str:
    chunks = []
    for st in rec.get("trajectory", []):
        for tr in st.get("tool_results", []):
            r = tr.get("result", "")
            if isinstance(r, str) and not r.startswith("Error:"):
                chunks.append(r)
    return " ".join(chunks)


def _gold_retrieved(rec: dict, hay_toks: list[str]) -> bool:
    for g in rec.get("gt", []) or []:
        if _contiguous_contains(hay_toks, _toks(g)):
            return True
    return False


def _bucket(rec: dict, retrieved: bool) -> str:
    em = float(rec.get("em", 0) or 0)
    cover = float(rec.get("cover_em", 0) or 0)
    f1 = float(rec.get("f1", 0) or 0)
    if em >= 1.0:
        return "CORRECT"
    if cover >= 1.0:
        return "FORMAT"          # pred literally contains gold: over-complete / alias / string
    if f1 >= 0.5:
        return "PARTIAL"         # substantial overlap, softer near-miss
    return "REASONING" if retrieved else "RETRIEVAL"


def analyze(path: str) -> dict:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    n = len(recs)
    buckets: dict[str, int] = defaultdict(int)
    n_retr = n_em = n_cover = 0
    em_if_retr = tot_retr = em_if_not = tot_not = 0
    by_hop: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # hop -> [n, em, retrieved]
    no_answer = 0
    examples: dict[str, list] = defaultdict(list)
    for rec in recs:
        hay = _toks(_retrieved_text(rec))
        retrieved = _gold_retrieved(rec, hay)
        em = float(rec.get("em", 0) or 0)
        cover = float(rec.get("cover_em", 0) or 0)
        b = _bucket(rec, retrieved)
        buckets[b] += 1
        n_retr += int(retrieved)
        n_em += int(em >= 1.0)
        n_cover += int(cover >= 1.0)
        if rec.get("pred") in (None, "", "None"):
            no_answer += 1
        if retrieved:
            tot_retr += 1
            em_if_retr += int(em >= 1.0)
        else:
            tot_not += 1
            em_if_not += int(em >= 1.0)
        hop = str(rec.get("hop_type") or "?")
        by_hop[hop][0] += 1
        by_hop[hop][1] += int(em >= 1.0)
        by_hop[hop][2] += int(retrieved)
        if len(examples[b]) < 3:
            examples[b].append({"uid": rec.get("uid"), "q": rec.get("question"),
                                 "gt": rec.get("gt"), "pred": rec.get("pred")})
    return {
        "path": os.path.basename(path), "n": n,
        "em_pct": 100.0 * n_em / n, "cover_pct": 100.0 * n_cover / n,
        "recall_pct": 100.0 * n_retr / n, "no_answer": no_answer,
        "em_if_retrieved": (100.0 * em_if_retr / tot_retr) if tot_retr else 0.0,
        "em_if_not_retrieved": (100.0 * em_if_not / tot_not) if tot_not else 0.0,
        "tot_retr": tot_retr, "tot_not": tot_not,
        "buckets": dict(buckets), "by_hop": {k: v for k, v in by_hop.items()},
        "examples": dict(examples),
    }


def _label(path: str) -> str:
    b = os.path.basename(path)
    if "base" in b:
        return "base"
    for s in ("step10", "step20", "step30", "step40", "step50", "step60"):
        if s in b:
            return s
    return b


def main(paths: list[str]) -> None:
    res = {_label(p): analyze(p) for p in paths}
    order = [k for k in ["base", "step10", "step20", "step30", "step40", "step50", "step60"] if k in res]
    order += [k for k in res if k not in order]

    print("=" * 92)
    print("HEADLINE  (recall = answer-string ever surfaced in a retrieved passage = the EM ceiling)")
    print("=" * 92)
    print(f"{'run':>8} | {'EM%':>6} {'cover%':>7} | {'recall%':>8} | {'EM|retr%':>9} {'EM|¬retr%':>10} | {'no_ans':>6}")
    print("-" * 92)
    for k in order:
        r = res[k]
        print(f"{k:>8} | {r['em_pct']:6.1f} {r['cover_pct']:7.1f} | {r['recall_pct']:8.1f} | "
              f"{r['em_if_retrieved']:9.1f} {r['em_if_not_retrieved']:10.1f} | {r['no_answer']:6d}")

    print("\n" + "=" * 92)
    print("MISS TAXONOMY  (counts /200)")
    print("=" * 92)
    cats = ["CORRECT", "FORMAT", "PARTIAL", "REASONING", "RETRIEVAL"]
    print(f"{'run':>8} | " + " ".join(f"{c:>10}" for c in cats))
    print("-" * 92)
    for k in order:
        b = res[k]["buckets"]
        print(f"{k:>8} | " + " ".join(f"{b.get(c,0):>10}" for c in cats))
    print("\nlegend: FORMAT=right answer, wrong string (recoverable by reward/alias) | "
          "REASONING=fact retrieved, wrong entity | RETRIEVAL=fact never surfaced")

    # hop breakdown for base vs most-trained
    for k in ("base", order[-1]):
        r = res[k]
        print(f"\n--- {k}: by hop_type (n / EM% / recall%) ---")
        for hop in sorted(r["by_hop"]):
            n, em, rt = r["by_hop"][hop]
            print(f"  {hop:>6}: n={n:3d}  EM={100.0*em/n:5.1f}%  recall={100.0*rt/n:5.1f}%")

    # example misses for base
    print(f"\n--- base: example misses per bucket ---")
    for c in ["RETRIEVAL", "REASONING", "FORMAT", "PARTIAL"]:
        for ex in res["base"]["examples"].get(c, [])[:2]:
            print(f"  [{c}] Q: {ex['q']}")
            print(f"        gt={ex['gt']}  pred={ex['pred']!r}")

    with open("/tmp/musique_diag/diagnostic_summary.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n[wrote /tmp/musique_diag/diagnostic_summary.json]")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        import glob
        args = sorted(glob.glob("/tmp/musique_diag/musique_*_traces.jsonl"))
    main(args)
