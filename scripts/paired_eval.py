#!/usr/bin/env python3
"""Paired comparison of eval artifacts that scored the SAME questions: a baseline vs one or more
checkpoints.

    python3 scripts/paired_eval.py BASE.json CKPT1.json [CKPT2.json ...] [--json-out FILE]

Two marginal accuracies cannot tell you whether a delta is real: what matters is how many
questions CHANGED outcome, in which direction. Per checkpoint this reports the discordant pairs
(gained = base wrong -> checkpoint right, lost = the reverse), the exact two-sided McNemar p, and a
paired-bootstrap 95% CI of the accuracy delta, broken down by hop type / level when present.

Picking the best of K checkpoints on the same questions and then testing it as if it were the only
one inflates significance. So it also reports a SELECTION-AWARE p: a sign-flip permutation test of
max_k |gained_k - lost_k| -- per question, the base/checkpoint roles are swapped jointly for every
checkpoint, which is exact under the sharp null that no checkpoint differs from the base. With one
checkpoint it reduces to the exact two-sided McNemar test.

Every artifact must hold the same question ids (and question text, when recorded); otherwise the
comparison is refused. Artifacts written under engine/serve/eval_contract.py must also be `valid`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

KEY_FIELDS = ("uid", "idx")
GROUP_FIELDS = ("hop_type", "level")


def load(path: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    art = json.loads(raw)
    results = art.get("results") or []
    key = next((k for k in KEY_FIELDS if results and k in results[0]), None)
    if key is None:
        raise SystemExit(f"{path}: no per-question results with a {'/'.join(KEY_FIELDS)} field")
    if art.get("valid") is False:
        raise SystemExit(f"{path}: the eval marked itself INVALID ({art.get('invalid_reasons')}) -- not comparable")
    rows = {}
    for r in results:
        k = str(r[key])
        if k in rows:
            raise SystemExit(f"{path}: duplicate question id {k}")
        if r.get("status", "scored") != "scored":
            raise SystemExit(f"{path}: question {k} was not scored ({r.get('status')}) -- not comparable")
        rows[k] = r
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows, "key": key}


def exact_mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(d: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def max_stat_permutation_p(D: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """D: questions x checkpoints of (ckpt correct - base correct). Flip each question's sign
    jointly across checkpoints; p = P(max_k |sum_i s_i D_ik| >= observed)."""
    active = D[np.any(D != 0, axis=1)]
    if len(active) == 0:
        return 1.0
    observed = np.abs(active.sum(axis=0)).max()
    hits, done = 0, 0
    while done < n_perm:
        m = min(20000, n_perm - done)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(m, len(active)))
        hits += int((np.abs(signs @ active).max(axis=1) >= observed).sum())
        done += m
    return (hits + 1) / (n_perm + 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--json-out")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--permutations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="provenance to record in the output (where the artifacts came from)")
    args = ap.parse_args(argv)

    base = load(args.base)
    arts = [load(p) for p in args.checkpoints]
    ids = sorted(base["rows"])
    for a in arts:
        if sorted(a["rows"]) != ids:
            only_b = sorted(set(base["rows"]) - set(a["rows"]))[:5]
            only_a = sorted(set(a["rows"]) - set(base["rows"]))[:5]
            raise SystemExit(f"{a['path']}: different questions than the base "
                             f"(only in base {only_b}, only here {only_a}) -- not a paired comparison")
        for k in ids:
            qb, qa = base["rows"][k].get("question"), a["rows"][k].get("question")
            if qb is not None and qa is not None and qb != qa:
                raise SystemExit(f"{a['path']}: question {k} has different text than in the base")

    rng = np.random.default_rng(args.seed)
    b = np.array([bool(base["rows"][k]["correct"]) for k in ids], dtype=np.int8)
    group_field = next((f for f in GROUP_FIELDS if f in base["rows"][ids[0]]), None)
    groups = [str(base["rows"][k].get(group_field)) for k in ids] if group_field else None
    per, D = [], []
    for a in arts:
        c = np.array([bool(a["rows"][k]["correct"]) for k in ids], dtype=np.int8)
        d = c - b
        D.append(d)
        gained, lost = int((d == 1).sum()), int((d == -1).sum())
        lo, hi = bootstrap_ci(d.astype(float), args.bootstrap, rng)
        entry = {"artifact": a["path"], "sha256": a["sha256"], "correct": int(c.sum()), "n": len(ids),
                 "accuracy": float(c.mean()), "delta": float(d.mean()), "delta_ci95": [lo, hi],
                 "gained": gained, "lost": lost, "exact_mcnemar_p": exact_mcnemar(gained, lost)}
        if groups:
            entry["by_" + group_field] = {
                g: {"n": int(sum(1 for x in groups if x == g)),
                    "base_correct": int(sum(int(b[i]) for i, x in enumerate(groups) if x == g)),
                    "correct": int(sum(int(c[i]) for i, x in enumerate(groups) if x == g))}
                for g in sorted(set(groups))}
        per.append(entry)

    Dm = np.stack(D, axis=1)
    best = max(range(len(per)), key=lambda i: per[i]["gained"] - per[i]["lost"])
    out = {
        "note": args.note,
        "base": {"artifact": base["path"], "sha256": base["sha256"], "correct": int(b.sum()), "n": len(ids),
                 "accuracy": float(b.mean())},
        "question_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "checkpoints": per,
        "best_checkpoint": per[best]["artifact"],
        "selection_adjusted_p": max_stat_permutation_p(Dm, args.permutations, rng),
        "n_checkpoints_compared": len(per),
        "method": {"test": "exact two-sided McNemar per checkpoint; max-statistic sign-flip permutation "
                           "across all checkpoints for the selection-adjusted p",
                   "bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed},
    }

    print(f"base: {out['base']['correct']}/{len(ids)} ({out['base']['accuracy']:.1%})  {base['path']}")
    for e in per:
        print(f"  {e['correct']:>4}/{e['n']} ({e['accuracy']:.1%})  +{e['gained']:<3} -{e['lost']:<3} "
              f"McNemar p={e['exact_mcnemar_p']:.3f}  delta CI95 [{e['delta_ci95'][0]:+.3f}, "
              f"{e['delta_ci95'][1]:+.3f}]  {Path(e['artifact']).name}")
    print(f"best of {len(per)}: {Path(out['best_checkpoint']).name}; selection-adjusted "
          f"p={out['selection_adjusted_p']:.3f}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
