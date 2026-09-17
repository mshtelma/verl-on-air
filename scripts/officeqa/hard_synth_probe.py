#!/usr/bin/env python3
"""Difficulty-probe tooling for the synthetic hard-question pilot.

Two subcommands:
  to-csv   : expand hard_synth_pilot_*.jsonl into the eval CSV the collector reads
             (uid, question, answer, source_docs, source_files, difficulty), duplicating each
             question N times (uid '<base>_sK') so a temp>0 collect yields N samples/question.
             Also writes an answer-key JSON {uid: answer_float} and a base map.
  score    : read the collect captures.jsonl, extract each episode's submitted numeric answer,
             compare to the key (relative+abs tolerance), and report PER-QUESTION pass-rate and
             the in-band bucketing (too-easy / learnable / too-hard) for GRPO.

The difficulty signal is deterministic ANSWER-CORRECTNESS pass-rate (can the base produce the
right number) -- no GPU judge needed for triage. Support/faithfulness grading is the training
reward's job, not the difficulty filter's.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re


def _answer_str(a) -> str:
    a = float(a)
    return str(int(a)) if abs(a - round(a)) < 1e-9 else f"{a:.2f}"


def to_csv(jsonl: str, out_csv: str, key_json: str, n: int) -> None:
    rows = [json.loads(l) for l in open(jsonl)]
    key, base = {}, {}
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "question", "answer", "source_docs", "source_files", "difficulty"])
        for idx, r in enumerate(rows):
            buid = f"HS{idx:04d}"
            ans = _answer_str(r["answer"])
            srcs = ";".join(r.get("source_files", []))
            base[buid] = {"answer": float(r["answer"]), "template": r["template"],
                          "question": r["question"], "n": n}
            for k in range(n):
                uid = f"{buid}_s{k}"
                key[uid] = float(r["answer"])
                w.writerow([uid, r["question"], ans, "", srcs, "hard"])
    json.dump({"key": key, "base": base}, open(key_json, "w"))
    print(f"wrote {len(rows)*n} rows ({len(rows)} questions x {n} samples) -> {out_csv}")
    print(f"wrote key ({len(key)} uids) -> {key_json}")


_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _extract_number(text: str):
    """Pull the final numeric value from a submitted answer string ('$44,463 million' -> 44463)."""
    if not text:
        return None
    t = text.replace(",", "").replace("$", "").replace("%", "")
    nums = _NUM.findall(t)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def _submitted_answer(rec: dict):
    """Extract the agent's answer field from a capture record's terminal_text JSON."""
    tt = rec.get("terminal_text") or ""
    try:
        obj = json.loads(tt)
        if isinstance(obj, dict) and "answer" in obj:
            return obj["answer"]
    except Exception:
        pass
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', tt)
    return m.group(1) if m else tt


def _correct(pred, gold: float, rel: float, abs_: float) -> bool:
    v = _extract_number(pred if isinstance(pred, str) else str(pred))
    if v is None:
        return False
    return abs(v - gold) <= max(abs_, rel * abs(gold))


def _read_parts(run_dir: str) -> list[dict]:
    """Read the collector's per-episode parts/*.json (each an independent, closed capture)."""
    import glob
    recs = []
    for fp in sorted(glob.glob(os.path.join(run_dir, "parts", "*.json"))):
        try:
            recs.append(json.load(open(fp)))
        except Exception:
            pass
    return recs


def _load_records(captures: str) -> list[dict]:
    """Read capture records from a captures.jsonl file OR a run directory.

    A directory resolves to its per-episode parts/*.json when present (the collector closes one
    file per episode as it completes, so a killed / timed-out run is still fully scorable from
    whatever finished); otherwise to the consolidated captures.jsonl written on a clean finish."""
    if os.path.isdir(captures):
        parts = _read_parts(captures)
        if parts:
            return parts
        jl = os.path.join(captures, "captures.jsonl")
        return [json.loads(l) for l in open(jl) if l.strip()] if os.path.isfile(jl) else []
    return [json.loads(l) for l in open(captures) if l.strip()]


def score(captures: str, key_json: str, rel: float, abs_: float, learn_lo: float,
          learn_hi: float) -> None:
    meta = json.load(open(key_json))
    key, base = meta["key"], meta["base"]
    records = _load_records(captures)
    print(f"loaded {len(records)} capture records from {captures}")
    per_q: dict[str, list] = {}
    for rec in records:
        uid = rec.get("episode_id") or rec.get("uid") or ""
        if uid not in key:
            continue
        buid = uid.rsplit("_s", 1)[0]
        per_q.setdefault(buid, []).append(_correct(_submitted_answer(rec), key[uid], rel, abs_))
    # bucket
    buckets = {"too_easy": [], "learnable": [], "too_hard": [], "no_data": []}
    rates = {}
    for buid, b in base.items():
        res = per_q.get(buid, [])
        if not res:
            buckets["no_data"].append(buid)
            continue
        p = sum(res) / len(res)
        rates[buid] = (p, len(res), b["template"])
        if p >= learn_hi:
            buckets["too_easy"].append(buid)
        elif p <= learn_lo:
            buckets["too_hard"].append(buid)
        else:
            buckets["learnable"].append(buid)
    n_scored = len(rates)
    print(f"questions with samples: {n_scored}/{len(base)}")
    for name in ("too_easy", "learnable", "too_hard", "no_data"):
        print(f"  {name:10s}: {len(buckets[name])}")
    # per-template learnable yield
    from collections import Counter, defaultdict
    by_t = defaultdict(lambda: [0, 0])
    for buid, (p, n, t) in rates.items():
        by_t[t][1] += 1
        if learn_lo < p < learn_hi:
            by_t[t][0] += 1
    print("  learnable / scored by template:")
    for t, (lk, tot) in sorted(by_t.items()):
        print(f"    {t:20s}: {lk}/{tot}")
    if rates:
        mean_p = sum(p for p, _, _ in rates.values()) / len(rates)
        print(f"  mean pass-rate: {mean_p:.3f}")
    # write learnable set (into the run dir if captures is a dir, else alongside the jsonl)
    out_base = captures if os.path.isdir(captures) else os.path.dirname(captures)
    out = os.path.join(out_base, "learnable_uids.json")
    json.dump({"learnable": buckets["learnable"], "rates": rates}, open(out, "w"))
    print(f"wrote learnable set -> {out}")


def aggregate(run_dir: str) -> None:
    """Consolidate the collector's additive per-episode parts/*.json into the canonical
    captures.jsonl. Safe + idempotent on a killed/timed-out run (or a clean one)."""
    recs = _read_parts(run_dir)
    out = os.path.join(run_dir, "captures.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"aggregated {len(recs)} episodes from parts/ -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("to-csv")
    c.add_argument("--jsonl", required=True)
    c.add_argument("--out-csv", required=True)
    c.add_argument("--key-json", required=True)
    c.add_argument("-n", "--num-samples", type=int, default=8)
    g = sub.add_parser("aggregate", help="glob parts/*.json in a run dir -> captures.jsonl")
    g.add_argument("--run-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--captures", required=True)
    s.add_argument("--key-json", required=True)
    s.add_argument("--rel-tol", type=float, default=0.001)
    s.add_argument("--abs-tol", type=float, default=0.5)
    s.add_argument("--learn-lo", type=float, default=0.05, help="p<=lo -> too hard")
    s.add_argument("--learn-hi", type=float, default=0.95, help="p>=hi -> too easy")
    a = ap.parse_args()
    if a.cmd == "to-csv":
        to_csv(a.jsonl, a.out_csv, a.key_json, a.num_samples)
    elif a.cmd == "aggregate":
        aggregate(a.run_dir)
    else:
        score(a.captures, a.key_json, a.rel_tol, a.abs_tol, a.learn_lo, a.learn_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
