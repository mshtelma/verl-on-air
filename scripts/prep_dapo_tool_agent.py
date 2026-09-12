#!/usr/bin/env python3
"""Prepare DAPO-Math-17k as a HARDER tool-agent training set for verl.

WHY: GSM8K saturated (~95-100%); then MATH L3-5 ALSO turned out saturated for this
model (base ~95% correct-among-answered, agentic, run 408973568177817). To get GRPO
advantage variance we need genuinely hard problems where the base sits well below the
ceiling. DAPO-Math-17k is the DAPO RL set: hard competition/olympiad problems with
INTEGER answers. The `qgallouedec/DAPO-Math-17k-Processed-Scored` mirror carries a
per-problem `Qwen3-32B_solve_rate` (a comparable model), so we FILTER to the headroom
band [DAPO_SOLVE_MIN, DAPO_SOLVE_MAX] -> problems that are hard-but-solvable = real
learning signal. Held-out eval is AIME 2025, so we DEDUP any training problem matching
an AIME 2025 problem (contamination guard; test must stay disjoint from train).

Same tool-agent parquet schema as scripts/prep_math_tool_agent.py, so it drops straight
into the agentic-judge training launcher. Ground truth = the integer answer (boxed
stripped), living ONLY in reward_model.ground_truth.

Knobs (env or flags): DAPO_OUT_DIR, DAPO_SOLVE_MIN (0.0), DAPO_SOLVE_MAX (1.0 = no
filter until we calibrate from the AIME baseline), DAPO_DEDUP_AIME (1), N_TRAIN/N_TEST.

Usage:
  DAPO_SOLVE_MIN=0.05 DAPO_SOLVE_MAX=0.6 \
  python3 scripts/prep_dapo_tool_agent.py --local_save_dir /Volumes/.../data/dapo_tool
"""

from __future__ import annotations

import argparse
import os

import datasets

# Keep in sync with scripts/tools/calc_tool.py (tool name) and judge_reward.py
# (_math_equiv). DAPO answers are integers, but we keep the general \boxed{} contract.
SYSTEM_PROMPT = (
    "You are a careful competition-math problem solver. Reason step by step. "
    "Whenever you need to do arithmetic, call the `calculator` tool with a single "
    "arithmetic expression (e.g. {\"expression\": \"12 * 7 + 3\"}) instead of "
    "computing it in your head, and use its result. You may call the tool several "
    "times. When you are confident, stop calling tools and give your final answer "
    "on its own line in the exact form \\boxed{<answer>} (put ONLY the final answer "
    "inside the box, e.g. \\boxed{204})."
)

# First candidate with a usable schema wins. The SCORED mirror is preferred because
# only it carries the Qwen3-32B_solve_rate column used for headroom filtering.
_CANDIDATES = [
    ("qgallouedec/DAPO-Math-17k-Processed-Scored", None, "train", "Qwen3-32B_solve_rate"),
    ("open-r1/DAPO-Math-17k-Processed", "en", "train", None),
    ("open-r1/DAPO-Math-17k-Processed", "all", "train", None),
    ("haizhongzheng/DAPO-Math-17K-cleaned", None, "train", None),
    ("sungyub/dapo-math-17k-verl", None, "train", None),
    ("BytedTsinghua-SIA/DAPO-Math-17k", None, "train", None),
]

# DAPO wraps each problem in a fixed instruction paragraph; strip it so the model gets
# just the problem plus OUR system prompt (which asks for the calculator + \boxed{}).
_DAPO_WRAP = "solve the following math problem"


def _last_boxed(s: str):
    """Content of the LAST \\boxed{...}/\\fbox{...}, brace-balanced; None if absent."""
    key = "\\boxed"
    i = s.rfind(key)
    if i < 0:
        key = "\\fbox"
        i = s.rfind(key)
        if i < 0:
            return None
    j = i + len(key)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    depth, start = 0, j
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
        j += 1
    return None


def _strip_wrapper(text: str) -> str:
    t = str(text).strip()
    if _DAPO_WRAP in t[:120].lower():
        parts = t.split("\n\n", 1)          # wrapper is a single leading paragraph
        if len(parts) == 2 and len(parts[1].strip()) > 10:
            return parts[1].strip()
    return t


def _extract_problem(example) -> str:
    p = example.get("problem") or example.get("question")
    if p:
        return _strip_wrapper(p)
    for key in ("prompt", "source_prompt"):
        pr = example.get(key)
        if isinstance(pr, list) and pr:
            # chat list -> last user turn (fall back to first element's content)
            for m in reversed(pr):
                if isinstance(m, dict) and m.get("content") and m.get("role") in (None, "user"):
                    return _strip_wrapper(m["content"])
            if isinstance(pr[0], dict):
                return _strip_wrapper(pr[0].get("content", ""))
        elif isinstance(pr, str) and pr:
            return _strip_wrapper(pr)
    return ""


def _extract_gt(example) -> str:
    rm = example.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") not in (None, ""):
        g = str(rm["ground_truth"])
    else:
        g = str(example.get("solution") or example.get("target")
                or example.get("answer") or "")
    g = g.strip()
    b = _last_boxed(g)
    return (b.strip() if b is not None else g)


def _solve_rate(example, key):
    if not key:
        return None
    try:
        return float(example.get(key))
    except Exception:  # noqa: BLE001
        return None


def _norm_prob(s) -> str:
    """Whitespace/case-insensitive fingerprint for dedup (first 200 non-space chars)."""
    return "".join(str(s).lower().split())[:200]


def _aime2025_fingerprints() -> set:
    for name, split in [("math-ai/aime25", "test"),
                        ("MathArena/aime_2025", "train"),
                        ("yentinglin/aime_2025", "train")]:
        try:
            ds = datasets.load_dataset(name, split=split)
        except Exception:  # noqa: BLE001
            continue
        fps = set()
        for r in ds:
            p = r.get("problem") or r.get("Problem") or r.get("question") or ""
            if p:
                fps.add(_norm_prob(p))
        if fps:
            print(f"[prep] AIME-2025 dedup set: {len(fps)} problems from {name}", flush=True)
            return fps
    print("[prep] WARN: could not load AIME-2025 for dedup; proceeding WITHOUT dedup", flush=True)
    return set()


def _load_dapo():
    for hid, cfg, split, sr_key in _CANDIDATES:
        try:
            ds = datasets.load_dataset(hid, cfg, split=split) if cfg else datasets.load_dataset(hid, split=split)
            print(f"[prep] loaded DAPO from '{hid}'{f' (config={cfg})' if cfg else ''}: "
                  f"{len(ds)} rows; cols={ds.column_names}; solve_rate_key={sr_key}", flush=True)
            return ds, hid, sr_key
        except Exception as e:  # noqa: BLE001
            print(f"[prep] candidate '{hid}' cfg={cfg} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("[prep] no DAPO-Math candidate could be loaded from the Hub")


def make_map_fn(split, data_source, sr_key, sr_min, sr_max, aime_fps):
    def process_fn(example, idx):
        problem = _extract_problem(example)
        gt = _extract_gt(example)
        sr = _solve_rate(example, sr_key)
        in_band = (sr is None) or (sr_min <= sr <= sr_max)  # no proxy -> keep (can't filter)
        not_leak = bool(problem) and (_norm_prob(problem) not in aime_fps)
        keep = bool(problem) and bool(gt) and in_band and not_leak
        # ALWAYS return a full dict (never None): datasets.map infers the schema from
        # row[0], so a None first return silently empties the split (air/56 bug).
        return {
            "data_source": data_source,
            "agent_name": "tool_agent" if keep else "",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "answer": gt,
                "solve_rate": sr if sr is not None else -1.0,
            },
            "_keep": keep,
        }

    return process_fn


def _prep_split(ds, split, data_source, sr_key, sr_min, sr_max, aime_fps, limit):
    mapped = ds.map(make_map_fn(split, data_source, sr_key, sr_min, sr_max, aime_fps),
                    with_indices=True, remove_columns=ds.column_names)
    kept = mapped.filter(lambda r: r["_keep"]).remove_columns(["_keep"])
    if limit > 0:
        kept = kept.select(range(min(limit, len(kept))))
    # solve_rate histogram (bucketed) so the log shows the difficulty distribution kept.
    try:
        from collections import Counter
        buckets = Counter()
        for e in kept["extra_info"]:
            sr = e["solve_rate"]
            buckets["no_sr" if sr < 0 else f"{int(sr*10)*10}-{int(sr*10)*10+10}%"] += 1
        hist = dict(sorted(buckets.items()))
    except Exception as e:  # noqa: BLE001
        hist = f"(unavailable: {type(e).__name__}: {e})"
    print(f"[prep] {split}: kept {len(kept)}/{len(ds)} rows; solve_rate histogram {hist}", flush=True)
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir",
                        default=os.environ.get("DAPO_OUT_DIR", "~/data/dapo_tool"))
    parser.add_argument("--solve_min", type=float, default=float(os.environ.get("DAPO_SOLVE_MIN", "0.0")))
    parser.add_argument("--solve_max", type=float, default=float(os.environ.get("DAPO_SOLVE_MAX", "1.0")))
    parser.add_argument("--dedup_aime", type=int, default=int(os.environ.get("DAPO_DEDUP_AIME", "1")))
    parser.add_argument("--train_limit", type=int, default=int(os.environ.get("N_TRAIN", "0")))
    parser.add_argument("--test_limit", type=int, default=int(os.environ.get("N_TEST", "500")))
    args = parser.parse_args()

    print(f"[prep] solve_rate band: [{args.solve_min}, {args.solve_max}]  dedup_aime={args.dedup_aime}", flush=True)
    aime_fps = _aime2025_fingerprints() if args.dedup_aime else set()

    ds, data_source, sr_key = _load_dapo()
    if sr_key is None:
        print("[prep] NOTE: chosen source has NO solve_rate column -> band filter is a no-op "
              "(all rows kept). Prefer the Scored mirror for headroom filtering.", flush=True)

    # DAPO ships a single train split; carve a small held-in tail as a nominal test
    # (unused in training: TEST_FREQ=-1, val_before_train=False) to keep schema parity.
    full = _prep_split(ds, "train", data_source, sr_key, args.solve_min, args.solve_max,
                       aime_fps, 0)
    n_test = min(args.test_limit, max(0, len(full) // 10))
    test_dataset = full.select(range(n_test)) if n_test else full.select(range(0))
    train_dataset = full.select(range(n_test, len(full)))
    if args.train_limit > 0:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"wrote {len(train_dataset)} train -> {train_path}")
    print(f"wrote {len(test_dataset)} test  -> {test_path}")
    for i in range(min(3, len(train_dataset))):
        r = train_dataset[i]
        print(f"[sample {i}] sr={r['extra_info']['solve_rate']} "
              f"gt={r['reward_model']['ground_truth']!r}  q={r['prompt'][1]['content'][:90]!r}")
