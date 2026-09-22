#!/usr/bin/env python3
"""Prepare Hendrycks MATH (competition math) as a tool-agent dataset for verl.

MATH is used rather than GSM8K deliberately: GSM8K is grade-school arithmetic — a
35B-A3B + calculator already solves it ~95-100%, so the GRPO reward SATURATES and
there is nothing left to learn (measured: critic/score pinned 0.94-1.0, flat). That
is the reward-variance gate in docs/tuning.md failing in practice, and it is the
single most common way an RL run wastes GPU hours. MATH has real headroom AND its
answers are LaTeX expressions (\\frac{1}{2}, 2\\sqrt2, matrices) that do NOT
exact-match cleanly — exactly where the reference-guided LLM judge earns its keep
over a rule. Same parquet schema / same pipeline as the GSM8K prep, so it drops
straight into air/57 (the longer agentic-judge run).

Output parquet schema (verl AgentLoop + reward loop):

  data_source   "<resolved hub id>"      (free-form; our reward is custom)
  agent_name    "tool_agent"             -> ToolAgentLoop (multi-turn + tools)
  prompt        [ {role: system}, {role: user} ]   (raw chat; return_raw_chat=True)
  ability       "math"
  reward_model  {style: "rule", ground_truth: "<boxed answer, e.g. \\frac{1}{2}>"}
  extra_info    {split, index, question, answer, level, type}

The ground truth is the content of the solution's LAST \\boxed{...} (balanced
braces) — the canonical MATH answer. It lives ONLY in reward_model.ground_truth
(the judge's reference / our rule fallback), never handed to the model.

Difficulty filter: MATH_LEVELS (comma-separated, e.g. "3,4,5") keeps only those
levels so the base model starts around ~0.3-0.6 (real learning headroom) instead
of the near-ceiling all-levels mean. Empty = all levels.

Usage:
  python3 usecases/math/prep_data.py --local_save_dir /Volumes/.../data/math_tool
  MATH_LEVELS=3,4,5 python3 usecases/math/prep_data.py --local_save_dir ...
"""

from __future__ import annotations

import argparse
import os

import datasets

# The model is TOLD to use the calculator and to end with \boxed{...}. Keep this
# in sync with the tool name in usecases/math/tool.py and the answer handling
# in usecases/math/reward.py (_extract_pred_str / _math_equiv). Unlike
# GSM8K's "#### <number>", MATH answers are expressions, so we ask for \boxed{}.
SYSTEM_PROMPT = (
    "You are a careful competition-math problem solver. Reason step by step. "
    "Whenever you need to do arithmetic, call the `calculator` tool with a single "
    "arithmetic expression (e.g. {\"expression\": \"12 * 7 + 3\"}) instead of "
    "computing it in your head, and use its result. You may call the tool several "
    "times. When you are confident, stop calling tools and give your final answer "
    "on its own line in the exact form \\boxed{<answer>} (put ONLY the final answer "
    "inside the box, e.g. \\boxed{\\frac{1}{2}} or \\boxed{24})."
)

# The seven MATH subject configs, for hubs that only expose per-subject configs
# (EleutherAI/hendrycks_math) and must be concatenated into one train/test.
_MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]

# Tried in order; first that yields BOTH a train and a test split wins. All expose
# columns problem / solution / level / type. (Hub ids drift — lighteval/MATH moved
# to DigitalLearningGmbH/MATH-lighteval; the original hendrycks/competition_math has
# been intermittently unavailable — so we fall through a list.)
_CANDIDATES = [
    ("DigitalLearningGmbH/MATH-lighteval", None),
    ("EleutherAI/hendrycks_math", "__subjects__"),
    ("lighteval/MATH", "all"),
    ("hendrycks/competition_math", None),
    ("competition_math", None),
]


def _last_boxed(s: str) -> str | None:
    """Content of the LAST \\boxed{...} / \\fbox{...} in `s`, brace-balanced
    (so \\boxed{\\frac{1}{2}} yields `\\frac{1}{2}`, not `\\frac{1`). None if absent."""
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


def _level_int(level) -> int | None:
    """'Level 3' / 3 / '3' -> 3; unknown -> None."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    m = "".join(ch for ch in str(level) if ch.isdigit())
    return int(m) if m else None


def _load_math() -> tuple[datasets.DatasetDict, str]:
    for hid, cfg in _CANDIDATES:
        try:
            if cfg == "__subjects__":
                trains, tests = [], []
                for s in _MATH_SUBJECTS:
                    d = datasets.load_dataset(hid, s)
                    trains.append(d["train"])
                    tests.append(d["test"])
                ds = datasets.DatasetDict(
                    train=datasets.concatenate_datasets(trains),
                    test=datasets.concatenate_datasets(tests),
                )
            elif cfg:
                ds = datasets.load_dataset(hid, cfg)
            else:
                ds = datasets.load_dataset(hid)
            if "train" in ds and "test" in ds:
                print(f"[prep] loaded MATH from '{hid}'"
                      f"{f' (config={cfg})' if cfg and cfg != '__subjects__' else ''}: "
                      f"{len(ds['train'])} train / {len(ds['test'])} test", flush=True)
                return ds, hid
            print(f"[prep] '{hid}' lacks train/test splits ({list(ds)}); trying next", flush=True)
        except Exception as e:  # noqa: BLE001 - probe the next candidate
            print(f"[prep] candidate '{hid}' cfg={cfg} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("[prep] no MATH dataset candidate could be loaded from the Hub")


def make_map_fn(split: str, data_source: str, levels: set[int] | None):
    def process_fn(example, idx):
        problem = example.get("problem") or example.get("question") or ""
        solution = example.get("solution") or example.get("answer") or ""
        # explicit answer field if the hub provides a clean one, else box from soln
        answer = example.get("answer")
        gt = answer if (answer and "\\boxed" not in str(answer)) else _last_boxed(solution)
        lvl = _level_int(example.get("level"))
        keep = bool(gt) and (levels is None or lvl in levels)
        # ALWAYS return a full dict (never None): datasets.map infers the output
        # schema from the FIRST example, so a None first return silently yields an
        # empty result -- that dropped the ENTIRE train split when train[0] was a
        # filtered Level-1 problem (air/56 run 156146535176796: 0 train / 3669 test).
        # Mark rows with `_keep` instead and drop them in _prep_split.
        return {
            "data_source": data_source,
            "agent_name": "tool_agent" if keep else "",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(gt).strip() if gt else ""},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "answer": solution,
                "level": lvl if lvl is not None else -1,
                "type": example.get("type") or example.get("subject") or "",
            },
            "_keep": keep,
        }

    return process_fn


def _prep_split(ds, split: str, data_source: str, levels, limit: int):
    from collections import Counter

    mapped = ds.map(make_map_fn(split, data_source, levels), with_indices=True,
                    remove_columns=ds.column_names)
    kept = mapped.filter(lambda r: r["_keep"]).remove_columns(["_keep"])
    if limit > 0:
        kept = kept.select(range(min(limit, len(kept))))
    # `level` lives under extra_info (a struct column), NOT as a top-level column;
    # keep this telemetry non-fatal so a histogram hiccup can never fail the prep.
    try:
        hist = dict(sorted(Counter(e["level"] for e in kept["extra_info"]).items()))
    except Exception as e:  # noqa: BLE001
        hist = f"(unavailable: {type(e).__name__}: {e})"
    print(f"[prep] {split}: kept {len(kept)}/{len(ds)} rows; level histogram {hist}", flush=True)
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir",
                        default=os.environ.get("MATH_TOOL_OUT_DIR", "~/data/math_tool"),
                        help="Directory to write train.parquet / test.parquet (may be a /Volumes UC mount).")
    parser.add_argument("--levels", default=os.environ.get("MATH_LEVELS", ""),
                        help="Comma-separated difficulty levels to keep (e.g. '3,4,5'); empty = all.")
    parser.add_argument("--train_limit", type=int, default=int(os.environ.get("N_TRAIN", "0")),
                        help="If >0, keep only the first N train rows (smoke).")
    parser.add_argument("--test_limit", type=int, default=int(os.environ.get("N_TEST", "0")),
                        help="If >0, keep only the first N test rows (smoke).")
    args = parser.parse_args()

    levels = {int(x) for x in args.levels.replace(" ", "").split(",") if x} or None
    print(f"[prep] level filter: {sorted(levels) if levels else 'ALL'}", flush=True)

    dataset, data_source = _load_math()

    train_dataset = _prep_split(dataset["train"], "train", data_source, levels, args.train_limit)
    test_dataset = _prep_split(dataset["test"], "test", data_source, levels, args.test_limit)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"wrote {len(train_dataset)} train -> {train_path}")
    print(f"wrote {len(test_dataset)} test  -> {test_path}")
    # A couple of samples so the log shows the ground-truth extraction worked.
    for i in range(min(3, len(train_dataset))):
        r = train_dataset[i]
        print(f"[sample {i}] L{r['extra_info']['level']} {r['extra_info']['type']} "
              f"gt={r['reward_model']['ground_truth']!r}  q={r['prompt'][1]['content'][:80]!r}")
