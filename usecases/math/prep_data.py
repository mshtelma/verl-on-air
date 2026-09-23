#!/usr/bin/env python3
"""Prepare Hendrycks MATH (competition math) as a tool-agent dataset for verl.

MATH is used rather than GSM8K deliberately: GSM8K is grade-school arithmetic — a
35B-A3B + calculator already solves it ~95-100%, so the GRPO reward SATURATES and
there is nothing left to learn (measured: critic/score pinned 0.94-1.0, flat). That
is the reward-variance gate in docs/tuning.md failing in practice, and it is the
single most common way an RL run wastes GPU hours. MATH has real headroom AND its
answers are LaTeX expressions (\\frac{1}{2}, 2\\sqrt2, matrices) that do NOT
exact-match cleanly — exactly where the reference-guided LLM judge earns its keep
over a rule.

Output parquet schema (verl AgentLoop + reward loop):

  data_source   "DigitalLearningGmbH/MATH-lighteval"   (free-form; our reward is custom)
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

MATH is read at a pinned commit and must hold the recorded content (row counts +
a digest of its problem/solution pairs). A mirror is used only with
ALLOW_FALLBACK_SOURCE=1, and only if it holds that same content. DATA_MANIFEST.json
beside the outputs records the source, the filters, the counts and each file's sha256.

Usage:
  python3 usecases/math/prep_data.py --local_save_dir /Volumes/.../data/math_tool
  MATH_LEVELS=3,4,5 python3 usecases/math/prep_data.py --local_save_dir ...
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

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

# MATH (Hendrycks et al.) at a pinned commit, and the content any accepted copy must hold: per
# split, the row count and an order-independent digest of its (problem, solution) pairs.
MATH = dm.Source("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")
MATH_CONTENT = {
    "train": (7500, "fbdc8b6908196fdde6f076560b176af917c439bbe85fe0bf6ea5db274ea60ccc"),
    "test": (5000, "13d5c4873d6a6b184618aad7d1ff5b748c0adcbc56656f916d2b0a8bf8de08b5"),
}
# The one mirror that holds that content: its seven subject configs, concatenated in this order,
# are the same rows in the same order (checked 2026-09-23). lighteval/MATH is gone, and
# hendrycks/competition_math is a loader script that datasets>=4 cannot run.
MATH_MIRROR = dm.Source("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3")
_MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
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


def _by_subject(src: dm.Source) -> datasets.DatasetDict:
    parts = [dm.Source(src.hf_id, src.revision, subject).load() for subject in _MATH_SUBJECTS]
    return datasets.DatasetDict(train=datasets.concatenate_datasets([d["train"] for d in parts]),
                                test=datasets.concatenate_datasets([d["test"] for d in parts]))


def _is_math(ds) -> str | None:
    """None if `ds` holds exactly MATH_CONTENT, else why not."""
    for split, (n, digest) in MATH_CONTENT.items():
        if split not in ds:
            return f"no {split!r} split"
        got_n, got = len(ds[split]), dm.content_digest(ds[split], ["problem", "solution"])
        if (got_n, got) != (n, digest):
            return f"{split}: {got_n} rows, digest {got[:12]} -- expected {n}, {digest[:12]}"
    return None


def _load_math() -> tuple[datasets.DatasetDict, dict]:
    try:
        ds, record = dm.load_verified([(MATH, lambda src: src.load()), (MATH_MIRROR, _by_subject)], _is_math)
    except dm.SourceError as e:
        raise SystemExit(f"[prep] {e}") from e
    print(f"[prep] loaded MATH from {record['hf_id']}@{record['revision'][:12]}: "
          f"{len(ds['train'])} train / {len(ds['test'])} test", flush=True)
    return ds, record


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
        # filtered Level-1 problem (measured: 0 train / 3669 test).
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
    return kept, {"source_rows": len(ds), "kept": len(kept), "levels": hist}


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    levels = {int(x) for x in args.levels.replace(" ", "").split(",") if x} or None
    print(f"[prep] level filter: {sorted(levels) if levels else 'ALL'}", flush=True)

    dataset, source = _load_math()
    # The dataset's name, not the mirror's: a verified mirror holds the same rows.
    data_source = MATH.hf_id

    train_dataset, train_stats = _prep_split(dataset["train"], "train", data_source, levels, args.train_limit)
    test_dataset, test_stats = _prep_split(dataset["test"], "test", data_source, levels, args.test_limit)

    save_dir = Path(os.path.expanduser(args.local_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / dm.DIR_MANIFEST).unlink(missing_ok=True)   # never beside files it does not describe
    outputs = []
    for name, ds in (("train.parquet", train_dataset), ("test.parquet", test_dataset)):
        path = dm.write_parquet(save_dir / name, ds)
        outputs.append(dm.output_record(path, len(ds)))
        print(f"wrote {len(ds)} {name.split('.')[0]} -> {path}")
    dm.write_manifest(
        save_dir / dm.DIR_MANIFEST, tool="usecases/math/prep_data.py",
        sources=[{**source, "content": MATH_CONTENT, "splits": {"train": train_stats, "test": test_stats}}],
        filters={"levels": sorted(levels) if levels else "all",
                 "ground_truth": "last \\boxed{} of the solution (or a clean `answer` field); rows without one dropped"},
        sampling={"train.parquet": f"first {args.train_limit} kept rows" if args.train_limit > 0 else "all kept rows",
                  "test.parquet": f"first {args.test_limit} kept rows" if args.test_limit > 0 else "all kept rows"},
        outputs=outputs)
    # A couple of samples so the log shows the ground-truth extraction worked.
    for i in range(min(3, len(train_dataset))):
        r = train_dataset[i]
        print(f"[sample {i}] L{r['extra_info']['level']} {r['extra_info']['type']} "
              f"gt={r['reward_model']['ground_truth']!r}  q={r['prompt'][1]['content'][:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
