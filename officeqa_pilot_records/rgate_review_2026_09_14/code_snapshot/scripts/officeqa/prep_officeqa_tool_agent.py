#!/usr/bin/env python3
"""Prepare a small OfficeQA tool-agent smoke dataset for verl.

This is intentionally **not** bulk synthesis. It packages a tiny, labeled set of
OfficeQA questions in the exact parquet schema the fully-async ToolAgentLoop and the
grounded reward expect, so the first GPU job can prove wiring without confusing an
integration failure with a learning result.

Output schema (same shape as the proven GSM8K/MATH tool-agent datasets):

    data_source   "officeqa"
    agent_name    "tool_agent"
    prompt        [{role: system, content: ...}, {role: user, content: question}]
    ability       "grounded-research"
    reward_model  {"style": "rule", "ground_truth": answer}
    extra_info    {split, index, uid, question, source_files, difficulty,
                   is_composite, answer, atom_spec}

The system prompt matches the eval harness, while the final instruction is tightened:
a real final answer must be one clean value/list (the strict gate's contract), and an
abstention remains an explicit DATA NOT AVAILABLE commitment.

Env/CLI:
    OFFICEQA_EVAL_CSV   input CSV (default Volume officeqa_full.csv)
    OFFICEQA_TOOL_OUT_DIR output dir (default Volume data/officeqa_tool)
    OFFICEQA_PREP_LIMIT   cap rows after filtering (default 8)
    OFFICEQA_PREP_MAX_PROMPT_CHARS drop rows above this rough prompt budget (default 12000)
    OFFICEQA_PREP_UIDS    optional comma-separated UIDs to force-include
    OFFICEQA_PREP_DIFFICULTY easy|hard|'' (default all)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import datasets

# Keep the system prompt identical to eval; the training-data user prompt carries the
# additional output-contract reminder so the strict reward has a learnable commitment.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from eval_officeqa_agentic import SYSTEM_PROMPT  # noqa: E402

USER_CONTRACT = (
    "Answer this Treasury Bulletin question using the tools. Work from retrieved "
    "evidence, then commit exactly once on a final assistant line as\n"
    "<FINAL_ANSWER>value</FINAL_ANSWER>\n"
    "The value must be one clean scalar/list in the requested unit and precision -- no "
    "candidate lists, prose, or units unless explicitly requested. If the evidence is "
    "not available, use <FINAL_ANSWER>DATA NOT AVAILABLE</FINAL_ANSWER>.\n\n"
    "Question: {question}"
)


def _bool_from_sources(source_files: str) -> bool:
    parts = [p for p in str(source_files or "").replace("\r", "\n").replace(";", "\n").replace(",", "\n").split("\n") if p.strip()]
    return len(parts) > 1


def _rows(path: str, difficulty: str, uids: set[str], limit: int,
          max_prompt_chars: int = 12_000) -> list[dict]:
    out = []
    with open(path, newline="") as fh:
        for idx, r in enumerate(csv.DictReader(fh)):
            uid = (r.get("uid") or "").strip()
            question = (r.get("question") or "").strip()
            answer = (r.get("answer") or "").strip()
            if not uid or not question or not answer:
                continue
            diff = (r.get("difficulty") or "").strip().lower()
            if difficulty and diff != difficulty:
                continue
            if uids and uid not in uids:
                continue
            rec = {
                "uid": uid,
                "index": idx,
                "question": question,
                "answer": answer,
                "difficulty": diff,
                "source_files": (r.get("source_files") or "").strip(),
            }
            # A rough character guard catches the very long OfficeQA prompts before
            # tokenizer-side filtering can silently reduce the training set to zero.
            if max_prompt_chars > 0 and len(SYSTEM_PROMPT) + len(question) > max_prompt_chars:
                continue
            out.append(rec)
    # Force-included UIDs keep their order; otherwise deterministic dataset order.
    if uids:
        rank = {u: i for i, u in enumerate(sorted(uids))}
        out.sort(key=lambda r: rank.get(r["uid"], len(rank)))
    return out[:limit] if limit > 0 else out


def _to_verl(r: dict, split: str) -> dict:
    return {
        "data_source": "officeqa",
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTRACT.format(question=r["question"])},
        ],
        "ability": "grounded-research",
        "reward_model": {"style": "rule", "ground_truth": r["answer"]},
        "extra_info": {
            "split": split,
            "index": r["index"],
            "uid": r["uid"],
            "question": r["question"],
            "answer": r["answer"],
            "source_files": r["source_files"],
            "difficulty": r["difficulty"],
            "is_composite": _bool_from_sources(r["source_files"]),
            "atom_spec": "",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.environ.get("OFFICEQA_EVAL_CSV",
                   "/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv"))
    p.add_argument("--out-dir", default=os.environ.get("OFFICEQA_TOOL_OUT_DIR",
                   "/Volumes/main/mshtelma/verl/data/officeqa_tool"))
    p.add_argument("--limit", type=int, default=int(os.environ.get("OFFICEQA_PREP_LIMIT", "8")))
    p.add_argument("--difficulty", default=os.environ.get("OFFICEQA_PREP_DIFFICULTY", "").lower())
    p.add_argument("--uids", default=os.environ.get("OFFICEQA_PREP_UIDS", ""))
    p.add_argument("--max-prompt-chars", type=int,
                   default=int(os.environ.get("OFFICEQA_PREP_MAX_PROMPT_CHARS", "12000")))
    args = p.parse_args()
    uids = {u.strip() for u in args.uids.split(",") if u.strip()}
    rows = _rows(args.csv, args.difficulty, uids, args.limit, args.max_prompt_chars)
    if not rows:
        raise SystemExit(f"no OfficeQA rows selected from {args.csv}")
    train = datasets.Dataset.from_list([_to_verl(r, "train") for r in rows])
    # verl always constructs the validation dataset; HF datasets cannot infer a schema
    # from zero rows. Use one held-out row (not used for learning) for this smoke.
    val = datasets.Dataset.from_list([_to_verl(rows[-1], "val")])
    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    train_path = os.path.join(os.path.expanduser(args.out_dir), "train.parquet")
    val_path = os.path.join(os.path.expanduser(args.out_dir), "test.parquet")
    train_path = os.path.join(os.path.expanduser(args.out_dir), "train.parquet")
    val_path = os.path.join(os.path.expanduser(args.out_dir), "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(val)} val   -> {val_path}")
    for r in train:
        print(f"  {r['extra_info']['uid']} [{r['extra_info']['difficulty']}] "
              f"composite={r['extra_info']['is_composite']} gt={r['reward_model']['ground_truth']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
