#!/usr/bin/env python3
"""Prepare the OfficeQA PATH-REPORT GRPO training dataset (Phase 1: bootstrap on EASY).

Packages OfficeQA questions in the parquet schema verl's fully-async rollout expects, routed
to the CUSTOM ``path_report_agent`` loop (scripts/officeqa/path_report_agent_loop.py) with the
SUBMIT contract + 3-phase funnel. The reward (scripts/reward/officeqa_path_report_reward.py)
scores the loop-emitted path_report_record; ``reward_model.ground_truth`` carries the gold answer.

Phase 1 trains on EASY questions only (prove the loop LEARNS): the 133 hard are the HELD-OUT eval
(training on them = contamination), and synthetic hard questions come LATER (Phase 2).

Output schema:
    data_source   "officeqa_path_report"
    agent_name    "path_report_agent"          -> our registered custom loop
    prompt        [{system: PATH_REPORT_SYSTEM_PROMPT_SUBMIT (funnel-filled)}, {user: question}]
    ability       "grounded-research"
    reward_model  {"style": "rule", "ground_truth": answer}
    extra_info    {split, index, uid, question, question_requirements, difficulty, answer}

CRITICAL: the system prompt's funnel numbers ([[MAX_TURNS]]/[[RETRIEVAL_LOCK]]/[[SUBMIT_ONLY]])
are filled from --max-turns/--nudge-window/--submit-only-window here and MUST equal the launcher's
rollout.multi_turn.max_assistant_turns / OQ_NUDGE_WINDOW / OQ_SUBMIT_ONLY_WINDOW, or the prompt
would describe a different funnel than the loop enforces (train/deploy drift).

Env/CLI mirror prep_officeqa_tool_agent.py, plus the funnel knobs.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# The SUBMIT system prompt (with the [[...]] funnel placeholders) is the SAME text the collector
# advertises, so training conditions on exactly what was validated + what deploy will use.
from path_report_collect import PATH_REPORT_SYSTEM_PROMPT_SUBMIT  # noqa: E402


def _fill_prompt(max_turns: int, nudge_window: int, submit_only_window: int) -> str:
    return (PATH_REPORT_SYSTEM_PROMPT_SUBMIT
            .replace("[[MAX_TURNS]]", str(max_turns))
            .replace("[[RETRIEVAL_LOCK]]", str(nudge_window))
            .replace("[[SUBMIT_ONLY]]", str(submit_only_window)))


def _rows(path: str, difficulty: str, uids: set[str], limit: int, max_prompt_chars: int,
          sys_prompt_len: int) -> list[dict]:
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
            if max_prompt_chars > 0 and sys_prompt_len + len(question) > max_prompt_chars:
                continue
            out.append({"uid": uid, "index": idx, "question": question, "answer": answer,
                        "difficulty": diff,
                        "question_requirements": (r.get("question_requirements") or "").strip()})
    if uids:
        rank = {u: i for i, u in enumerate(sorted(uids))}
        out.sort(key=lambda r: rank.get(r["uid"], len(rank)))
    return out[:limit] if limit > 0 else out


def _to_verl(r: dict, split: str, sys_prompt: str) -> dict:
    return {
        "data_source": "officeqa_path_report",
        "agent_name": "path_report_agent",
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": r["question"]},
        ],
        "ability": "grounded-research",
        "reward_model": {"style": "rule", "ground_truth": r["answer"]},
        "extra_info": {
            "split": split, "index": r["index"], "uid": r["uid"],
            "question": r["question"], "question_requirements": r["question_requirements"],
            "difficulty": r["difficulty"], "answer": r["answer"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.environ.get("OFFICEQA_EVAL_CSV",
                   "/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv"))
    p.add_argument("--out-dir", default=os.environ.get("OFFICEQA_PR_OUT_DIR",
                   "/Volumes/main/mshtelma/verl/data/officeqa_path_report"))
    p.add_argument("--difficulty", default=os.environ.get("OFFICEQA_PR_DIFFICULTY", "easy").lower())
    p.add_argument("--limit", type=int, default=int(os.environ.get("OFFICEQA_PR_LIMIT", "0")))
    p.add_argument("--uids", default=os.environ.get("OFFICEQA_PR_UIDS", ""))
    p.add_argument("--max-prompt-chars", type=int,
                   default=int(os.environ.get("OFFICEQA_PR_MAX_PROMPT_CHARS", "12000")))
    # Funnel knobs -- MUST match the launcher (see module docstring).
    p.add_argument("--max-turns", type=int, default=int(os.environ.get("EVAL_MAX_TURNS", "80")))
    p.add_argument("--nudge-window", type=int, default=int(os.environ.get("OQ_NUDGE_WINDOW", "20")))
    p.add_argument("--submit-only-window", type=int,
                   default=int(os.environ.get("OQ_SUBMIT_ONLY_WINDOW", "5")))
    p.add_argument("--val-frac", type=float, default=float(os.environ.get("OFFICEQA_PR_VAL_FRAC", "0.1")))
    args = p.parse_args()

    sys_prompt = _fill_prompt(args.max_turns, args.nudge_window, args.submit_only_window)
    if "[[" in sys_prompt:
        raise SystemExit("unfilled placeholder remains in the system prompt; check the funnel knobs")
    uids = {u.strip() for u in args.uids.split(",") if u.strip()}
    rows = _rows(args.csv, args.difficulty, uids, args.limit, args.max_prompt_chars, len(sys_prompt))
    if not rows:
        raise SystemExit(f"no OfficeQA rows selected from {args.csv} (difficulty={args.difficulty!r})")

    # Deterministic train/val split: hold out a slice for verl's validation dataset (not learned on).
    n_val = max(1, int(len(rows) * args.val_frac)) if len(rows) > 1 else 1
    val_rows, train_rows = rows[:n_val], rows[n_val:] or rows[:1]
    train = datasets.Dataset.from_list([_to_verl(r, "train", sys_prompt) for r in train_rows])
    val = datasets.Dataset.from_list([_to_verl(r, "val", sys_prompt) for r in val_rows])

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"funnel: max_turns={args.max_turns} nudge_window={args.nudge_window} "
          f"submit_only_window={args.submit_only_window}")
    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(val)} val   -> {val_path}")
    print(f"difficulty={args.difficulty or 'all'}  (Phase-1 bootstrap; hard is held-out eval)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
