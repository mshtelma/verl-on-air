#!/usr/bin/env python3
"""Prepare GSM8K as a tool-agent (multi-turn) dataset for verl.

Output parquet schema is what verl's AgentLoop rollout + reward loop expect:

  data_source   "openai/gsm8k"           (routes the reward; free-form for us)
  agent_name    "tool_agent"             -> selects ToolAgentLoop (multi-turn + tools)
  prompt        [ {role: system}, {role: user} ]   (raw chat; return_raw_chat=True)
  ability       "math"
  reward_model  {style: "rule", ground_truth: "<final number>"}
  extra_info    {split, index, question, answer, num_turns(filled at rollout)}

Deliberately DIFFERENT from verl's examples/data_preprocess/gsm8k_tool_agent_loop.py:
that one wires a `calc_gsm8k_reward` BaseTool and passes the ground truth into it
via extra_info.tools_kwargs — an answer-checking oracle that leaks the label into
the rollout. We instead use a stateless `calculator` @function_tool (see
scripts/tools/calc_tool.py). Function tools are global and ignore tools_kwargs
(verl/experimental/agent_loop/tool_agent_loop.py:510-514), so NO tools_kwargs are
emitted here — a row just needs agent_name="tool_agent". The ground truth lives
only in reward_model.ground_truth, used by the LLM-judge reward for validation /
fallback (scripts/reward/judge_reward.py), never handed to the model.

Usage:
  python3 scripts/prep_gsm8k_tool_agent.py --local_save_dir /Volumes/.../data/gsm8k_tool
  # or from an already-downloaded copy:
  python3 scripts/prep_gsm8k_tool_agent.py --local_dataset_path /path/to/gsm8k \
      --local_save_dir /Volumes/.../data/gsm8k_tool
"""

import argparse
import os
import re

import datasets

# The model is TOLD to use the calculator and to end with `#### <answer>`. Keep
# this instruction in sync with the tool name in scripts/tools/calc_tool.py and
# with the answer regex in scripts/reward/judge_reward.py (_HASH_RE).
SYSTEM_PROMPT = (
    "You are a careful math problem solver. Reason step by step. Whenever you need "
    "to do arithmetic, call the `calculator` tool with a single arithmetic "
    "expression (e.g. {\"expression\": \"18 - 3 - 4\"}) instead of computing it in "
    "your head, and use its result. You may call the tool several times. When you "
    "are confident, stop calling tools and give the final answer on its own line "
    "in the exact format `#### <answer>` (a single number)."
)

_SOLUTION_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_solution(answer_raw: str) -> str:
    m = _SOLUTION_RE.search(answer_raw)
    assert m is not None, f"no `#### answer` found in: {answer_raw!r}"
    return m.group(0).split("#### ")[1].replace(",", "").strip()


def make_map_fn(split: str):
    def process_fn(example, idx):
        question_raw = example.pop("question")
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        return {
            "data_source": "openai/gsm8k",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_raw},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question_raw,
                "answer": answer_raw,
            },
        }

    return process_fn


if __name__ == "__main__":
    # Defaults read from env so the air job can configure via env_variables:
    # (matching scripts/prep_geo3k.py); CLI flags still override.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default=os.environ.get("GSM8K_TOOL_OUT_DIR", "~/data/gsm8k_tool"),
                        help="Directory to write train.parquet / test.parquet (may be a /Volumes UC mount).")
    parser.add_argument("--local_dataset_path", default=os.environ.get("GSM8K_LOCAL_PATH") or None,
                        help="Path to an already-downloaded GSM8K dataset; omit to pull from the Hub.")
    parser.add_argument("--train_limit", type=int, default=int(os.environ.get("N_TRAIN", "0")),
                        help="If >0, keep only the first N train rows (smoke).")
    parser.add_argument("--test_limit", type=int, default=int(os.environ.get("N_TEST", "0")),
                        help="If >0, keep only the first N test rows (smoke).")
    args = parser.parse_args()

    src = args.local_dataset_path or "openai/gsm8k"
    dataset = datasets.load_dataset(src, "main")

    train_dataset = dataset["train"].map(make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(make_map_fn("test"), with_indices=True)
    if args.train_limit > 0:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))
    if args.test_limit > 0:
        test_dataset = test_dataset.select(range(min(args.test_limit, len(test_dataset))))

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"wrote {len(train_dataset)} train -> {train_path}")
    print(f"wrote {len(test_dataset)} test  -> {test_path}")
