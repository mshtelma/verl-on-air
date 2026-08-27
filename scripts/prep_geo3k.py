#!/usr/bin/env python3
"""geo3k -> verl parquet on a UC Volume.

verl's RL dataset schema (one row per prompt):
    data_source   str   -> selects the reward scorer in
                           verl/utils/reward_score/__init__.py. MUST stay
                           "hiyouga/geometry3k" to reach geo3k.compute_score.
    prompt        list  -> chat messages
    images        list  -> PIL images (data.image_key=images)
    ability       str
    reward_model  dict  -> {"style": "rule", "ground_truth": ...}
    extra_info    dict

The instruction suffix is NOT cosmetic: geo3k's reward gives 10% weight to a
regex requiring `<think>...</think>` followed by `\\boxed{...}`. Without it,
format_reward is pinned at 0 and you lose that component permanently.

Env:
    GEO3K_OUT_DIR   output directory (default: UC volume path)
    N_TRAIN         train rows, 0 = all   (default 64, smoke-sized)
    N_TEST          test rows,  0 = all   (default 8)
"""

from __future__ import annotations

import os

import datasets

DS = "hiyouga/geometry3k"
OUT = os.environ.get("GEO3K_OUT_DIR", "/Volumes/main/mshtelma/verl/data/geo3k")
N_TRAIN = int(os.environ.get("N_TRAIN", "64"))
N_TEST = int(os.environ.get("N_TEST", "8"))

INSTRUCTION = (
    r"You FIRST think about the reasoning process as an internal monologue and then "
    r"provide the final answer. The reasoning process MUST BE enclosed within "
    r"<think> </think> tags. The final answer MUST BE put in \boxed{}."
)


def make_mapper(split: str):
    def _map(example, idx):
        problem = example.pop("problem")
        answer = example.pop("answer")
        images = example.pop("images")
        return {
            "data_source": DS,
            "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
            },
        }

    return _map


def slice_split(ds, n: int):
    return ds if n <= 0 else ds.select(range(min(n, len(ds))))


def main() -> None:
    print(f"loading {DS} ...")
    raw = datasets.load_dataset(DS)

    train = slice_split(raw["train"], N_TRAIN).map(make_mapper("train"), with_indices=True)
    test = slice_split(raw["test"], N_TEST).map(make_mapper("test"), with_indices=True)

    os.makedirs(OUT, exist_ok=True)
    train_path = os.path.join(OUT, "train.parquet")
    test_path = os.path.join(OUT, "test.parquet")
    train.to_parquet(train_path)
    test.to_parquet(test_path)

    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(test)} test  -> {test_path}")
    print("\nsample row (images elided):")
    row = {k: v for k, v in train[0].items() if k != "images"}
    for key, val in row.items():
        print(f"  {key}: {str(val)[:160]}")


if __name__ == "__main__":
    main()
