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

geometry3k is read at a pinned commit; DATA_MANIFEST.json beside the outputs records it, the
row counts and each file's sha256 (engine/lib/data_manifest.py).

Env:
    GEO3K_OUT_DIR   output directory (default: UC volume path)
    N_TRAIN         train rows, 0 = all   (default 64, smoke-sized)
    N_TEST          test rows,  0 = all   (default 128: the variance gate reads 64)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

DS = "hiyouga/geometry3k"
SOURCE = dm.Source(DS, "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7")   # unchanged since 2025-04
OUT = os.environ.get("GEO3K_OUT_DIR", "/Volumes/main/mshtelma/verl/data/geo3k")
N_TRAIN = int(os.environ.get("N_TRAIN", "64"))
N_TEST = int(os.environ.get("N_TEST", "128"))

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


def _print_versions() -> None:
    """Log the resolved versions first.

    A datasets/huggingface_hub skew shows up as an opaque
    ``HfFileSystem.find() got multiple values for keyword argument 'maxdepth'``
    from deep inside Hub glob resolution. Printing versions up front turns that
    into a one-glance diagnosis instead of a stack-trace hunt.
    """
    import importlib.metadata as md

    for pkg in ("datasets", "huggingface_hub", "fsspec", "pyarrow", "pillow"):
        try:
            print(f"  {pkg:16s} {md.version(pkg)}")
        except Exception:
            print(f"  {pkg:16s} <not installed>")


def main() -> None:
    print("resolved versions:")
    _print_versions()
    print(f"\nloading {SOURCE.label} ...")
    raw = SOURCE.load()

    train = slice_split(raw["train"], N_TRAIN).map(make_mapper("train"), with_indices=True)
    test = slice_split(raw["test"], N_TEST).map(make_mapper("test"), with_indices=True)

    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / dm.DIR_MANIFEST).unlink(missing_ok=True)   # never beside files it does not describe
    outputs = []
    for name, ds in (("train.parquet", train), ("test.parquet", test)):
        path = dm.write_parquet(out / name, ds)
        outputs.append(dm.output_record(path, len(ds)))
        print(f"wrote {len(ds)} {name.split('.')[0]} -> {path}")
    dm.write_manifest(
        out / dm.DIR_MANIFEST, tool="infra/geo3k/prep_geo3k.py",
        sources=[SOURCE.record(splits={k: len(v) for k, v in raw.items()})],
        sampling={"train.parquet": f"first {N_TRAIN} rows of train" if N_TRAIN > 0 else "all of train",
                  "test.parquet": f"first {N_TEST} rows of test" if N_TEST > 0 else "all of test"},
        outputs=outputs)
    print("\nsample row (images elided):")
    row = {k: v for k, v in train[0].items() if k != "images"}
    for key, val in row.items():
        print(f"  {key}: {str(val)[:160]}")


if __name__ == "__main__":
    main()
