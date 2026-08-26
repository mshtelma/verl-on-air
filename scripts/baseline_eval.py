#!/usr/bin/env python3
"""Measure whether a dataset can actually teach this model anything, BEFORE
spending 16 H100s on GRPO.

Why this exists
---------------
GRPO's gradient comes entirely from reward variance *within* a group of `n`
samples for the same prompt. The advantage is the group-normalised reward, so
if all n samples score identically the advantage is 0 and that prompt
contributes **nothing** to the update. Two ways to get a flat run:

  * model too strong -> every sample correct  -> zero variance
  * model too weak   -> every sample wrong    -> zero variance

Qwen3.5-35B-A3B is a post-trained model whose card claims it beats
Qwen3-235B-A22B, and geo3k is not hard. So the number that decides whether the
demo shows a rising curve is NOT pass@1 — it is:

    effective_fraction = P(group has non-zero reward variance)

That fraction is the share of your batch that produces gradient. This script
measures it with rollout only (no training), so it costs one short 8xH100 job.

Env:
    MODEL_PATH   model dir or HF id
    EVAL_FILE    verl-format parquet (uses the `test` split you prepared)
    N_SAMPLES    samples per prompt (match rollout_n, default 5)
    N_PROMPTS    prompts to evaluate (default 64)
    TEMPERATURE  sampling temperature (default 1.0, matching rollout)
    GEN_TP       vLLM tensor parallel size (default 8)
"""

from __future__ import annotations

import os
import statistics

import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/mshtelma/rlonair/verl/models/Qwen3.5-35B-A3B")
EVAL_FILE = os.environ.get("EVAL_FILE", "/Volumes/mshtelma/rlonair/verl/data/geo3k/test.parquet")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "5"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "64"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
GEN_TP = int(os.environ.get("GEN_TP", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))


def score_fn(data_source: str):
    """Resolve the same scorer verl would use for this data_source."""
    from verl.utils.reward_score import default_compute_score

    def _score(response: str, ground_truth: str) -> float:
        return float(default_compute_score(data_source, response, ground_truth))

    return _score


def main() -> None:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(EVAL_FILE).head(N_PROMPTS)
    print(f"model   : {MODEL_PATH}")
    print(f"data    : {EVAL_FILE}  ({len(df)} prompts x {N_SAMPLES} samples)")

    data_source = df["data_source"].iloc[0]
    score = score_fn(data_source)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=GEN_TP,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        limit_mm_per_prompt={"image": 4},
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
    )
    params = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=1.0,
                            max_tokens=MAX_TOKENS)

    requests = []
    for _, row in df.iterrows():
        text = processor.apply_chat_template(
            list(row["prompt"]), tokenize=False, add_generation_prompt=True
        )
        entry: dict = {"prompt": text}
        images = row.get("images")
        if images is not None and len(images) > 0:
            entry["multi_modal_data"] = {"image": list(images)}
        requests.append(entry)

    outputs = llm.generate(requests, params)

    per_group_mean, effective, all_right, all_wrong = [], 0, 0, 0
    for row, out in zip(df.itertuples(), outputs, strict=True):
        gt = row.reward_model["ground_truth"]
        rewards = [score(c.text, gt) for c in out.outputs]
        per_group_mean.append(statistics.fmean(rewards))
        if len(set(rewards)) > 1:
            effective += 1
        elif rewards[0] > 0.5:
            all_right += 1
        else:
            all_wrong += 1

    n = len(per_group_mean)
    frac = effective / n
    print("\n" + "=" * 66)
    print(f"mean reward              : {statistics.fmean(per_group_mean):.3f}")
    print(f"groups with variance     : {effective}/{n}  ({frac:.1%})   <-- usable signal")
    print(f"  saturated (all correct): {all_right}/{n}  ({all_right / n:.1%})")
    print(f"  floored   (all wrong)  : {all_wrong}/{n}  ({all_wrong / n:.1%})")
    print("=" * 66)

    if frac >= 0.40:
        verdict = "GOOD - healthy gradient signal, train on this."
    elif frac >= 0.20:
        verdict = ("MARGINAL - workable but slow. Raise rollout_n, or move to "
                   "harder data (math_dapo / aime).")
    elif all_right > all_wrong:
        verdict = ("SATURATED - the model already solves this. Use harder data "
                   "(math_dapo / aime) or the -Base checkpoint.")
    else:
        verdict = ("FLOORED - the model cannot solve any of it. Use easier data "
                   "or relax the reward (e.g. weight format higher).")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
