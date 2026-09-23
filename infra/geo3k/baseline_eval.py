#!/usr/bin/env python3
"""Measure whether a dataset can actually teach this model anything, BEFORE
spending 16 H100s on GRPO.

Why this exists
---------------
GRPO's task-reward policy gradient comes from reward variance *within* a group
of `n` samples for the same prompt. The advantage is the group-normalised
reward, so if all n samples score identically that prompt contributes no
task-reward signal to the update (KL and other regularisers still act). Two
ways to get a flat run:

  * model too strong -> every sample correct  -> zero variance
  * model too weak   -> every sample wrong    -> zero variance

Aggregate pass@1 does not tell you which regime you are in, so the number
this measures is:

    effective_fraction = P(group has non-zero reward variance)

-- the share of your batch that produces a task-reward signal -- with a 95%
Wilson interval, using rollout only (no training): one short 8xH100 job.

The prompts are built the way verl's RL dataset builds them: images decoded
from the parquet records (pandas returns an HF Image feature as a
{"bytes", "path"} dict, which vLLM does not accept), each `<image>` marker
turned into a typed image part, and the chat template applied by the model's
own processor, so the prompt carries the model's real image placeholders.

Env:
    MODEL_PATH   model dir or HF id
    EVAL_FILE    verl-format parquet (uses the `test` split you prepared)
    N_SAMPLES    samples per prompt (match rollout_n, default 5)
    N_PROMPTS    prompts to evaluate (default 64); fewer rows in EVAL_FILE is an error
    TEMPERATURE  sampling temperature (default 1.0, matching rollout)
    GEN_TP       vLLM tensor parallel size (default 8)
"""

from __future__ import annotations

import math
import os
import re
import statistics
from io import BytesIO
from typing import Any

import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
EVAL_FILE = os.environ.get("EVAL_FILE", "/Volumes/main/mshtelma/verl/data/geo3k/test.parquet")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "5"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "64"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
GEN_TP = int(os.environ.get("GEN_TP", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))


def decode_image(img: Any):
    """A PIL RGB image from what the parquet round-trip left behind."""
    from PIL import Image

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, (str, os.PathLike)):
        return Image.open(img).convert("RGB")
    raise TypeError(f"cannot decode an image from {type(img).__name__}")


def build_messages(prompt: list[dict], n_images: int) -> list[dict]:
    """verl's RLHFDataset._build_messages: each `<image>` marker becomes a typed image part (the
    processor's chat template renders the model's own placeholder for it); text stays text."""
    out, used = [], 0
    for m in prompt:
        content = m["content"]
        if n_images == 0 or not isinstance(content, str):
            out.append(dict(m))
            continue
        parts = []
        for seg in (s for s in re.split("(<image>)", content) if s):
            if seg == "<image>":
                parts.append({"type": "image"})
                used += 1
            else:
                parts.append({"type": "text", "text": seg})
        out.append({**m, "content": parts})
    if used != n_images:
        raise ValueError(f"prompt has {used} <image> marker(s) for {n_images} image(s)")
    return out


def build_requests(df: pd.DataFrame, processor) -> list[dict]:
    requests = []
    for _, row in df.iterrows():
        images = row.get("images")
        images = [] if images is None else [decode_image(i) for i in list(images)]
        messages = build_messages(list(row["prompt"]), len(images))
        entry: dict = {"prompt": processor.apply_chat_template(messages, tokenize=False,
                                                               add_generation_prompt=True)}
        if images:
            entry["multi_modal_data"] = {"image": images}
        requests.append(entry)
    return requests


def load_prompts(path: str, n: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if len(df) < n:
        raise SystemExit(f"FATAL: {path} has {len(df)} rows but N_PROMPTS={n}: re-run the prep "
                         f"with N_TEST>={n} (or N_TEST=0 for the full split), or lower N_PROMPTS.")
    return df.head(n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def score_fn(data_source: str):
    """Resolve the same scorer verl would use for this data_source."""
    from verl.utils.reward_score import default_compute_score

    def _score(response: str, ground_truth: str) -> float:
        return float(default_compute_score(data_source, response, ground_truth))

    return _score


def main() -> None:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    df = load_prompts(EVAL_FILE, N_PROMPTS)
    print(f"model   : {MODEL_PATH}")
    print(f"data    : {EVAL_FILE}  ({len(df)} prompts x {N_SAMPLES} samples)")

    data_source = df["data_source"].iloc[0]
    score = score_fn(data_source)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    requests = build_requests(df, processor)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=GEN_TP,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        limit_mm_per_prompt={"image": 4},
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
        # As every other vLLM here (serve_and_eval.sh, serve_judge.sh, the training rollout): the
        # intra-node 8-way custom all-reduce hangs or crashes on these H100s -- acceptance run A3
        # died on "RPC call to sample_tokens timed out" with it on.
        disable_custom_all_reduce=True,
    )
    params = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=1.0,
                            max_tokens=MAX_TOKENS)
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
    lo, hi = wilson(effective, n)
    print("\n" + "=" * 66)
    print(f"mean reward              : {statistics.fmean(per_group_mean):.3f}")
    print(f"groups with variance     : {effective}/{n}  ({frac:.1%}, 95% CI {lo:.1%}-{hi:.1%})   <-- usable signal")
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
    print(f"verdict: {verdict}" + ("" if hi - lo < 0.3 else
                                   f"  (wide interval over {n} prompts: raise N_PROMPTS before trusting it)"))


if __name__ == "__main__":
    main()
