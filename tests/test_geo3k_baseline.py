"""infra/geo3k/baseline_eval.py builds vLLM requests the way verl's RL dataset does (REVIEW.md R08).

Reviewer reproduction: an HF Image feature written to parquet and read with pandas yields a dict
{'bytes', 'path'}, which the gate passed straight to vLLM as multi_modal_data (vLLM accepts
images/arrays/tensors), with the raw `<image>` marker left in an unchanged prompt string.
"""
from __future__ import annotations

from pathlib import Path

import datasets
import pandas as pd
import pytest
from PIL import Image

from support import REPO, load_module

B = load_module(REPO / "infra" / "geo3k" / "baseline_eval.py")
PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class FakeProcessor:
    """Renders typed content parts the way a Qwen-VL chat template does."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        out = []
        for m in messages:
            c = m["content"]
            body = c if isinstance(c, str) else "".join(
                PLACEHOLDER if p["type"] == "image" else p["text"] for p in c)
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(out) + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _geo3k_parquet(path: Path, n: int = 3, images_per_row: int = 1) -> Path:
    feats = datasets.Features({
        "data_source": datasets.Value("string"),
        "prompt": [{"role": datasets.Value("string"), "content": datasets.Value("string")}],
        "images": datasets.Sequence(datasets.Image()),
        "reward_model": {"style": datasets.Value("string"), "ground_truth": datasets.Value("string")},
    })
    datasets.Dataset.from_dict({
        "data_source": ["hiyouga/geometry3k"] * n,
        "prompt": [[{"role": "user", "content": "<image>" * images_per_row + f"Find x ({i})."}] for i in range(n)],
        "images": [[Image.new("RGB", (4 + i, 4))] * images_per_row for i in range(n)],
        "reward_model": [{"style": "rule", "ground_truth": "3"}] * n,
    }, features=feats).to_parquet(str(path))
    return path


def test_pandas_really_returns_image_dicts(tmp_path: Path):
    row = pd.read_parquet(_geo3k_parquet(tmp_path / "t.parquet")).iloc[0]
    assert isinstance(row["images"][0], dict) and set(row["images"][0]) >= {"bytes", "path"}


def test_requests_carry_decoded_images_and_the_models_placeholder(tmp_path: Path):
    df = B.load_prompts(str(_geo3k_parquet(tmp_path / "t.parquet")), 3)
    reqs = B.build_requests(df, FakeProcessor())
    assert len(reqs) == 3
    for i, r in enumerate(reqs):
        (img,) = r["multi_modal_data"]["image"]
        assert isinstance(img, Image.Image) and img.mode == "RGB" and img.size == (4 + i, 4)
        assert PLACEHOLDER in r["prompt"] and "<image>" not in r["prompt"]
        assert f"Find x ({i})." in r["prompt"]


def test_multiple_images_keep_their_order(tmp_path: Path):
    df = B.load_prompts(str(_geo3k_parquet(tmp_path / "t.parquet", n=1, images_per_row=2)), 1)
    (req,) = B.build_requests(df, FakeProcessor())
    assert req["prompt"].count(PLACEHOLDER) == 2 and len(req["multi_modal_data"]["image"]) == 2


def test_marker_and_image_counts_must_agree():
    with pytest.raises(ValueError, match="1 <image> marker"):
        B.build_messages([{"role": "user", "content": "<image>Find x."}], 2)


def test_too_few_prompts_is_an_error_not_a_smaller_sample(tmp_path: Path):
    with pytest.raises(SystemExit, match="has 3 rows but N_PROMPTS=64"):
        B.load_prompts(str(_geo3k_parquet(tmp_path / "t.parquet")), 64)


def test_wilson_interval():
    lo, hi = B.wilson(10, 64)
    assert 0.08 < lo < 0.10 and 0.26 < hi < 0.28
    assert B.wilson(0, 0) == (0.0, 1.0)


def test_prep_default_covers_the_gate():
    prep = (REPO / "infra/geo3k/prep_geo3k.py").read_text()
    assert 'N_TEST = int(os.environ.get("N_TEST", "128"))' in prep
