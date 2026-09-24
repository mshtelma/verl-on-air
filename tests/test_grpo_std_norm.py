"""NORM_ADV_BY_STD_IN_GRPO: explained correctly, set explicitly where it matters, parsed strictly (R14).

Before: the docs said std-normalisation "collapses a graded reward to binary" (it does not: it keeps
a group's order and gaps), advertised False for math while the math job never set it (it ran
True), and the launcher treated any value but the exact string "False" as "use the default"."""
from __future__ import annotations

from pathlib import Path

import pytest

from support import REPO, load_module

cc = load_module(REPO / "scripts" / "compose_check.py")
MATH = REPO / "usecases/math/air/4_train.yaml"
SEARCH_SYNC = REPO / "usecases/agentic-search/air/4_train_sync.yaml"


def test_the_documented_example_is_what_verl_computes():
    # verl's compute_grpo_outcome_advantage: (r - mean) / (torch.std(r) + 1e-6); torch.std is unbiased
    r = [0.0, 0.05, 0.7, 1.0]
    m = sum(r) / len(r)
    sd = (sum((x - m) ** 2 for x in r) / (len(r) - 1)) ** 0.5
    shown = "[" + ", ".join(f"{(x - m) / (sd + 1e-6):.2f}".replace("-", "−") for x in r) + "]"
    assert shown == "[−0.89, −0.79, 0.53, 1.14]"          # order and gaps kept: not binary
    for doc in ("docs/tuning.md", "docs/configuration.md"):
        assert shown in (REPO / doc).read_text(), doc


def test_no_doc_still_claims_it_collapses_a_graded_reward():
    for p in [*REPO.glob("docs/*.md"), *REPO.glob("usecases/*/README.md"), *REPO.glob("usecases/*/air/*.yaml")]:
        text = p.read_text().lower()
        assert "collapses to binary" not in text and "collapses a graded reward" not in text, p


def test_the_math_job_says_what_it_runs(tmp_path: Path):
    ov = cc.render(MATH, tmp_path, 29700, {"RUN_ID": "r1"})["overrides"]
    assert "algorithm.norm_adv_by_std_in_grpo=True" in ov


@pytest.mark.parametrize("job", [MATH, SEARCH_SYNC], ids=["async", "sync"])
def test_a_non_boolean_stops_either_launcher_and_spellings_normalise(tmp_path: Path, job: Path):
    out = cc.render(job, tmp_path, 29710, {"RUN_ID": "r1", "NORM_ADV_BY_STD_IN_GRPO": "maybe"})
    assert out["returncode"] != 0 and "NORM_ADV_BY_STD_IN_GRPO='maybe' is not a boolean" in out["stderr"]
    for spelling in ("False", "false", "0"):   # engine/lib/preflight.py normalises every boolean knob
        good = cc.render(job, tmp_path, 29720, {"RUN_ID": "r1", "NORM_ADV_BY_STD_IN_GRPO": spelling})
        assert "algorithm.norm_adv_by_std_in_grpo=False" in good["overrides"], spelling
