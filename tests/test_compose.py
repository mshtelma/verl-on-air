"""Every training job's REAL emitted overrides compose against the pinned verl, with invariants.

The same check `make compose-check` runs; kept here so topology/reward-wiring regressions fail
`make test` too, and so negative cases (a bad override) can be asserted directly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from support import REPO, load_module

cc = load_module(REPO / "scripts" / "compose_check.py")


@pytest.fixture(scope="module")
def verl_src() -> Path:
    return cc.ensure_verl_src()


@pytest.mark.parametrize("job", cc.training_jobs(), ids=lambda p: str(p.relative_to(REPO)))
def test_training_job_composes_with_invariants(verl_src, job, tmp_path):
    res = cc.check_job(verl_src, job, tmp_path, 29400)
    assert res["violations"] == [], res["violations"]


def test_training_jobs_are_discovered():
    names = {p.name for p in cc.training_jobs()}
    assert {"4_train.yaml", "rung1_2b_fsdp_8gpu.yaml"} <= names
