"""W2.1 run identity (REVIEW.md R16): every training run writes to <output_dir>/<RUN_ID>/, resuming
is an explicit choice, the rendezvous is keyed by the run, and evals are named by what they scored.

Before: all six training configs inherited verl's trainer.resume_mode=auto on a fixed output_dir,
so re-running a job could silently resume an earlier run; repeated evals overwrote one file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import ENGINE, REPO, env, load_module, run

cc = load_module(REPO / "scripts" / "compose_check.py")
rc = load_module(ENGINE / "lib" / "run_control.py")
SEARCH = REPO / "usecases/agentic-search/air/4_train.yaml"
ROOT = "/Volumes/main/mshtelma/verl/ckpt/agentic-search-grpo"


def render(tmp: Path, **knobs: str) -> dict:
    return cc.render(SEARCH, tmp, 29600, {"RUN_ID": "r42", **knobs})


def test_default_is_a_fresh_run_in_its_own_directory(tmp_path: Path):
    ov = render(tmp_path)["overrides"]
    assert "trainer.resume_mode=disable" in ov and f"trainer.default_local_dir={ROOT}/r42" in ov


@pytest.mark.parametrize("resume,want", [
    ("auto", ["trainer.resume_mode=auto"]),
    (f"{ROOT}/r42/global_step_20", ["trainer.resume_mode=resume_path", f"trainer.resume_from_path={ROOT}/r42/global_step_20"]),
])
def test_resuming_is_explicit(tmp_path: Path, resume: str, want: list[str]):
    ov = render(tmp_path, RESUME=resume)["overrides"]
    assert all(w in ov for w in want), ov


@pytest.mark.parametrize("knobs,msg", [
    ({"RESUME": "sometimes"}, "expected never | auto"),
    ({"RUN_ID": "a/b"}, "must match"),
    ({"MAX_CKPT_TO_KEEP": "0"}, "MAX_CKPT_TO_KEEP='0': must be >= 1"),     # engine/lib/preflight.py
])
def test_bad_identity_settings_are_refused(tmp_path: Path, knobs: dict, msg: str):
    job = render(tmp_path, **knobs)
    assert job["returncode"] != 0 and msg in job["stderr"]


def test_checkpoint_retention_is_opt_in(tmp_path: Path):
    assert not any("max_actor_ckpt_to_keep" in o for o in render(tmp_path)["overrides"])
    assert "trainer.max_actor_ckpt_to_keep=3" in render(tmp_path, MAX_CKPT_TO_KEEP="3")["overrides"]


def test_the_rendezvous_is_keyed_by_the_run(tmp_path: Path):
    assert f"rdv={tmp_path}/rdv/r42" in render(tmp_path)["stdout"]


def test_ray_actors_rebuild_the_same_rendezvous():
    with env(VOA_RDV_DIR=None, RENDEZVOUS_ROOT="/x", RUN_ID="r42", MASTER_ADDR="10.0.0.1", MASTER_PORT="1"):
        assert rc.rendezvous_dir() == Path("/x/r42")
    with env(VOA_RDV_DIR=None, RENDEZVOUS_ROOT="/x", RUN_ID=None, MASTER_ADDR="10.0.0.1", MASTER_PORT="1"):
        assert rc.rendezvous_dir() == Path("/x/10.0.0.1_1")


def make_dry(*args: str) -> str:
    r = run(["make", "--no-print-directory", "--dry-run", *args, "PIP_INDEX_URL="])
    return r.stdout if r.returncode == 0 else f"RC={r.returncode}\n{r.stdout}"


def test_make_gives_every_submission_an_identity():
    out = make_dry("search-train", "RUN_ID=r42")
    assert "env_variables.RUN_ID=r42" in out and "env_variables.GIT_SHA=" in out and "env_variables.VOA_IMAGE=" in out
    assert "env_variables.RESUME=auto" in make_dry("math-train", "RUN_ID=r42", "RESUME=auto")


def test_make_names_evals_after_what_they_scored():
    out = make_dry("search-eval", f"CKPT={ROOT}/r42/global_step_20", "RUN_ID=e1")
    assert f"env_variables.EVAL_MODEL_PATH={ROOT}/r42/global_step_20" in out
    assert "eval/search_r42-step20_e1.json" in out and "eval/search_r42-step20_e1_traces.jsonl" in out
    assert "eval/search_base_e1.json" in make_dry("search-baseline", "RUN_ID=e1")
    assert "eval/math500_base_e1.json" in make_dry("math-baseline", "RUN_ID=e1")


def test_make_eval_needs_a_checkpoint():
    out = make_dry("search-eval")
    assert out.startswith("RC=") and "set CKPT=" in out


def test_make_index_needs_a_named_warehouse():
    out = make_dry("search-index")
    assert out.startswith("RC=") and "set WAREHOUSE_ID=" in out
    assert "env_variables.QA_VS_WAREHOUSE_ID=wh1" in make_dry("search-index", "WAREHOUSE_ID=wh1", "RUN_ID=r42")


def test_the_run_manifest_ties_the_run_to_its_data(tmp_path: Path):
    dm = load_module(ENGINE / "lib" / "data_manifest.py")
    train = dm.write_parquet(tmp_path / "train.parquet", [{"a": 1}])
    dm.write_manifest(tmp_path / dm.DIR_MANIFEST, tool="prep", sources=[{"hf_id": "org/ds", "revision": "a" * 40}],
                      outputs=[dm.output_record(train, 1)])
    out = tmp_path / "run" / "run_manifest.json"
    r = run(["python3", str(ENGINE / "lib" / "run_manifest.py"), str(out), "--",
             f"data.train_files={train}", f"data.val_files='{tmp_path}/missing.parquet'"])
    assert r.returncode == 0, r.stdout
    data = json.loads(out.read_text())["data"]
    assert data["data.train_files"]["file_matches_manifest"] is True
    assert data["data.train_files"]["sources"] == [{"hf_id": "org/ds", "revision": "a" * 40}]
    assert data["data.val_files"]["path"] == f"{tmp_path}/missing.parquet" and "error" in data["data.val_files"]
