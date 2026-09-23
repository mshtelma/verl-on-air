"""engine/train/dispatch_agentic.sh: node roles, mode selection and plugin-path resolution."""
from __future__ import annotations

from pathlib import Path

import yaml

from support import REPO, load_module

cc = load_module(REPO / "scripts" / "compose_check.py")
SEARCH_TRAIN = REPO / "usecases/agentic-search/air/4_train.yaml"


def test_plugin_paths_resolve_from_literal_env_values(tmp_path: Path):
    job = cc.render(SEARCH_TRAIN, tmp_path, 29300)
    assert job["returncode"] == 0, job["stdout"][-2000:]
    assert f"reward.custom_reward_function.path={REPO}/usecases/agentic-search/reward.py" in job["overrides"]


def test_a_missing_plugin_file_fails_before_any_role_starts(tmp_path: Path):
    job = cc.render(SEARCH_TRAIN, tmp_path, 29301,
                    {"FUNCTION_TOOL_PATH": "${CODE_SOURCE_PATH}/usecases/agentic-search/no_such_tool.py"})
    assert job["returncode"] != 0
    assert f"FUNCTION_TOOL_PATH does not exist: {REPO}/usecases/agentic-search/no_such_tool.py" in job["stderr"]
    assert not job["overrides"]


# --- R03: node roles must match the job's intent -------------------------------------------------
SEARCH_SYNC = REPO / "usecases/agentic-search/air/4_train_sync.yaml"
MATH_TRAIN = REPO / "usecases/math/air/4_train.yaml"
# the override the README used to advertise as "one env var" for switching to sync
README_SYNC_SWITCH = {"TRAIN_MODE": "sync", "ROLLOUT_NNODES": "0", "NUM_NODES": "4"}


def test_readme_sync_switch_is_refused_on_every_rank(tmp_path: Path):
    for rank in ("0", "2"):
        job = cc.render(SEARCH_TRAIN, tmp_path, 29310, {**README_SYNC_SWITCH, "POD_RANK": rank, "NODE_RANK": rank})
        assert job["returncode"] != 0, rank
        assert "leaves 2 node(s) to serve an LLM judge, but no judge is configured" in job["stderr"]
        assert not job["overrides"], "a training role started anyway"


def test_judge_model_without_judge_nodes_is_refused(tmp_path: Path):
    job = cc.render(SEARCH_TRAIN, tmp_path, 29311, {"JUDGE_MODEL_PATH": "/Volumes/x/models/judge"})
    assert job["returncode"] != 0 and "leaves no node to serve it" in job["stderr"]


def test_sync_with_a_colocated_judge_is_refused(tmp_path: Path):
    job = cc.render(MATH_TRAIN, tmp_path, 29312, {"TRAIN_MODE": "sync", "ROLLOUT_NNODES": "0"})
    assert job["returncode"] != 0 and "has never been run" in job["stderr"]


def test_sync_without_an_explicit_step_budget_is_refused(tmp_path: Path):
    spec = yaml.safe_load(SEARCH_SYNC.read_text())
    del spec["parameters"]["total_training_steps"]
    no_budget = tmp_path / "4_train_sync.yaml"
    no_budget.write_text(yaml.safe_dump(spec))
    job = cc.render(no_budget, tmp_path, 29313)
    assert job["returncode"] != 0 and "needs an explicit parameters.total_training_steps" in job["stderr"]


def test_sync_recipe_trains_on_all_nodes_with_the_async_budget(tmp_path: Path):
    job = cc.render(SEARCH_SYNC, tmp_path, 29314)
    assert job["returncode"] == 0, job["stderr"][-2000:]
    ov = job["overrides"]
    assert "trainer.nnodes=4" in ov and "trainer.total_training_steps=100" in ov
    assert f"reward.custom_reward_function.path={REPO}/usecases/agentic-search/reward.py" in ov
    assert "data.shuffle=True" in ov


# --- PRE_TRAIN_CHECK: a use-case gate between "judge is up" and "training starts" ----------------
def test_math_job_resolves_its_judge_selfcheck(tmp_path: Path):
    job = cc.render(MATH_TRAIN, tmp_path, 29320)
    assert job["returncode"] == 0, job["stderr"][-2000:]
    assert f"pre-train check {REPO}/usecases/math/judge_selfcheck.py resolved (not run)" in job["stdout"]


def test_a_missing_pre_train_check_is_fatal(tmp_path: Path):
    job = cc.render(SEARCH_TRAIN, tmp_path, 29321, {"PRE_TRAIN_CHECK": "${CODE_SOURCE_PATH}/nope.py"})
    assert job["returncode"] != 0 and "PRE_TRAIN_CHECK does not exist" in job["stderr"]


def test_a_failing_pre_train_check_stops_the_job_before_training(tmp_path: Path):
    check = tmp_path / "check.py"
    check.write_text("import sys; print('calibration: 3/8'); sys.exit(1)\n")
    job = cc.render(SEARCH_TRAIN, tmp_path, 29322, {"PRE_TRAIN_CHECK": str(check), "DRY_RUN": "0"})
    assert job["returncode"] != 0 and "pre-train check failed" in job["stderr"]
    assert "calibration: 3/8" in job["stdout"]
    assert "-> run_grpo_fully_async.sh" not in job["stdout"], "training started anyway"
