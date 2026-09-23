"""engine/train/dispatch_agentic.sh: node roles, mode selection and plugin-path resolution."""
from __future__ import annotations

from pathlib import Path

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
