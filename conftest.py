"""Root pytest config: makes tests/support.py importable and provides the shared fixtures."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from support import REPO, StubBin  # noqa: E402


@pytest.fixture
def stub_bin(tmp_path: Path) -> StubBin:
    """Recording fake executables on PATH (vllm, ray, air, docker, databricks, ...)."""
    return StubBin(tmp_path)


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    """A disposable copy of the working tree, for scripts that rewrite files in place."""
    dst = tmp_path / "repo"
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(
        ".git", ".venv", ".cache", "logs", "__pycache__", ".idea", ".pytest_cache", "*.whl"))
    return dst
