"""Shared helpers for the CPU test suite (no GPU, no cloud).

Use-case modules import their siblings by bare name (``import reward``, ``import tool``), exactly as
they do inside a job. Both use cases have a ``reward.py``, so a test must never import one by bare
name: ``load_usecase()`` loads a fresh copy under a unique name, resolves its siblings from that use
case's own directory, and leaves ``sys.modules`` as it found it.

Shell entrypoints are exercised for real, with GPU/cloud binaries (vllm, ray, air, docker, ...)
replaced by recording stubs on PATH -- see ``StubBin``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
USECASES = REPO / "usecases"
ENGINE = REPO / "engine"
# The interpreter running the tests has the dev toolchain (pyyaml, ...); put it first on PATH so
# the scripts' own `python3` calls resolve to it rather than to a bare system python.
PY_BIN = str(Path(sys.executable).parent)
_uniq = itertools.count()


@contextlib.contextmanager
def env(**overrides: str | None):
    """Temporarily set (str) or unset (None) environment variables."""
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_module(path: str | Path, *, search_dir: str | Path | None = None,
                env_overrides: dict[str, str | None] | None = None) -> ModuleType:
    """Execute ``path`` as a fresh module with ``search_dir`` (default: its own dir) first on
    sys.path, so its bare sibling imports resolve there. Module-level env reads see
    ``env_overrides``. Nothing named after a sibling is left behind in sys.modules."""
    path = Path(path)
    search_dir = Path(search_dir or path.parent)
    siblings = {p.stem for p in search_dir.glob("*.py")}
    saved = {n: sys.modules.pop(n) for n in siblings if n in sys.modules}
    name = f"_voa_test_{next(_uniq)}_{path.stem}"
    sys.path.insert(0, str(search_dir))
    try:
        with env(**(env_overrides or {})):
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec and spec.loader, path
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(search_dir))
        for n in siblings:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def load_usecase(usecase: str, module: str, **env_overrides: str | None) -> ModuleType:
    """``load_usecase("math", "reward", JUDGE_DEBUG="0")`` -> a fresh usecases/math/reward.py."""
    d = USECASES / usecase
    return load_module(d / f"{module}.py", search_dir=d, env_overrides=env_overrides)


def run(args: list[str], *, env: dict[str, str] | None = None, cwd: str | Path | None = None,
        timeout: float = 120, input: str | None = None) -> subprocess.CompletedProcess:
    """Run a command with stdout+stderr merged (assert on ``.returncode`` / ``.stdout``)."""
    e = dict(os.environ if env is None else env)
    e["PATH"] = PY_BIN + os.pathsep + e.get("PATH", "")
    return subprocess.run(args, cwd=str(cwd or REPO), env=e, text=True, input=input,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


class StubBin:
    """A directory of fake executables that shadow real ones on PATH. Every call is appended to
    ``calls.log`` as ``<name> <args...>`` so a test can assert what did (or did not) run."""

    def __init__(self, root: Path):
        self.dir = root / "stub-bin"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = root / "stub-calls.log"
        self.log.touch()

    def add(self, name: str, body: str = "exit 0") -> Path:
        p = self.dir / name
        p.write_text("#!/usr/bin/env bash\n"
                     f"printf '%s\\n' \"{name} $*\" >> '{self.log}'\n"
                     f"{body}\n")
        p.chmod(0o755)
        return p

    def calls(self, name: str | None = None) -> list[str]:
        lines = self.log.read_text().splitlines()
        return [ln for ln in lines if name is None or ln.split(" ", 1)[0] == name]

    def env(self, base: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
        e = dict(os.environ if base is None else base)
        e["PATH"] = f"{self.dir}{os.pathsep}{PY_BIN}{os.pathsep}{e.get('PATH', '')}"
        e.update(extra)
        return e
