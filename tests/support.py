"""Shared helpers for the CPU test suite (no GPU, no cloud).

Use-case modules import their siblings by bare name (``import reward``, ``import tool``), exactly as
they do inside a job. Both use cases have a ``reward.py``, so a test must never import one by bare
name: ``load_usecase()`` loads a fresh copy under a unique name, resolves its siblings from that use
case's own directory, and leaves ``sys.modules`` as it found it.

Shell entrypoints are exercised for real, with GPU/cloud binaries (vllm, ray, air, docker, ...)
replaced by recording stubs on PATH -- see ``StubBin``.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import itertools
import json
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


# verl packages whose __init__ imports torch or ray. Each becomes an empty package over its real
# directory, so `import verl.x.y` still loads verl's own y.py.
_VERL_PACKAGES = ("verl", "verl.tools", "verl.utils", "verl.utils.reward_score", "verl.experimental",
                  "verl.experimental.agent_loop")


def _verl_get_event_loop():
    """verl.utils.ray_utils.get_event_loop, verbatim -- the one thing the tool parser needs from a
    module that imports ray."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@contextlib.contextmanager
def pinned_verl():
    """verl's own modules at the pinned commit (the checkout scripts/compose_check.py keeps in
    .cache/), importable on CPU for the duration. Only the package __init__s that pull in torch or
    ray are replaced (by empty packages over the real directories), plus two stand-ins: ray_utils'
    get_event_loop and utils.metric's Metric. What a test imports -- the tool parser,
    @function_tool and its schemas, prime_math -- is verl's real code. On exit every verl module,
    and with it verl's tool registry, is dropped again."""
    root = load_module(REPO / "scripts" / "compose_check.py").ensure_verl_src() / "verl"
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "verl" or k.startswith("verl.")}
    try:
        for name in _VERL_PACKAGES:
            pkg = ModuleType(name)
            pkg.__path__ = [str(root.joinpath(*name.split(".")[1:]))]
            sys.modules[name] = pkg
        stand_ins = {"verl.utils.ray_utils": {"get_event_loop": _verl_get_event_loop},
                     "verl.utils.metric": {"Metric": type("Metric", (), {})}}
        for name, attrs in stand_ins.items():
            mod = ModuleType(name)
            mod.__dict__.update(attrs)
            sys.modules[name] = mod
        yield root
    finally:
        for k in [k for k in sys.modules if k == "verl" or k.startswith("verl.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def run(args: list[str], *, env: dict[str, str] | None = None, cwd: str | Path | None = None,
        timeout: float = 120, input: str | None = None) -> subprocess.CompletedProcess:
    """Run a command with stdout+stderr merged (assert on ``.returncode`` / ``.stdout``)."""
    e = dict(os.environ if env is None else env)
    if PY_BIN not in e.get("PATH", "").split(os.pathsep):  # StubBin.env() already placed it
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


# --- fake model / checkpoint trees (the real verl fully-async + mbridge layout) --------------
def fake_hf_model(d: Path, *, shards: int = 2, shard_bytes: int = 64) -> Path:
    """A minimal HuggingFace model dir: config, tokenizer, an index and its shards."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}))
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text("{}")
    names = [f"model.safetensors-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    for n in names:
        (d / n).write_bytes(b"x" * shard_bytes)
    (d / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": shard_bytes * shards - 16},  # headers make shards a bit larger
        "weight_map": {f"layer.{i}.weight": n for i, n in enumerate(names)}}))
    return d


def fake_train_checkpoint(run_dir: Path, step: int, *, manifest: bool = True, hf: bool = True) -> Path:
    """<run_dir>/global_step_<step>/actor/... as verl writes it. manifest=False reproduces an
    interrupted save (actor/{extra,model,optimizer} exist, no ckpt_contents.json); hf=False with
    manifest=True reproduces the mkdir'd-but-empty model/huggingface/ side effect."""
    actor = run_dir / f"global_step_{step}" / "actor"
    for sub in ("extra", "optimizer", "model/huggingface"):
        (actor / sub).mkdir(parents=True, exist_ok=True)
    if hf:
        fake_hf_model(actor / "model" / "huggingface")
    if manifest:
        (actor / "ckpt_contents.json").write_text(json.dumps({
            "global_step": step, "role": "actor", "schema_version": 2,
            "contents": {"model": {"backend": "mbridge", "format": "huggingface",
                                   "path": "model/huggingface"}}}))
    return run_dir / f"global_step_{step}"


# --- a fake OpenAI-compatible server (judge / vLLM stand-in), on its own thread -----------------
class FakeOpenAIServer:
    """``with FakeOpenAIServer(reply) as srv: ... srv.url ...``. ``reply(path, payload, n)`` ->
    ``(http_status, json_body, delay_s)``, where n counts requests so far (1-based). Runs in a
    background thread with its own event loop, so code under test may call asyncio.run()."""

    def __init__(self, reply):
        import asyncio
        import threading

        self.reply = reply
        self.requests: list[tuple[str, dict]] = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self.port = 0

    def _serve(self) -> None:
        import asyncio

        from aiohttp import web

        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.port = self._runner.addresses[0][1]
        self._ready.set()
        self._loop.run_forever()

    async def _handle(self, request):
        import asyncio

        from aiohttp import web

        payload = await request.json() if request.can_read_body else {}
        self.requests.append((request.path, payload))
        status, body, delay = self.reply(request.path, payload, len(self.requests))
        if delay:
            await asyncio.sleep(delay)
        return web.json_response(body, status=status)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "FakeOpenAIServer":
        self._thread.start()
        assert self._ready.wait(10), "fake server did not start"
        return self

    def __exit__(self, *exc) -> None:
        import asyncio

        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


def chat_completion(content: str | None, finish_reason: str = "stop", reasoning: str | None = None) -> dict:
    return {"choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}]}
