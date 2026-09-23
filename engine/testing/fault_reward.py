"""ACCEPTANCE TESTING ONLY: a reward that fails on purpose (acceptance run A5a).

Wraps a real use-case reward and raises after ``FAULT_AFTER_CALLS`` calls in this process. In
the fully-async recipe the exception is swallowed by the Rollouter (``asyncio.gather(...,
return_exceptions=True)``) and verl ends the run as a *normal* stop that exits 0 -- the path the
completion certificate exists for. A5a passes when the launcher nevertheless reports FAILED.

    CUSTOM_REWARD_PATH: engine/testing/fault_reward.py
    FAULT_REWARD_PATH:  usecases/agentic-search/reward.py    # the reward to wrap (repo-relative)
    FAULT_AFTER_CALLS:  '64'                                  # per reward worker process

Never point a real training job at this file.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_calls = 0
_real = None


def _wrapped():
    global _real
    if _real is None:
        rel = os.environ.get("FAULT_REWARD_PATH", "")
        path = Path(rel) if Path(rel).is_absolute() else _REPO / rel
        if not rel or not path.is_file():
            raise RuntimeError(f"FAULT_REWARD_PATH={rel!r} is not a reward file")
        spec = importlib.util.spec_from_file_location("_fault_wrapped_reward", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _real = mod.compute_score
    return _real


def compute_score(*args: Any, **kwargs: Any) -> Any:
    global _calls
    with _lock:
        _calls += 1
        n = _calls
    after = int(os.environ.get("FAULT_AFTER_CALLS", "0"))
    if n > after:
        raise RuntimeError(f"injected reward failure (acceptance A5a): call {n} > FAULT_AFTER_CALLS={after}")
    return _wrapped()(*args, **kwargs)
