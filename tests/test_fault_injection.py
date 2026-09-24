"""The acceptance-only fault injectors (engine/testing/): they fail exactly when told to.

A5a (a reward that raises after N calls -- the Rollouter swallows it and verl exits 0) and A5b
(the Trainer killed after its first save) are GPU runs; what they prove depends on the injectors
behaving, which is checked here on CPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from support import ENGINE, REPO, env, load_module, run


def _fault_reward():
    return load_module(ENGINE / "testing" / "fault_reward.py")


def test_the_fault_reward_passes_through_then_raises():
    with env(FAULT_REWARD_PATH="usecases/agentic-search/reward.py", FAULT_AFTER_CALLS="2"):
        fr = _fault_reward()
        traj = "<answer>Paris</answer>"
        spans = {"role_spans": [["assistant", 0, len(traj)]]}
        for _ in range(2):
            out = fr.compute_score("musique", traj, {"target": ["Paris"]}, spans)
            assert out["score"] == 1.0 and out["provenance_ok"] == 1.0, out
        with pytest.raises(RuntimeError, match="injected reward failure"):
            fr.compute_score("musique", traj, {"target": ["Paris"]}, spans)


def test_the_fault_reward_refuses_a_missing_target():
    with env(FAULT_REWARD_PATH="usecases/nope.py", FAULT_AFTER_CALLS="5"):
        with pytest.raises(RuntimeError, match="not a reward file"):
            _fault_reward().compute_score("musique", "", {"target": ["x"]}, {})


def test_fault_inject_does_nothing_without_the_marker(tmp_path: Path):
    r = run(["python3", str(ENGINE / "testing" / "fault_inject.py"), "kill-trainer-after-save",
             str(tmp_path), "--timeout-s", "0"])
    assert r.returncode == 1 and "nothing injected" in r.stdout


def test_an_unknown_fault_is_refused_before_anything_runs():
    pf = load_module(ENGINE / "lib" / "preflight.py")
    problems, _ = pf.check_knobs("async", {"FAULT_INJECT": "kill-everything"})
    assert problems and "FAULT_INJECT" in problems[0]
    problems, _ = pf.check_knobs("sync", {"FAULT_INJECT": "kill-trainer-after-save"})
    assert problems, "the sync launcher has no injector; the knob must not be silently ignored"
    assert REPO.joinpath("engine/testing/fault_inject.py").is_file()
