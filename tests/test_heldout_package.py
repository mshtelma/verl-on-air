"""scripts/heldout_package.py: the held-out evidence file pairs each model with the base, per policy."""
from __future__ import annotations

import json
from pathlib import Path

from support import REPO, load_module

hp = load_module(REPO / "scripts" / "heldout_package.py")


def _artifact(path: Path, correct: list[bool], policy: dict) -> None:
    results = [{"uid": str(i), "question": f"q{i}", "status": "scored", "correct": c,
                "hop_type": "2hop" if i % 2 else "3hop1"} for i, c in enumerate(correct)]
    path.write_text(json.dumps({"valid": True, "model": path.stem, "n_scored": len(correct),
                                "em": sum(correct) / len(correct), "answered": len(correct),
                                "mean_tool_calls": 1.0, "eval_policy": policy, "results": results}))


def test_the_package_pairs_what_exists_and_lists_what_is_missing(tmp_path: Path):
    tools, closed = {"version": 2, "tools": ["t"]}, {"version": 2, "closed_book": True}
    base = [True] * 5 + [False] * 5
    _artifact(tmp_path / "test_base_tools.json", base, tools)
    _artifact(tmp_path / "test_pureem_step20_tools.json", [True] * 7 + [False] * 3, tools)
    _artifact(tmp_path / "test_base_closedbook.json", [False] * 10, closed)
    _artifact(tmp_path / "test_pureem_step20_closedbook.json", [True] + [False] * 9, closed)
    out = tmp_path / "pkg.json"
    assert hp.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    t = doc["policies"]["tools"]
    assert t["missing"] == ["test_seed7_step20_tools.json"]
    (c,) = t["checkpoints"]
    assert c["label"] == "pureem_step20" and c["paired"]["gained"] == 2 and c["paired"]["lost"] == 0
    assert c["by_hops"] == {"2hop": {"n": 5, "correct": 3}, "3hop": {"n": 5, "correct": 4}}
    assert doc["policies"]["closed_book"]["checkpoints"][0]["paired"]["gained"] == 1
    assert doc["split"]["test"]["n"] == 500
