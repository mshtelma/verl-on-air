"""Reward variation must use rollout groups, even when a question is repeated."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib

from support import REPO, load_module, run


def test_repeated_question_keeps_distinct_rollout_groups(tmp_path):
    exporter = load_module(REPO / "usecases/math/export_run.py")
    for i, (group, version, correct) in enumerate([("a", 0, 0), ("a", 0, 1), ("b", 1, 1), ("b", 1, 1)]):
        record = {"episode": {"rollout_group_id": group, "parameter_version": version},
                  "index": 17, "metrics": {"score": correct, "acc": correct,
                                           "judge_valid": 1, "judge_fallback": 0,
                                           "judge_input_truncated": 0}}
        (tmp_path / f"{i}.json").write_text(json.dumps(record))
    result = exporter.training_signal(tmp_path)
    assert result["trajectories"] == 4 and len(result["groups"]) == 2
    assert result["group_sizes"] == {2: 2}
    assert result["mixed_reward_groups"] == result["mixed_correctness_groups"] == 1
    assert result["by_parameter_version"]["0"]["answer_accuracy"] == 0.5
    assert result["by_parameter_version"]["1"]["answer_accuracy"] == 1


def test_log_export_preserves_eval_artifact_and_checksum(tmp_path):
    path = tmp_path / "eval.json"
    artifact = {"valid": True, "n": 2, "results": [{"idx": 1, "correct": True}, {"idx": 2, "correct": False}]}
    path.write_text(json.dumps(artifact))
    result = run(["python3", str(REPO / "usecases/math/export_run.py"), "--eval-out", str(path)])
    assert result.returncode == 0
    envelopes = [json.loads(line) for line in result.stdout.splitlines()]
    chunks = sorted((x for x in envelopes if "math_artifact_chunk" in x), key=lambda x: x["math_artifact_chunk"])
    raw = zlib.decompress(base64.b64decode("".join(x["data"] for x in chunks)))
    footer = envelopes[-1]
    assert footer["sha256"] == hashlib.sha256(raw).hexdigest() and footer["bytes"] == len(raw)
    payload = json.loads(raw)
    assert payload["files"]["eval.json"] == artifact
    assert payload["receipts"]["eval.json"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
