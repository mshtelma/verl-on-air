import hashlib
import json

import pytest

from select_train import mixed_training_uids, official_training_receipt


def artifact(train_values, heldout_values=(0, 1, 0, 1)):
    return {"valid": True, "eval_policy": {"samples_per_question": 4}, "results": [
        {"uid": uid, "split": split, "status": "scored", "reward": value}
        for uid, split, values in [("train", "train_probe", train_values), ("heldout", "heldout", heldout_values)]
        for value in values
    ]}


def test_selects_only_mixed_training_questions():
    assert mixed_training_uids(artifact((0, 1, 0, 0)), {"train"}) == ["train"]


def test_heldout_success_cannot_rescue_zero_training_signal():
    with pytest.raises(ValueError, match="no mixed training"):
        mixed_training_uids(artifact((0, 0, 0, 0)), {"train"})


def test_wrong_split_or_incomplete_groups_are_rejected():
    with pytest.raises(ValueError, match="outside the training pilot"):
        mixed_training_uids(artifact((0, 1, 0, 0)), {"different"})
    with pytest.raises(ValueError, match="incomplete"):
        mixed_training_uids(artifact((0, 1)), {"train"})


def test_official_training_cannot_admit_a_heldout_question(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "train.parquet"
    rows = [{"extra_info": {"uid": f"q{i}", "difficulty": "easy" if i < 113 else "hard"}} for i in range(203)]
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = {"outputs": [{"name": path.name, "rows": 203, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
                "split": {"train_ids": [f"q{i}" for i in range(203)], "heldout_ids": [f"h{i}" for i in range(43)]}}
    manifest_path = tmp_path / "DATA_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    parameters = {"train_files": str(path), "total_epochs": 10, "total_rollout_steps": 400}
    receipt = official_training_receipt(tmp_path, parameters)
    assert len(receipt["selected_training_uids"]) == 203
    assert receipt["heldout_used_for_selection"] is False
    manifest["split"]["heldout_ids"][0] = "q202"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        official_training_receipt(tmp_path, parameters)
