#!/usr/bin/env python3
"""Verify the full official training split, or freeze a pilot's mixed-reward groups."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402


def mixed_training_uids(artifact: dict, allowed: set[str], *, maximum=8) -> list[str]:
    if artifact.get("valid") is not True:
        raise ValueError("baseline evaluation is invalid")
    samples = artifact["eval_policy"]["samples_per_question"]
    groups = {}
    for row in artifact["results"]:
        if row["split"] != "train_probe":
            continue
        if row["uid"] not in allowed:
            raise ValueError("baseline train_probe contains a question outside the training pilot")
        if row["status"] != "scored" or row["reward"] not in (0.0, 1.0):
            raise ValueError("baseline has unresolved/non-binary training rewards")
        groups.setdefault(row["uid"], []).append(row["reward"])
    if any(len(values) != samples for values in groups.values()):
        raise ValueError("baseline has incomplete training groups")
    selected = [uid for uid, values in groups.items() if min(values) < max(values)]
    selected.sort(key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())
    if not selected:
        raise ValueError("baseline has no mixed training reward groups; do not spend a training run on zero task advantages")
    return selected[:maximum]


def official_training_receipt(source: Path, hyperparameters: dict) -> dict:
    """Use all 203 train questions, without reading held-out answers or outcomes."""
    import pyarrow.parquet as pq
    path = source / "train.parquet"
    manifest_path = source / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = next(row for row in manifest["outputs"] if row["name"] == path.name)
    if Path(hyperparameters["train_files"]) != path:
        raise ValueError("official training must use the frozen train.parquet")
    if dm.sha256_file(path) != receipt["sha256"]:
        raise ValueError("official training data checksum mismatch")
    rows = pq.read_table(path).to_pylist()
    uids = [row["extra_info"]["uid"] for row in rows]
    split = manifest["split"]
    train_ids, heldout_ids = set(split["train_ids"]), set(split["heldout_ids"])
    if len(uids) != 203 or len(set(uids)) != 203 or set(uids) != train_ids or len(heldout_ids) != 43:
        raise ValueError("official training split must contain 203 unique train questions and 43 held-out IDs")
    if train_ids & heldout_ids:
        raise ValueError("official train/held-out UID overlap")
    difficulties = [row["extra_info"]["difficulty"] for row in rows]
    if difficulties.count("easy") != 113 or difficulties.count("hard") != 90:
        raise ValueError("official training must contain 113 easy and 90 hard questions")
    if len(rows) * hyperparameters["total_epochs"] <= hyperparameters["total_rollout_steps"]:
        raise ValueError("official data/epoch budget is too small for the requested rollout groups")
    return {"sources": [{"path": str(path), "sha256": receipt["sha256"]},
                        {"path": str(manifest_path), "sha256": dm.sha256_file(manifest_path)}],
            "outputs": [receipt], "selected_training_uids": uids,
            "selection": "all_official_training_questions", "heldout_used_for_selection": False,
            "split": {"easy_train": 113, "hard_train": 90, "hard_heldout": 43}}


def main():
    import pyarrow.parquet as pq
    import yaml
    root = Path(os.environ["OQ_ARTIFACT_ROOT"]) / os.environ["RUN_ID"] / "signal-data"
    target = root / "train.parquet"
    manifest_path = root / "DATA_MANIFEST.json"
    hyperparameters = yaml.safe_load(Path(os.environ["HYPERPARAMETERS_PATH"]).read_text())
    selection = os.environ.get("OQ_TRAIN_SELECTION", "mixed_pilot")
    if selection == "official":
        receipt = official_training_receipt(Path(os.environ["OQ_DATA_DIR"]), hyperparameters)
        root.mkdir(parents=True, exist_ok=True)
        if int(os.environ.get("POD_RANK", "0")) == 0:
            if manifest_path.exists():
                raise ValueError("official training receipt already exists; use a new RUN_ID")
            dm.write_manifest(manifest_path, tool="officeqa/select_train.py", **receipt)
        print(json.dumps({"officeqa_training_signal": "PASS", "selection": selection,
                          "rows": 203, "hard_train": 90, "heldout_used": False}), flush=True)
        return
    if selection != "mixed_pilot":
        raise ValueError("OQ_TRAIN_SELECTION must be official or mixed_pilot")
    baseline = Path(os.environ["OQ_BASELINE_OUT"])
    if Path(hyperparameters["train_files"]) != target:
        raise ValueError(f"parameters.train_files must be {target}")
    if int(os.environ.get("POD_RANK", "0")) != 0:
        deadline = time.monotonic() + 300
        while not manifest_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("rank 0 did not freeze the pilot's training data")
            time.sleep(2)
        receipt = json.loads(manifest_path.read_text())["outputs"][0]
        if dm.sha256_file(target) != receipt["sha256"]:
            raise ValueError("selected training data checksum mismatch")
        return
    if manifest_path.exists() or target.exists():
        raise ValueError("selected pilot data already exists; use a new RUN_ID")
    source = Path(os.environ["OQ_DATA_DIR"]) / "pilot_train.parquet"
    rows = pq.read_table(source).to_pylist()
    artifact = json.loads(baseline.read_text())
    policy = artifact["eval_policy"]
    expected = {"max_turns": int(os.environ["MAX_TURNS"]),
                "generation_tokens_per_turn": int(os.environ["OQ_MAX_GENERATION_TOKENS"]),
                "prompt_tokens": hyperparameters["max_prompt_length"],
                "tool_max_chars": int(os.environ["MAX_TOOL_RESPONSE_LEN"]),
                "samples_per_question": hyperparameters["rollout_n"],
                "reward": "binary_correct_answer_and_supported_path"}
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"baseline/training policy mismatch: {key}")
    manifest = Path(os.environ["OQ_DATA_DIR"]) / "DATA_MANIFEST.json"
    if artifact["dataset"]["manifest"]["sha256"] != dm.sha256_file(manifest):
        raise ValueError("baseline uses a different OfficeQA snapshot")
    selected = mixed_training_uids(artifact, {row["extra_info"]["uid"] for row in rows})
    selected_rows = [row for row in rows if row["extra_info"]["uid"] in selected]
    if len(selected_rows) * hyperparameters["total_epochs"] < hyperparameters["total_rollout_steps"]:
        raise ValueError("selected data/epoch budget cannot supply the requested rollout groups")
    root.mkdir(parents=True, exist_ok=True)
    dm.write_parquet(target, selected_rows)
    dm.write_manifest(manifest_path, tool="officeqa/select_train.py",
                      sources=[{"path": str(baseline), "sha256": dm.sha256_file(baseline)},
                               {"path": str(source), "sha256": dm.sha256_file(source)}],
                      outputs=[dm.output_record(target, len(selected_rows))],
                      selected_training_uids=selected, selection="mixed_training_reward_only",
                      heldout_used_for_selection=False, baseline_policy=policy)
    print(json.dumps({"officeqa_training_signal": "PASS", "selected_training_uids": selected,
                      "rows": len(selected_rows), "train_file": str(target), "heldout_used": False}), flush=True)


if __name__ == "__main__":
    main()
