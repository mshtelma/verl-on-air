#!/usr/bin/env python3
"""Read bounded pilot evidence on AIR when external Volume downloads are unavailable.

The compressed JSON envelope can be recovered verbatim from the AIR log. This job
only reads named result files and a few episode traces; it never loads a model.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path


def representative(episodes, limit=3):
    """Keep varied questions/terminal outcomes instead of repeats of one group."""
    groups = {}
    for row in episodes:
        key = (row.get("uid"), row.get("record", {}).get("termination"),
               row.get("metrics", {}).get("report_valid"))
        groups.setdefault(key, row)
    return list(groups.values())[:limit]


def main():
    root = Path(os.environ["OQ_EXPORT_DIR"])
    if not root.is_dir():
        raise FileNotFoundError(root)
    payload = {"root": str(root), "files": {}, "receipts": {}, "selected_episodes": []}
    for name in ["eval.json", "calibration.json", "stage-0.json", "stage-1.json", "signal-data/DATA_MANIFEST.json"]:
        path = root / name
        if path.is_file():
            raw = path.read_bytes()
            payload["files"][name] = json.loads(raw)
            payload["receipts"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if (root / "eval.json.parts").is_dir():
        episodes = [json.loads(path.read_text()) for path in sorted((root / "eval.json.parts").glob("*.json"))]
        successes = [row for row in episodes if row.get("reward", 0) > 0]
        failures = [row for row in episodes if row.get("reward", 0) == 0 and row.get("split") == "train_probe"]
        payload["selected_episodes"] = representative(successes) + representative(failures)
    groups = defaultdict(list)
    training_traces = []
    for path in sorted((root / "reward-traces").glob("*.json")):
        trace = json.loads(path.read_text())
        record = trace.get("record") or {}
        group = record.get("rollout_group_id")
        if group:
            training_traces.append(trace)
            groups[group].append({"score": trace["metrics"]["score"], "question_uid": record.get("question_uid"),
                                  "parameter_version": record.get("parameter_version"),
                                  "infrastructure_error": trace["metrics"]["infrastructure_error"]})
    if groups:
        payload["training_signal"] = {
            "trajectories": len(training_traces), "groups": dict(groups),
            "mixed_groups": sum(len({row["score"] for row in rows}) > 1 for rows in groups.values()),
            "successful_trajectories": sum(trace["metrics"]["score"] > 0 for trace in training_traces),
            "infrastructure_errors": sum(trace["metrics"]["infrastructure_error"] for trace in training_traces),
        }
        payload["selected_training_traces"] = [trace for trace in training_traces if trace["metrics"]["score"] > 0][:3]
    checkpoint = os.environ.get("OQ_EXPORT_CKPT_DIR")
    if checkpoint:
        for name in ["run_result.json", "RUN_MANIFEST.json"]:
            path = Path(checkpoint) / name
            if path.is_file():
                payload["files"][f"checkpoint/{name}"] = json.loads(path.read_text())
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if not payload["files"] and not groups:
        raise RuntimeError("no pilot artifacts found")
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode()
    # Keep log envelopes short; a reader reassembles chunks in index order.
    chunks = [encoded[i:i+6000] for i in range(0, len(encoded), 6000)]
    for index, chunk in enumerate(chunks):
        print(json.dumps({"officeqa_artifact_chunk": index, "chunks": len(chunks), "data": chunk}), flush=True)
    print(json.dumps({"officeqa_export": "PASS", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "files": list(payload["files"]), "chunks": len(chunks)}), flush=True)


if __name__ == "__main__":
    main()
