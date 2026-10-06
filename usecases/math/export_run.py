"""Export small math result evidence through checksummed AIR log envelopes."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path


def training_signal(directory: Path) -> dict:
    groups = defaultdict(list)
    versions = defaultdict(list)
    selected = {"correct": [], "incorrect": []}
    count = 0
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        episode, metrics = record["episode"], record["metrics"]
        group, version = episode["rollout_group_id"], episode["parameter_version"]
        row = {"score": metrics["score"], "acc": metrics["acc"],
               "judge_valid": metrics["judge_valid"], "judge_fallback": metrics["judge_fallback"],
               "judge_input_truncated": metrics["judge_input_truncated"],
               "parameter_version": version, "index": record["index"]}
        groups[group].append(row)
        versions[version].append(row)
        count += 1
        label = "correct" if metrics["acc"] else "incorrect"
        if len(selected[label]) < 2:
            selected[label].append(record)
    if not count:
        raise RuntimeError("no math trajectory audit records")
    return {
        "trajectories": count, "groups": dict(groups),
        "group_sizes": dict(Counter(len(rows) for rows in groups.values())),
        "mixed_reward_groups": sum(len({row["score"] for row in rows}) > 1 for rows in groups.values()),
        "mixed_correctness_groups": sum(len({row["acc"] for row in rows}) > 1 for rows in groups.values()),
        "invalid_judge_responses": sum(row["judge_valid"] != 1 for rows in groups.values() for row in rows),
        "fallbacks": sum(row["judge_fallback"] for rows in groups.values() for row in rows),
        "by_parameter_version": {
            str(version): {"trajectories": len(rows),
                           "judge_score": statistics.mean(row["score"] for row in rows),
                           "answer_accuracy": statistics.mean(row["acc"] for row in rows)}
            for version, rows in sorted(versions.items())},
        "selected_trajectories": selected,
    }


def emit_payload(payload: dict) -> None:
    """Keep complete evidence in AIR logs with an independently checked checksum."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode()
    chunks = [encoded[i:i+6000] for i in range(0, len(encoded), 6000)]
    for index, chunk in enumerate(chunks):
        print(json.dumps({"math_artifact_chunk": index, "chunks": len(chunks), "data": chunk}), flush=True)
    print(json.dumps({"math_artifact_export": "PASS", "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(), "chunks": len(chunks)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-out", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--reward-dir", type=Path)
    args = parser.parse_args()
    payload = {"files": {}, "receipts": {}}
    paths = []
    if args.eval_out:
        paths.append(("eval.json", args.eval_out))
    if args.checkpoint_dir:
        paths.extend((name, args.checkpoint_dir / name) for name in ("run_result.json", "run_manifest.json"))
    for name, path in paths:
        raw = path.read_bytes()
        payload["files"][name] = json.loads(raw)
        payload["receipts"][name] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if args.reward_dir:
        payload["training_signal"] = training_signal(args.reward_dir)
    if not paths and not args.reward_dir:
        parser.error("at least one artifact is required")
    emit_payload(payload)


if __name__ == "__main__":
    main()
