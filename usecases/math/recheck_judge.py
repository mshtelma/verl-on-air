"""Replay recorded training answers against the real judge before restarting GPUs."""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from export_run import emit_payload
import judge_selfcheck
import reward


async def recheck(directory: Path) -> dict:
    calibration = await judge_selfcheck.grade_all()
    if not all(ok for _, _, ok, _ in calibration):
        raise RuntimeError("judge calibration failed")
    rows = []
    records = [(path, json.loads(path.read_text())) for path in directory.glob("*.json")]
    # Test the verdict that stopped training first, then every other recorded answer.
    records.sort(key=lambda pair: (pair[1]["metrics"]["judge_valid"], pair[0].name))
    try:
        for path, record in records:
            evidence = {}
            score = await asyncio.wait_for(reward.call_judge(record["question"], record["solution"],
                                                           record["ground_truth"], evidence=evidence),
                                           reward._deadline_s())
            agrees = (score >= 0.5) == (record["metrics"]["acc"] >= 0.5)
            rows.append({**record, "recheck_score": score, "recheck_verdict": evidence,
                         "recheck_rule_agree": agrees, "trace_file": path.name})
            print(json.dumps({"recheck_index": record["index"], "score": score,
                              "previous_valid": record["metrics"]["judge_valid"],
                              "rule_agree": agrees, "verdict": evidence}), flush=True)
    finally:
        await reward.close_sessions()
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode"]["rollout_group_id"]].append(row["recheck_score"])
    if not rows:
        raise RuntimeError("no recorded episodes")
    previous_invalid = sum(row["metrics"]["judge_valid"] != 1 for row in rows)
    if not previous_invalid:
        raise RuntimeError("the replay does not include the previously invalid verdict")
    return {"status": "PASS", "calibration": calibration, "trajectories": len(rows),
            "previous_invalid": previous_invalid, "mixed_reward_groups": sum(len(set(v)) > 1 for v in groups.values()),
            "rule_agreement": sum(row["recheck_rule_agree"] for row in rows) / len(rows),
            "verdict_mode": reward._verdict_mode(),
            "trace_dir": str(directory), "records": rows}


def main() -> None:
    directory = Path(os.environ["RECHECK_TRACE_DIR"])
    result = asyncio.run(recheck(directory))
    emit_payload({"files": {"judge_recheck.json": result}, "receipts": {}})
    print(json.dumps({"judge_recheck": "PASS", "trajectories": result["trajectories"],
                      "previous_invalid": result["previous_invalid"],
                      "mixed_reward_groups": result["mixed_reward_groups"]}), flush=True)


if __name__ == "__main__":
    main()
