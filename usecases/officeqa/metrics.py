"""OfficeQA pass@k and grouped binary-reward summaries (pure CPU)."""
from __future__ import annotations

import math
import random


def pass_at_k(n: int, correct: int, k: int) -> float:
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0<=correct<=n and 1<=k<=n")
    return 1.0 if n - correct < k else 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def summarize(results: list[dict], samples: int) -> dict:
    groups = {}
    for result in results:
        groups.setdefault(result["group"], []).append(result)
    full = [group for group in groups.values() if len(group) == samples and all(r["status"] == "scored" for r in group)]
    rewards = [sum(r["reward"] > 0 for r in group) for group in full]
    scored = [r for group in full for r in group]
    return {
        "questions": len(groups), "fully_scored_questions": len(full), "samples_per_question": samples,
        "successful_episodes": sum(rewards),
        "pass_at_1": sum(pass_at_k(samples, c, 1) for c in rewards) / len(full) if full else None,
        "pass_at_3": sum(pass_at_k(samples, c, 3) for c in rewards) / len(full) if full and samples >= 3 else None,
        "mixed_reward_groups": sum(0 < c < samples for c in rewards),
        "all_zero_groups": sum(c == 0 for c in rewards),
        "all_successful_groups": sum(c == samples for c in rewards),
        "answer_accuracy": sum(r["metrics"]["answer_correct"] for r in scored) / len(scored) if scored else None,
        "submission_rate": sum(r["metrics"]["terminal_seen"] for r in scored) / len(scored) if scored else None,
        "mean_delivered_observations": sum(r["metrics"]["delivered_observations"] for r in scored) / len(scored) if scored else None,
    }


def compare(before: dict, after: dict, *, bootstrap_samples=10000, seed=42) -> dict:
    """Compare frozen evaluations, resampling questions rather than correlated episodes."""
    if before.get("valid") is not True or after.get("valid") is not True:
        raise ValueError("both evaluations must be valid")
    for key in ("eval_policy", "question_ids_sha256", "dataset"):
        if before[key] != after[key]:
            raise ValueError(f"paired evaluation mismatch: {key}")
    if before["code"]["git_sha"] != after["code"]["git_sha"]:
        raise ValueError("paired evaluation uses different source snapshots")
    samples = before["eval_policy"]["samples_per_question"]
    rows = [{row["id"]: row for row in artifact["results"]} for artifact in (before, after)]
    if set(rows[0]) != set(rows[1]) or any(len(rows[i]) != len(artifact["results"]) for i, artifact in enumerate((before, after))):
        raise ValueError("paired episode IDs are missing or duplicated")
    if any(row["status"] != "scored" or row["reward"] not in (0, 1) for group in rows for row in group.values()):
        raise ValueError("paired evaluation contains an unresolved/nonbinary reward")
    output = {}
    for split in sorted({row["split"] for row in before["results"]}):
        uids = sorted({row["uid"] for row in before["results"] if row["split"] == split})
        values = {}
        for uid in uids:
            groups = [[row for row in artifact["results"] if row["uid"] == uid and row["split"] == split]
                      for artifact in (before, after)]
            if any(len(group) != samples for group in groups):
                raise ValueError("paired evaluation contains an incomplete question group")
            counts = [sum(row["reward"] for row in group) for group in groups]
            values[uid] = {
                "pass_at_1": [count / samples for count in counts],
                "answer_accuracy": [sum(row["metrics"]["answer_correct"] for row in group) / samples for group in groups],
            }
            if samples >= 3:
                values[uid]["pass_at_3"] = [pass_at_k(samples, int(count), 3) for count in counts]
        comparison = {}
        for name in values[uids[0]]:
            pairs = [values[uid][name] for uid in uids]
            deltas = [pair[1] - pair[0] for pair in pairs]
            rng = random.Random(seed)
            resampled = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(bootstrap_samples))
            comparison[name] = {"baseline": sum(pair[0] for pair in pairs) / len(pairs),
                                "final": sum(pair[1] for pair in pairs) / len(pairs),
                                "delta": sum(deltas) / len(deltas),
                                "delta_ci95": [resampled[int(0.025 * bootstrap_samples)], resampled[int(0.975 * bootstrap_samples)]],
                                "improved_questions": sum(delta > 0 for delta in deltas),
                                "regressed_questions": sum(delta < 0 for delta in deltas),
                                "unchanged_questions": sum(delta == 0 for delta in deltas)}
        output[split] = {"questions": len(uids), "samples_per_question": samples,
                         "metrics": comparison, "per_question": values}
    return {"bootstrap_unit": "question", "bootstrap_samples": bootstrap_samples, "seed": seed,
            "heldout_used_for_selection": False, "splits": output}
