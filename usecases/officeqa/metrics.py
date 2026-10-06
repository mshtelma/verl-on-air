"""OfficeQA pass@k and grouped binary-reward summaries (pure CPU)."""
from __future__ import annotations

import math


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
