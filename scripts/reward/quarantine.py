#!/usr/bin/env python3
"""Whole-group quarantine for UNKNOWN verifier outcomes (Gate-R training side).

Why: in verl's fully-async reward loop, ``compute_score`` runs PER SAMPLE as rollouts
complete (``RewardLoopWorker.run_single``) -- it is group-blind by construction, so a
judge "unknown" cannot be handled inside the reward function beyond marking it. Before
this module, an unknown returned scalar 0.0 and ``RateLimitedRewardManager`` passed it
straight through, silently injecting a fabricated 0 into the GRPO group mean and
pulling every sibling's advantage.

Design (no verl fork; see docs/officeqa_rl_plan.md Gate R):
  1. ``compute_score`` (officeqa_grounded_reward) marks an unknown with
     ``score = OQ_UNKNOWN_SENTINEL`` (-1.0, outside the graded range [0, ~1.15]) when
     ``OQ_REWARD_QUARANTINE=1`` -- the sentinel rides the normal rm_scores channel, so
     it survives per-sample streaming, chunking, and serialization untouched.
  2. Trainer-side, ``scripts/_site/sitecustomize.py`` (auto-imported via PYTHONPATH
     when the same env is set) wraps the registered GRPO advantage estimator: after
     the original computation it ZEROES the advantages/returns of every uid-group that
     contains a sentinel. The sentinel only ever contaminates the mean of its own
     group -- and that whole group is dropped, so uncontaminated groups are bit-exact
     vs. vanilla GRPO.

This file is the PURE, dependency-free core (lists in / lists out) so the grouping
logic is unit-testable on any CPU box; the torch boundary in sitecustomize is a thin
adapter over :func:`quarantine_row_mask`.

NOTE: ``OQ_UNKNOWN_SENTINEL`` is the single source of truth for the sentinel value;
sitecustomize loads this module BY PATH so trainer and reward workers always agree.
"""

from __future__ import annotations

import os

# Outside every legitimate reward range in this project (graded rewards live in
# [0, 1.15]; binary answer-mode in {0, 1}). Exact float, compared exactly.
OQ_UNKNOWN_SENTINEL = -1.0

# Env knob (string "1") shared by the reward function (emit sentinel) and the
# sitecustomize advantage patch (zero sentinel groups). Single name by design.
OQ_QUARANTINE_ENV = "OQ_REWARD_QUARANTINE"


def quarantine_enabled(env: dict | None = None) -> bool:
    return (env if env is not None else os.environ).get(OQ_QUARANTINE_ENV) == "1"


def unknown_score(env: dict | None = None) -> float:
    """Score a verifier 'unknown' should emit: sentinel under quarantine mode,
    legacy 0.0 otherwise (offline scoring / rgate keep 0.0 + status='unknown')."""
    return OQ_UNKNOWN_SENTINEL if quarantine_enabled(env) else 0.0


def sentinel_rows(sample_rewards: list[float], sentinel: float = OQ_UNKNOWN_SENTINEL) -> list[bool]:
    """Per-sample True where the sample's scalar reward IS the unknown sentinel."""
    return [r == sentinel for r in sample_rewards]


def quarantine_row_mask(
    sample_rewards: list[float],
    index: list,
    sentinel: float = OQ_UNKNOWN_SENTINEL,
) -> list[bool]:
    """Per-sample True where the sample must be QUARANTINED: its whole uid-group is
    dropped because at least one sibling's reward is the unknown sentinel.

    Whole-group (not per-sample) by policy: a group with an unverifiable member has
    an untrustworthy baseline, so no sibling in it may contribute policy gradient.
    """
    bad_groups = {idx for r, idx in zip(sample_rewards, index) if r == sentinel}
    if not bad_groups:
        return [False] * len(sample_rewards)
    return [idx in bad_groups for idx in index]


def apply_quarantine(
    advantages: list[list[float]],
    returns: list[list[float]],
    sample_rewards: list[float],
    index: list,
    sentinel: float = OQ_UNKNOWN_SENTINEL,
) -> tuple[list[list[float]], list[list[float]], int]:
    """Zero advantages/returns of quarantined groups. Pure-list mirror of the torch
    adapter in scripts/_site/sitecustomize.py (keep them in lockstep).

    Returns (advantages, returns, n_quarantined_groups).
    """
    mask = quarantine_row_mask(sample_rewards, index, sentinel)
    n_bad = len({idx for m, idx in zip(mask, index) if m})
    if not any(mask):
        return advantages, returns, 0
    adv = [[0.0] * len(row) if m else list(row) for row, m in zip(advantages, mask)]
    ret = [[0.0] * len(row) if m else list(row) for row, m in zip(returns, mask)]
    return adv, ret, n_bad


if __name__ == "__main__":
    # smoke: 3 groups, group 'b' has one unknown -> whole group b zeroed
    rewards = [1.0, 0.0, 0.5, OQ_UNKNOWN_SENTINEL, 0.25]
    index = ["a", "a", "b", "b", "c"]
    adv = [[r] * 3 for r in rewards]
    a2, r2, n = apply_quarantine(adv, adv, rewards, index)
    assert [row[0] for row in a2] == [1.0, 0.0, 0.0, 0.0, 0.25] and n == 1
    print("quarantine smoke OK:", a2)
