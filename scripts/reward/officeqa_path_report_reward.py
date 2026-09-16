"""In-loop GRPO reward for the OfficeQA PATH-REPORT agent.

This promotes the *validated* path-report verifier (Deliverable A 34/34; Deliverable B on
real rollouts) to a training reward WITHOUT changing it: the custom agent loop
(``path_report_agent``) accumulates, during the rollout, a capture RECORD in
``agent_data.extra_fields["path_report_record"]`` -- byte-identical in shape to the offline
collector's capture (episode_id/question/terminal_text/observations/...). verl carries that
through ``non_tensor_batch`` into the reward's ``extra_info`` (verified against verl v0.9.0:
agent_loop extra_fields -> non_tensor_batch -> compute_score(extra_info)). Here we run the
EXACT offline scorer (``path_report_pilot.score_episode``) and map its candidate status to a
scalar reward. Same code path = the reward equals the verifier we proved out.

Reward mapping (from ``candidate_preview`` -- the retained contract, unchanged):
    SCORED      -> path_score in (0.0, 1.0]    correct answer + supported + trace-faithful (GRADED)
    ZERO        -> 0.0                          wrong/missing answer, malformed, invalid/fabricated
                                                path, or judge-unsupported
    ABSTENTION  -> ABSTENTION_REWARD (0.0)      explicit DATA NOT AVAILABLE -- recorded apart, NOT
                                                punished below a wrong answer
    UNKNOWN     -> UNKNOWN_REWARD (0.0)         verifier failure; "UNKNOWN != negative" (contract)
    no record   -> 0.0                          actor/interface failure (never emitted a submit)

*** OPEN CALIBRATION (the pilot deferred these; they MUST be settled before trusting a delta) ***
  1. ``algorithm.norm_adv_by_std_in_grpo=False`` is REQUIRED, or the graded 0.3-1.0 scale
     collapses to identical advantages (see the OfficeQA RL memory). The launcher must set it.
  2. UNKNOWN-in-RL: a *verifier* failure must not read as a negative signal. The correct fix is
     to EXCLUDE the sample from the GRPO advantage baseline; that needs reward-manager support
     and is NOT yet wired. Until it is, UNKNOWN returns ``UNKNOWN_REWARD`` (default 0.0) and every
     UNKNOWN is COUNTED + logged (``reward_diagnostics()``) so the in-loop UNKNOWN rate stays
     visible -- it must be near-zero (bump JUDGE_MAX_MODEL_LEN / retries) or the signal is noisy.
  3. No dense shaping for wrong answers (retained from the reward-contract rebuild): a wrong or
     unsupported trajectory is a hard 0.0, not a partial credit.

All knobs are read at CALL time (mirrors reward/judge_reward.py) so the air recipe configures
them without touching call sites. Judge URL is resolved from the rendezvous file at call time.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICEQA = os.path.join(os.path.dirname(_HERE), "officeqa")
for _p in (_OFFICEQA, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import path_report as pr                       # noqa: E402  (stdlib-only; status constants + scoring)
import path_report_pilot as prp                # noqa: E402  (score_episode + make_http_judge)

# ---------------------------------------------------------------------------
# call-time config (env; defaults match the offline verifier)
# ---------------------------------------------------------------------------
def _cfg():
    return {
        "answer_tol": float(os.environ.get("OQ_RL_ANSWER_TOL", "0.0")),
        "max_history_bytes": int(os.environ.get("OQ_RL_MAX_HISTORY_BYTES",
                                                str(pr.DEFAULT_MAX_HISTORY_BYTES))),
        "judge_retries": int(os.environ.get("OQ_RL_JUDGE_RETRIES", "2")),
        "unknown_reward": float(os.environ.get("OQ_RL_UNKNOWN_REWARD", "0.0")),
        "abstention_reward": float(os.environ.get("OQ_RL_ABSTENTION_REWARD", "0.0")),
    }

# ---------------------------------------------------------------------------
# diagnostics: count candidate statuses seen in-loop (esp. UNKNOWN -- see calibration note)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_counts: dict[str, int] = {}


def _tally(status: str) -> None:
    with _lock:
        _counts[status] = _counts.get(status, 0) + 1


def reward_diagnostics() -> dict:
    """Snapshot of candidate-status counts since process start (for the training log)."""
    with _lock:
        return dict(_counts)


# ---------------------------------------------------------------------------
# record extraction: the custom loop writes extra_fields["path_report_record"];
# verl may hand it back as a dict OR a JSON string (non_tensor_batch stringifies).
# ---------------------------------------------------------------------------
def _record_from_extra_info(extra_info, solution_str) -> dict | None:
    if isinstance(extra_info, dict):
        rec = extra_info.get("path_report_record")
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except Exception:  # noqa: BLE001
                rec = None
        if isinstance(rec, dict) and rec.get("observations") is not None:
            return rec
    return None


# ---------------------------------------------------------------------------
# judge: resolve the served GLM-5.3 URL at call time (rendezvous), build the dialect-aware
# judge_fn. Judge unreachable/disabled -> None -> valid reports score UNKNOWN (fail-closed),
# never a spurious positive.
# ---------------------------------------------------------------------------
def _make_judge():
    try:
        from judge_reward import _resolve_judge_url  # noqa: E402 (reuse the proven rendezvous)
        base_url = _resolve_judge_url()
    except Exception:  # noqa: BLE001
        base_url = os.environ.get("JUDGE_BASE_URL")
    if not base_url:
        return None
    model = os.environ.get("JUDGE_MODEL", "judge")
    try:
        return prp.make_http_judge(base_url, model)
    except Exception:  # noqa: BLE001
        return None


def _reward_from_candidate(out: dict, cfg: dict) -> float:
    cand = out.get("candidate") or {}
    status = cand.get("status")
    _tally(str(status))
    if status == pr.SCORED:
        s = cand.get("score")
        return float(s) if s is not None else cfg["unknown_reward"]
    if status == pr.ABSTENTION:
        return cfg["abstention_reward"]
    if status == pr.UNKNOWN:
        return cfg["unknown_reward"]
    # ZERO and anything else -> hard zero (no dense shaping for wrong)
    return 0.0


def score_record(rec: dict, judge_fn, ground_truth, cfg: dict | None = None) -> tuple[float, dict]:
    """Synchronous core (unit-testable): score one capture record with an injected judge_fn.
    Returns (reward, scored_output). ``judge_fn=None`` disables the semantic judge (valid
    reports then resolve to UNKNOWN -- fail-closed, never a spurious positive)."""
    cfg = cfg or _cfg()
    eid = str(rec.get("episode_id") or "")
    answer_key = {eid: str(ground_truth)} if ground_truth is not None else None
    out = prp.score_episode(rec, judge_fn, max_retries=cfg["judge_retries"],
                            max_history_bytes=cfg["max_history_bytes"],
                            answer_key=answer_key, answer_tol=cfg["answer_tol"])
    return _reward_from_candidate(out, cfg), out


# ---------------------------------------------------------------------------
# verl entry point: rate_limited reward manager awaits this directly (async).
# Signature matches verl's compute_score(data_source, solution_str, ground_truth, extra_info).
# The sync scorer (which does blocking judge HTTP) runs in a thread so it never blocks the loop.
# ---------------------------------------------------------------------------
async def compute_score(data_source=None, solution_str=None, ground_truth=None,
                        extra_info=None, **kwargs) -> float:
    rec = _record_from_extra_info(extra_info, solution_str)
    if rec is None:
        # No path-report record: the episode never emitted a submit (actor/interface failure,
        # not a verifier failure) -> 0.0, distinct from UNKNOWN.
        _tally("no_record")
        return 0.0
    cfg = _cfg()
    judge_fn = _make_judge()
    reward, _ = await asyncio.to_thread(score_record, rec, judge_fn, ground_truth, cfg)
    return reward
