#!/usr/bin/env python3
"""Concurrency load test for the judge server.

Measures the metric that actually matters for RL — AGGREGATE grading throughput
under concurrent load — not the single-request latency that VALIDATE_ONLY reports.
A dedicated judge on 16 H100 (TP16) is driven by verl's rate-limited reward manager
firing many gradings at once; a TP16 server with continuous batching turns that burst
into high throughput even though any one grading is slow. This script reproduces that
burst and reports what training will actually see.

It reuses judge_reward.compute_score (the exact reward path used in training, incl.
JSON parsing + rule fallback), so judge_ok here means the same thing it does in a run.

Env:
  JUDGE_LOADTEST_TOTAL        total gradings to fire (default 128)
  JUDGE_LOADTEST_CONCURRENCY  max in flight at once (default 32)
  JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_TIMEOUT ... (read by judge_reward)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_reward import compute_score  # same reward path training uses

TOTAL = int(os.environ.get("JUDGE_LOADTEST_TOTAL", "128"))
CONCURRENCY = int(os.environ.get("JUDGE_LOADTEST_CONCURRENCY", "32"))

# A representative GSM8K tool-agent trajectory (correct answer, 2 tool calls). The
# judge system prompt is identical every call, so vLLM's automatic prefix caching
# should make the shared-prefix prefill ~free — exactly the training pattern.
_TRAJ = (
    "<think>18 eggs, eats 3, bakes 4, sells the rest.</think>"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "18 - 3 - 4"}}</tool_call>'
    "The remainder is 11, at $2 each.\n"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "11 * 2"}}</tool_call>'
    "So she makes $22.\n#### 22"
)
_QUESTION = (
    "Janet's ducks lay 18 eggs per day. She eats 3 for breakfast and bakes muffins "
    "with 4. She sells the remainder at $2 per egg. How much does she make per day?"
)


async def _one(sem: asyncio.Semaphore) -> tuple[float, float]:
    async with sem:
        t0 = time.perf_counter()
        try:
            out = await compute_score(
                data_source="openai/gsm8k",
                solution_str=_TRAJ,
                ground_truth="22",
                extra_info={"question": _QUESTION, "num_turns": 5},
            )
            return time.perf_counter() - t0, float(out.get("judge_ok", 0.0))
        except Exception:  # noqa: BLE001 - count as an error, keep the sweep going
            return time.perf_counter() - t0, -1.0


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100.0 * len(sorted_vals)))]


async def main() -> int:
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"[loadtest] firing {TOTAL} gradings, concurrency={CONCURRENCY}, "
          f"url={os.environ.get('JUDGE_BASE_URL')}", flush=True)
    wall0 = time.perf_counter()
    results = await asyncio.gather(*(_one(sem) for _ in range(TOTAL)))
    wall = time.perf_counter() - wall0

    lat = sorted(r[0] for r in results)
    n_ok = sum(1 for _, ok in results if ok == 1.0)
    n_fallback = sum(1 for _, ok in results if ok == 0.0)
    n_err = sum(1 for _, ok in results if ok < 0.0)

    print(f"[loadtest] wall={wall:.1f}s  throughput={TOTAL / wall:.2f} gradings/s  "
          f"(effective concurrency ~{sum(lat) / wall:.1f})", flush=True)
    print(f"[loadtest] judge_ok={n_ok}/{TOTAL}  rule_fallback={n_fallback}  errors={n_err}", flush=True)
    print(f"[loadtest] per-grading latency s: p50={_pct(lat, 50):.1f} "
          f"p90={_pct(lat, 90):.1f} p95={_pct(lat, 95):.1f} max={lat[-1]:.1f}", flush=True)
    # Non-fatal by design: this is a measurement, not a gate. Signal only egregious failure.
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
