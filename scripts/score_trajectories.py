#!/usr/bin/env python3
"""Score a bundle of OfficeQA trajectories with the GROUNDED reward (deterministic
source-identity + GLM-5.3 grounding-audit judge) and write per-record reward
breakdowns for HUMAN VALIDATION.

This is the validation pilot the user asked for: produce ~100 (trajectory, reward)
records so the reward -- especially its ability to separate genuinely-grounded
correct answers from lucky-wrong-source ones -- can be eyeballed BEFORE it ever
drives training.

Flow:
  1. read the JSONL trace bundle from a trace-enabled eval (air/72) -- each line has
     uid/question/gt/source_files + a structured `trajectory` (reasoning + tool
     calls + tool outputs);
  2. (stratify) keep ALL answer-correct records (only those can be "lucky", the case
     we most need to inspect) + a sample of wrong ones, up to OQ_SCORE_LIMIT;
  3. score each with reward.officeqa_grounded_reward.score_record against the judge
     served locally (JUDGE_BASE_URL, set from EVAL_BASE_URL by serve_and_eval.sh);
  4. write reward records to OQ_SCORES_OUT and print a distribution + examples.

Env: OQ_TRACES_IN (JSONL in), OQ_SCORES_OUT (JSONL out), OQ_SCORE_LIMIT (0=all),
     OQ_JUDGE_ALL (1=also judge wrong answers, for judging the judge), JUDGE_CONCURRENCY,
     plus the JUDGE_* knobs consumed by officeqa_grounded_reward.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Point the grounded-reward judge client at the locally-served judge (serve_and_eval.sh
# exports EVAL_BASE_URL / EVAL_MODEL for the vLLM server it brought up).
os.environ.setdefault("JUDGE_BASE_URL", os.environ.get("EVAL_BASE_URL", ""))
os.environ.setdefault("JUDGE_MODEL", os.environ.get("EVAL_MODEL", "judge"))

from reward.answer_extract import XMLTagExtractor          # noqa: E402
from reward.officeqa_grounded_reward import score_record    # noqa: E402
from reward.officeqa_reward import score_answer             # noqa: E402

TRACES_IN = os.environ.get("OQ_TRACES_IN", "/Volumes/main/mshtelma/verl/eval/officeqa_traces.jsonl")
# Allow a snapshot-relative bundle (e.g. reward/tests/rgate_adversarial.jsonl) so fixtures
# can travel WITH the code snapshot instead of being uploaded to the Volume.
if not os.path.isabs(TRACES_IN) and not os.path.exists(TRACES_IN):
    _cand = os.path.join(_HERE, TRACES_IN)
    if os.path.exists(_cand):
        TRACES_IN = _cand
SCORES_OUT = os.environ.get("OQ_SCORES_OUT", "/Volumes/main/mshtelma/verl/eval/officeqa_grounded_scores.jsonl")
LIMIT = int(os.environ.get("OQ_SCORE_LIMIT", "0"))
JUDGE_ALL = os.environ.get("OQ_JUDGE_ALL", "1") == "1"
CONC = int(os.environ.get("JUDGE_CONCURRENCY", "16"))
_TIGHT_TOL = float(os.environ.get("OQ_TIGHT_TOL", "0.0"))

_ext = XMLTagExtractor(tag="FINAL_ANSWER")


def _read_bundle(path: str) -> list[dict]:
    recs = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def _flatten(rec: dict) -> str:
    return "\n".join(
        str(r.get("result") or "") for st in (rec.get("trajectory") or [])
        for r in (st.get("tool_results") or [])
    )


def _is_correct(rec: dict) -> bool:
    # Historical bundles carry an authoritative `pred`; otherwise extract from assistant
    # reasoning only (never tool output, which may print a final-answer-looking tag).
    pred = rec.get("pred")
    if not pred:
        assistant_text = "\n".join(str(st.get("reasoning") or "") for st in (rec.get("trajectory") or []))
        pred = _ext.extract(assistant_text) or ""
    try:
        return bool(pred) and score_answer(str(rec.get("gt", "")), pred, _TIGHT_TOL) > 0
    except Exception:  # noqa: BLE001
        return False


def _stratify(recs: list[dict], limit: int) -> list[dict]:
    """Keep ALL answer-correct records (the only ones that can be lucky -> the band
    we most need validated), then fill with a sample of wrong ones up to `limit`."""
    if limit <= 0 or limit >= len(recs):
        return recs
    correct = [r for r in recs if _is_correct(r)]
    wrong = [r for r in recs if r not in correct]
    rng = random.Random(0)
    rng.shuffle(wrong)
    take_wrong = max(0, limit - len(correct))
    chosen = correct[:limit] + wrong[:take_wrong]
    rng.shuffle(chosen)
    print(f"[score] stratified: {len(correct)} correct kept + "
          f"{min(take_wrong, len(wrong))} wrong sampled -> {len(chosen)}", flush=True)
    return chosen


async def _run():
    recs = _read_bundle(TRACES_IN)
    print(f"[score] read {len(recs)} traces from {TRACES_IN}", flush=True)
    if not recs:
        print("[score] FATAL: no traces to score", file=sys.stderr)
        sys.exit(1)
    recs = _stratify(recs, LIMIT)
    print(f"[score] scoring {len(recs)} records (judge_all={JUDGE_ALL} conc={CONC} "
          f"judge={os.environ.get('JUDGE_BASE_URL')!r} model={os.environ.get('JUDGE_MODEL')!r})", flush=True)

    sem = asyncio.Semaphore(CONC)

    async def _one(rec):
        async with sem:
            try:
                return await score_record(rec, judge_all=JUDGE_ALL)
            except Exception as e:  # noqa: BLE001
                return {"uid": rec.get("uid"), "error": f"{type(e).__name__}: {e}",
                        "reward": 0.0, "benchmark_correct": _is_correct(rec)}

    out = await asyncio.gather(*[_one(r) for r in recs])

    os.makedirs(os.path.dirname(SCORES_OUT) or ".", exist_ok=True)
    with open(SCORES_OUT, "w") as fh:
        for o in out:
            fh.write(json.dumps(o, default=list) + "\n")

    # --- summary --------------------------------------------------------------
    n = len(out)
    correct = [o for o in out if o.get("benchmark_correct")]
    rewarded = [o for o in out if (o.get("reward") or 0) > 0]
    lucky = [o for o in correct if (o.get("reward") or 0) <= 0.05]
    verdicts: dict = {}
    for o in out:
        v = (o.get("judge_verdict") or {}).get("verdict") if o.get("judge_verdict") else None
        verdicts[v] = verdicts.get(v, 0) + 1
    mean_r = sum((o.get("reward") or 0) for o in out) / n if n else 0.0
    mean_r_correct = (sum((o.get("reward") or 0) for o in correct) / len(correct)) if correct else 0.0

    print("\n==================== GROUNDED REWARD SUMMARY ====================", flush=True)
    print(f"records={n}  answer_correct={len(correct)}  reward>0={len(rewarded)}", flush=True)
    print(f"correct-but-LUCKY (reward<=0.05)={len(lucky)}  "
          f"(these are the wrong-path/right-number rollouts the reward demotes)", flush=True)
    print(f"mean_reward(all)={mean_r:.3f}  mean_reward(correct)={mean_r_correct:.3f}", flush=True)
    print(f"judge verdicts: {verdicts}", flush=True)
    print("---- examples (grounded / lucky / wrong) ----", flush=True)
    def _ex(pred):
        for o in out:
            if pred(o):
                v = o.get("judge_verdict") or {}
                print(f"  [{'OK' if o.get('benchmark_correct') else 'XX'}] r={o.get('reward'):.2f} "
                      f"uid={o.get('uid')} verdict={v.get('verdict')} "
                      f"src_seen={(o.get('grounding') or {}).get('source_identity',{}).get('any_score')} "
                      f"reason={o.get('reward_reason')}", flush=True)
                return
    _ex(lambda o: o.get("benchmark_correct") and (o.get("reward") or 0) > 0.7)  # grounded
    _ex(lambda o: o in lucky)                                                     # lucky
    _ex(lambda o: not o.get("benchmark_correct"))                                # wrong
    print(f"[score] wrote {n} reward records -> {SCORES_OUT}", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    asyncio.run(_run())
