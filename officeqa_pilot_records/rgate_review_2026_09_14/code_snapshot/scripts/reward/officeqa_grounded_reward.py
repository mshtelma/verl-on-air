#!/usr/bin/env python3
"""GROUNDED OfficeQA reward -- strict answer GATE, then a GRADED, FAIL-CLOSED process
score. Catches the failure the user flagged: a wrong retrieval path (TOTAL vs ARMY, a
neighbouring period) that lands on a number close to / equal to the gold. Answer-match
alone credits that lucky path and GRPO reinforces it; here it collapses to zero.

DESIGN (docs/officeqa_rl_plan.md Section 6.4; owner decision: GRADED, not binary):

    pred = <FINAL_ANSWER> from the trajectory
    strict-correct?  (strict_answer.strict_correct -- arity/order/unit/exact, NOT the
                      fuzzy benchmark scorer, which a review showed accepts candidate
                      lists, reversed lists, wrong units, and split sci-notation)
      no   -> 0.0                                   # gated; no dense shaping for wrong
      yes  -> grade HOW it grounded:
        the JUDGE (self-hosted GLM-5.3) audits the transcript vs the answer key and is
        REQUIRED for any positive. Then, FAIL-CLOSED:
          judge unavailable / verdict unusable      -> status "unknown"  (retry/quarantine;
                                                       score 0.0 is a placeholder, NOT a
                                                       real negative -- see verifier_ok)
          no real retrieval at all                  -> 0.0  (deterministic fail-closed floor)
          judge did NOT affirm support (flag absent)-> "unknown"  (ELIGIBILITY is fail-closed:
                                                       a positive REQUIRES support==true, never
                                                       merely "not false" -- a missing flag is an
                                                       unusable verdict, not a free positive)
          judge: answer NOT supported (support=false)-> 0.0  (the lucky/unsupported case)
          composite & completeness flag absent      -> "unknown"
          composite & NOT all components retrieved  -> 0.0
          route <= LUCKY_ROUTE_MAX                   -> 0.0
          else  -> clip(BASE + W_ROUTE*route + W_TRAJ*traj_quality, BASE, 1.0)  # GRADED

CRITICAL vs the old v2 (all confirmed as bugs in review, now removed):
  * NO fail-open: a correct number with an empty grep/read of the gold file no longer
    scores 1.0 when the judge is down -- it is now "unknown" (quarantine).
  * NO 0.05 lucky floor: a correct-but-wrong-route answer scores 0.0 (paying anything
    for it is exactly the hazard, and GRPO amplifies a lone positive in a group).
  * The judge parser never invents certainty (see judge_prompt.parse_verdict).

REQUIRED training-config pairing: GRPO with std-normalization OFF
(`algorithm.norm_adv_by_std_in_grpo=False`) so the graded scale actually propagates
(with std-norm ON, 0.05 and 1.0 give identical advantages -- see the plan Section 6.2).

MODES (OQ_REWARD_MODE): `grounded` (default, judge audit) | `answer_source`
(deterministic gold-source gate, no judge -- cheap ablation) | `answer` (strict-answer
only -> 1/0, the real answer-only control).

TWO entry points, ONE formula (`_assemble`):
  * ``compute_score`` -- verl ``rate_limited`` reward-manager contract (training).
  * ``score_record``  -- offline scorer for the validation pilot.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

# --- import bootstrap: work whether scripts/ or scripts/reward/ is on sys.path ---
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):   # scripts/ , scripts/reward/
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from reward import grounding, judge_prompt          # type: ignore
    from reward.answer_extract import XMLTagExtractor    # type: ignore
    from reward.officeqa_reward import score_answer      # type: ignore
    from reward.strict_answer import strict_correct      # type: ignore
    from reward.quarantine import unknown_score          # type: ignore
except Exception:  # noqa: BLE001
    import grounding, judge_prompt                       # type: ignore
    from answer_extract import XMLTagExtractor           # type: ignore
    from officeqa_reward import score_answer             # type: ignore
    from strict_answer import strict_correct             # type: ignore
    from quarantine import unknown_score                 # type: ignore
# Reuse the battle-tested rendezvous URL resolution + shared aiohttp session.
try:
    from reward.judge_reward import _get_session, _resolve_judge_url  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from judge_reward import _get_session, _resolve_judge_url     # type: ignore
    except Exception:  # noqa: BLE001
        _resolve_judge_url = _get_session = None  # type: ignore

_extractor = XMLTagExtractor(tag="FINAL_ANSWER")

_ASSISTANT_OPEN_RE = r"<\|im_start\|>assistant"


def _assistant_segments(solution_str: str) -> list[str]:
    """Assistant-owned text only. Tool outputs and their embedded tags are NEVER an
    answer commitment (the review's tool-only <FINAL_ANSWER> leak). Structured traces
    are handled by callers from their assistant reasoning; for flat training text,
    restrict extraction to assistant segments when markers exist."""
    import re
    text = str(solution_str or "")
    if not re.search(_ASSISTANT_OPEN_RE, text, re.IGNORECASE):
        return [text]
    parts = re.split(r"(<\|im_start\|>assistant|<\|im_start\|>tool)", text, flags=re.IGNORECASE)
    out: list[str] = []
    for i in range(1, len(parts), 2):
        marker, body = parts[i].lower(), parts[i + 1] if i + 1 < len(parts) else ""
        if marker == "<|im_start|>assistant":
            out.append(re.split(r"<\|im_end\|>", body, maxsplit=1, flags=re.IGNORECASE)[0])
    return out or [text]


def _committed_answer(text: str) -> str:
    """Extract the terminal committed answer from assistant text only."""
    pred = ""
    for seg in _assistant_segments(text):
        got = _extractor.extract(seg)
        if got:
            pred = got
    return pred

# --- reward shape (env-tunable) ----------------------------------------------
_BASE = float(os.environ.get("OQ_REWARD_BASE", "0.30"))         # correct + grounded floor
_W_ROUTE = float(os.environ.get("OQ_REWARD_W_ROUTE", "0.50"))   # judge route weight
_W_TRAJ = float(os.environ.get("OQ_REWARD_W_TRAJ", "0.20"))     # trajectory-quality weight
# correct-but-wrong-route ("lucky"): 0.0 by owner decision (was 0.05). Kept tunable but
# a POSITIVE floor is dangerous under GRPO -- do not raise it without std-norm OFF.
_LUCKY_SCORE = float(os.environ.get("OQ_REWARD_LUCKY_FLOOR", "0.0"))
_LUCKY_ROUTE_MAX = float(os.environ.get("OQ_REWARD_LUCKY_ROUTE_MAX", "0.20"))
# answer_source mode only: pulled SOME real table but not the exact gold file.
_UNCLEAR_ROUTE = float(os.environ.get("OQ_REWARD_UNCLEAR_ROUTE", "0.5"))
# Official benchmark tolerance for the REPORTING metric `acc` (not the gate). The gate is
# always strict/exact via strict_answer; the benchmark 1%/5% bands are reported separately.
_BENCH_TOL = float(os.environ.get("OQ_BENCH_TOL", "0.0"))
_MODE = os.environ.get("OQ_REWARD_MODE", "grounded").lower()
# Transient judge outage: retry the SAME immutable rollout before declaring unknown.
_JUDGE_RETRIES = int(os.environ.get("JUDGE_RETRIES", "2"))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _graded(route: float, tq: float) -> float:
    return max(_BASE, _clip01(_BASE + _W_ROUTE * float(route) + _W_TRAJ * float(tq)))


def _scored(score: float, reason: str, **extra) -> dict:
    out = {"score": float(score), "status": "scored", "verifier_ok": 1.0, "reward_reason": reason}
    out.update(extra)
    return out


def _unknown(reason: str, **extra) -> dict:
    """Verifier failure: the score is a PLACEHOLDER, never a real negative. Offline
    (rgate / score_trajectories) it stays 0.0 + status='unknown'; in training with
    OQ_REWARD_QUARANTINE=1 it becomes OQ_UNKNOWN_SENTINEL (-1.0) so the trainer-side
    advantage patch (scripts/_site/sitecustomize.py) zeroes the WHOLE sibling group
    instead of letting a fabricated 0.0 drag the GRPO baseline."""
    out = {"score": float(unknown_score()), "status": "unknown", "verifier_ok": 0.0,
           "reward_reason": reason}
    out.update(extra)
    return out


# --- deterministic judge-evidence verification (Gate-R v2, 2026-09-13) -----------
# The expanded R-gate (air/82 attempt 2) showed the bare judge CONFABULATES support:
# with every tool output blanked it still "cited" lines and credited 121/220 planted
# negatives (55%). Prompt exhortation did not stop it, so positives now require the
# judge's evidence to be MECHANICALLY verifiable: supporting_quotes must appear
# verbatim in the evidence text, must not come from an anachronistic bulletin, and a
# quoted row's own year label must match the asked years. All three checks are pure
# string/date arithmetic -- no judge trust involved.
_Q_YEAR_RE = re.compile(r"\b(19[3-9]\d|20[0-2]\d)\b")
_BULLETIN_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt")


def _cell_value_re(gt: str) -> "re.Pattern":
    """Cell-exact matcher for a value: '6' must not fire inside '1963' or '6.82';
    '0.13' may still match a cell showing '0.13%'. Mirrors the R-gate generator."""
    return re.compile(rf"(?<![\d\.,]){re.escape(str(gt).strip())}(?![\d\.,])")


def _verify_supporting_quotes(quotes: list, evidence_text: str) -> list[str]:
    """The subset of judge quotes that appear VERBATIM in the evidence text (offline:
    tool outputs only -- never agent prose; training: the serialized episode,
    best-effort). Returns matched evidence lines (one per verified quote)."""
    matched = []
    if not evidence_text:
        return matched
    lines = evidence_text.split("\n")
    for q in quotes or []:
        q = str(q).strip()
        if not q:
            continue
        for ln in lines:
            if q in ln or ln.strip() in q and len(ln.strip()) >= 12:
                matched.append(ln)
                break
        else:
            if q in evidence_text:
                matched.append(q)
    return matched


def _quote_has_value(matched_lines: list[str], gt: str) -> bool:
    """A verified quote only SUPPORTS the answer if the gold value is actually in it
    (cell-exact). The dominant v2 false-accept: the judge quoted a surviving
    search-snippet line that contained NO value at all."""
    if not gt:
        return True
    vre = _cell_value_re(gt)
    return any(vre.search(ln) for ln in matched_lines)


def _numeric_cells(line: str) -> int:
    return len(re.findall(r"(?<![\d\.,])\d[\d,]*(?:\.\d+)?\s*%?(?![\d\.,])", line))


_HDR_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.txt lines \d+-\d+ of \d+")


def _quote_source_year(ln: str, evidence_text: str) -> int | None:
    """The bulletin year for a matched quote line: filename ON the line (grep-hit
    format), else the nearest preceding '<file> lines A-B of N' chunk header -- the
    ONLY unambiguous associations. Never a random search-listing filename (that
    caused the v2 true-accept collapse)."""
    m = _BULLETIN_RE.search(ln)
    if m:
        return int(m.group(1))
    idx = evidence_text.find(ln)
    if idx > 0:
        window = evidence_text[max(0, idx - 4000):idx]
        hdrs = list(_HDR_RE.finditer(window))
        if hdrs:
            return int(hdrs[-1].group(1))
    return None


def _anachronistic(matched_lines: list[str], evidence_text: str, q_years: set[str]) -> bool:
    """True when the verified quote's source bulletin predates the asked YEAR: a
    bulletin from year Y-1 cannot contain year-Y actuals. The source year comes ONLY
    from unambiguous associations (_quote_source_year); when no file can be
    attributed we do NOT demote (that would false-fire on grounded controls)."""
    if not q_years:
        return False
    asked = min(int(y) for y in q_years)
    for ln in matched_lines:
        year = _quote_source_year(ln, evidence_text)
        if year is not None and year < asked:
            return True
    return False


def _period_label_mismatch(matched_lines: list[str], q_years: set[str]) -> bool:
    """True when a verified quote line carries year label(s) and NONE equals an asked
    year (the judge quoted '| 1935 | ...' for a question about 1934). Lines without
    any year label are neutral (period may live in a header)."""
    if not q_years:
        return False
    for ln in matched_lines:
        years = set(_Q_YEAR_RE.findall(ln))
        # ignore years that are part of a bulletin filename on the line
        for m in _BULLETIN_RE.finditer(ln):
            years.discard(m.group(1))
        if years and not (years & q_years):
            return True
    return False


def _assemble(strict_ok: bool, report: dict, verdict: dict | None, *,
              mode: str = "grounded", is_composite: bool = False,
              question: str = "", evidence_text: str | None = None,
              gt: str = "") -> dict:
    """Pure reward formula (no I/O) -- the single source of truth for the shape.

    strict_ok = strict_answer.strict_correct(...).valid
    report    = grounding.grounding_report(...) output
    verdict   = judge_prompt.parse_verdict(...) output:
                None (judge not called / no JSON) | {"status":"unknown"} | {"status":"ok", ...}
    question/evidence_text: when provided (production paths), enable the deterministic
                quote/anachronism/period-label verification of the judge's claimed
                support. Legacy unit tests omit them and skip that layer.
    """
    if not strict_ok:
        return _scored(0.0, "answer wrong/absent (strict gate)", route_score=None)

    src = report.get("source_identity", {}) or {}
    strong = src.get("strong_score")                       # gold file(s) READ or grep-HIT
    files_pulled = len(report.get("files_pulled") or [])   # ANY real content pulled (read/grep-hit)
    tq = (report.get("trajectory_quality", {}) or {}).get("score", 0.0) or 0.0
    no_retrieval = files_pulled == 0                        # deterministic fail-closed floor

    # --- answer-only control: strict correctness is the whole reward -----------------
    if mode == "answer":
        return _scored(1.0, "correct (answer-only mode)", route_score=None)

    # --- deterministic source gate, NO judge (cheap ablation) ------------------------
    if mode == "answer_source":
        if no_retrieval:
            return _scored(0.0, "correct but no real retrieval (answer_source)", route_score=0.0)
        if strong is not None and strong > 0:
            route, basis = float(strong), "det:gold-pulled"
        else:
            route, basis = _UNCLEAR_ROUTE, "det:unverified-pull"
        if route <= _LUCKY_ROUTE_MAX:
            return _scored(_LUCKY_SCORE, f"correct but route {route:.2f} lucky ({basis})", route_score=route)
        return _scored(_graded(route, tq), f"correct + source-grounded ({basis})",
                       route_score=route, traj_quality=float(tq))

    # --- grounded (default): the JUDGE is REQUIRED for any positive; FAIL-CLOSED ------
    if verdict is None:
        return _unknown("judge unavailable -> quarantine (no fail-open credit)", route_score=None)
    if verdict.get("status") != "ok":
        return _unknown(f"judge verdict unusable: {verdict.get('reason', 'unknown')}", route_score=None)

    route = float(verdict.get("route_score", 0.0))
    support = verdict.get("answer_supported_by_retrieved_cells")   # bool | None
    components = verdict.get("retrieved_all_components")           # bool | None

    if no_retrieval:
        return _scored(0.0, "correct but NO real retrieval (fail-closed floor)", route_score=route)
    # ELIGIBILITY IS FAIL-CLOSED. A positive REQUIRES the judge to AFFIRMATIVELY support the
    # answer. `support is None` = the judge omitted/garbled the flag -> the verdict is UNUSABLE
    # -> quarantine (never a silent positive). The confirmed P0 was `if support is False`,
    # which let a bare {"route_score":1} with NO support flag grade a full 1.0.
    if support is None:
        return _unknown("judge verdict has no usable answer_supported flag", route_score=route)
    if support is False:
        return _scored(_LUCKY_SCORE, "correct but judge: answer NOT supported by retrieved cells",
                       route_score=route)
    if is_composite:
        if components is None:
            return _unknown("composite but judge omitted retrieved_all_components", route_score=route)
        if components is False:
            return _scored(_LUCKY_SCORE, "correct but judge: not all components retrieved",
                           route_score=route)

    # --- Gate-R v2: deterministic verification of the judge's claimed support --------
    # The expanded R-gate showed the bare judge confabulates support (55% false-accept).
    # A positive now requires the judge's verbatim quotes to check out mechanically.
    if evidence_text is not None and support is True:
        quotes = verdict.get("supporting_quotes") or []
        matched = _verify_supporting_quotes(quotes, evidence_text)
        q_years = set(_Q_YEAR_RE.findall(question or ""))
        if not quotes:
            return _scored(_LUCKY_SCORE,
                           "judge claimed support but gave NO verbatim quote (evidence rule)",
                           route_score=route)
        if not matched:
            return _scored(_LUCKY_SCORE,
                           "judge quotes absent from tool outputs (confabulated evidence)",
                           route_score=route)
        if is_composite:
            if len({m.strip() for m in matched}) < 2:
                return _scored(_LUCKY_SCORE,
                               "composite but fewer than two distinct verified evidence lines",
                               route_score=route)
            if sum(1 for m in matched if _numeric_cells(m) >= 2) < 2:
                return _scored(_LUCKY_SCORE,
                               "composite but verified lines lack component cells",
                               route_score=route)
        elif gt and not _quote_has_value(matched, gt):
            # A verified quote only supports the answer if the GOLD VALUE is in it.
            return _scored(_LUCKY_SCORE,
                           "verified quote(s) do not contain the committed value",
                           route_score=route)
        if _anachronistic(matched, evidence_text, q_years):
            return _scored(_LUCKY_SCORE,
                           "anachronistic source: cited bulletin predates the asked period",
                           route_score=route)
        if _period_label_mismatch(matched, q_years):
            return _scored(_LUCKY_SCORE,
                           "quoted row's own period label does not match the asked period",
                           route_score=route)

    if route <= _LUCKY_ROUTE_MAX:
        return _scored(_LUCKY_SCORE, f"correct but route {route:.2f} <= {_LUCKY_ROUTE_MAX} (lucky/wrong-route)",
                       route_score=route)
    return _scored(_graded(route, tq), "correct + grounded (graded, judge)",
                   route_score=route, traj_quality=float(tq))


# --- judge client (grounding audit; mirrors judge_reward._call_judge) ---------
async def _call_grounding_judge(question, reference, source_files, trajectory_text, atom_spec=""):
    if _resolve_judge_url is None or _get_session is None:
        return None

    # The judge must see the WHOLE trajectory (owner decision). We do NOT head/tail
    # truncate -- that dropped the middle of most correct traces in the pilot and let the
    # judge approve cells it never saw. Instead, FAIL CLOSED: a trajectory longer than the
    # judge's context budget is quarantined (unknown), never graded on a partial view.
    traj_max = int(os.environ.get("OQ_JUDGE_TRAJ_MAX", "300000"))   # ~ JUDGE_MAX_MODEL_LEN chars
    if traj_max > 0 and len(trajectory_text or "") > traj_max:
        return {"status": "unknown",
                "reason": f"trajectory {len(trajectory_text)} chars exceeds judge budget {traj_max}; "
                          "quarantine rather than grade a truncated view (raise JUDGE_MAX_MODEL_LEN)"}

    import aiohttp

    model = os.environ.get("JUDGE_MODEL", "judge")
    api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", "10240"))
    temperature = float(os.environ.get("JUDGE_TEMPERATURE", "0"))
    # GLM-5.3 thinks UNCONDITIONALLY; its only knob is reasoning_effort in {low,high,max}.
    # "max" (default) overran the budget -> 38% non-parse; "high" is balanced. Passed as a
    # chat-template var (the GLM-5.3 template reads `reasoning_effort`). See the plan Section 6.6.
    effort = os.environ.get("JUDGE_REASONING_EFFORT", "high").lower()
    # 0 => send the WHOLE trajectory (no head/tail truncation); the ceiling above is the guard.
    max_chars = int(os.environ.get("OQ_JUDGE_TRAJ_CHARS", "0"))

    session = await _get_session()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": judge_prompt.JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt.build_user_prompt(
                question, reference, source_files, trajectory_text,
                atom_spec=atom_spec, max_chars=max_chars)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = _resolve_judge_url().rstrip("/") + "/chat/completions"
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    msg = (data["choices"][0].get("message") or {})
    content = msg.get("content") or ""
    verdict = judge_prompt.parse_verdict(content)
    if verdict is None and msg.get("reasoning_content"):
        verdict = judge_prompt.parse_verdict(msg["reasoning_content"])
    if (verdict is None or verdict.get("status") != "ok") and os.environ.get("JUDGE_DEBUG"):
        fr = data["choices"][0].get("finish_reason")
        rc = msg.get("reasoning_content") or ""
        print(f"[oq-judge-debug] verdict={verdict} finish={fr} len(content)={len(content)} "
              f"len(reasoning)={len(rc)}\n  content_tail={content[-300:]!r}", flush=True)
    return verdict


async def _judge_with_retry(*args, **kwargs):
    """Call the judge, retrying ONLY transient connection/timeout failures (the plan:
    retry the same immutable rollout). A parseable-but-unusable verdict is NOT retried
    (temperature 0 -> same reply); it is returned as an ``unknown`` verdict. Returns the
    verdict dict, an ``unknown`` verdict on persistent outage, or None if wiring absent."""
    import aiohttp
    last_exc = None
    for attempt in range(_JUDGE_RETRIES + 1):
        try:
            return await _call_grounding_judge(*args, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:  # transient
            last_exc = e
            if attempt < _JUDGE_RETRIES:
                await asyncio.sleep(min(2.0 * (attempt + 1), 5.0))
        except Exception as e:  # noqa: BLE001 - non-transient; do not spin
            last_exc = e
            break
    global _fail_logged
    if _fail_logged < 5 or os.environ.get("JUDGE_DEBUG"):
        _u = _resolve_judge_url() if _resolve_judge_url else "?"
        print(f"[oq-judge] call FAILED after retries ({type(last_exc).__name__}: {last_exc}) "
              f"url={_u} -> UNKNOWN (quarantine, NOT fail-open)", flush=True)
        _fail_logged += 1
    return {"status": "unknown", "reason": f"judge outage: {type(last_exc).__name__}"}


def _gold_from_extra(extra_info: dict) -> dict:
    return {
        "source_files": extra_info.get("source_files", "") or "",
        "atom_values": extra_info.get("atom_values"),
        "is_composite": extra_info.get("is_composite"),
        "truncated": extra_info.get("truncated"),
    }


def render_trajectory(steps: list) -> str:
    """Flatten structured steps into the text the judge reads (reasoning + each tool
    call with its args + the tool output). Used by the offline scorer; training already
    has the flat ``solution_str``."""
    parts = []
    for st in steps or []:
        if st.get("reasoning"):
            parts.append(str(st["reasoning"]))
        for c in (st.get("tool_calls") or []):
            parts.append(f"[tool_call] {c.get('name')}({c.get('args')})")
        for r in (st.get("tool_results") or []):
            parts.append(f"[tool_output:{r.get('name')}]\n{r.get('result')}")
    return "\n".join(parts)


_fail_logged = 0


def _out_fields(base: dict, ans_correct: bool, report: dict, verdict: dict | None) -> dict:
    """Attach diagnostics common to both entry points."""
    src = report.get("source_identity", {}) or {}
    base.update({
        "score": float(base["score"]),
        "acc": 1.0 if ans_correct else 0.0,                       # official benchmark correctness
        "source_seen": float(src.get("any_score") or 0.0),
        "source_pulled": float(src.get("strong_score") or 0.0),
        "route_score": float(base.get("route_score") or 0.0),
        "traj_quality": float((report.get("trajectory_quality", {}) or {}).get("score", 0.0) or 0.0),
        "judge_status": (verdict or {}).get("status", "none"),
        "n_calls": float(report.get("n_calls", 0)),
    })
    return base


async def compute_score(data_source: str = "", solution_str: str = "",
                        ground_truth=None, extra_info: dict | None = None, **kwargs):
    """verl rate_limited reward-manager contract (see judge_reward.py).

    NOTE: ``status == "unknown"`` marks a verifier failure whose 0.0 score is a
    placeholder; the batch-assembly / quarantine wiring (drop the sibling group, log the
    selection effect) is a training-loop integration TODO -- there is no OfficeQA train
    YAML yet -- but the reward now EXPOSES the signal instead of fabricating a positive.
    """
    extra_info = extra_info or {}
    question = str(extra_info.get("question", "") or "")
    gold = _gold_from_extra(extra_info)

    pred = _committed_answer(solution_str)
    strict_ok = bool(pred) and strict_correct(str(ground_truth), pred).valid
    try:
        bench_ok = bool(pred) and score_answer(str(ground_truth), pred, _BENCH_TOL) > 0
    except Exception:  # noqa: BLE001
        bench_ok = False

    report = grounding.grounding_report(solution_str, gold)
    is_comp = bool(report.get("is_composite"))

    verdict = None
    # Judge fires on strictly-correct answers in grounded mode -- including gold-file MISSES
    # (a correct answer grounded in a valid ALTERNATIVE bulletin can still be graded). Wrong
    # answers and the non-grounded modes never call it -> per-rollout cost stays bounded.
    if _MODE == "grounded" and strict_ok:
        verdict = await _judge_with_retry(
            question, ground_truth, gold["source_files"], solution_str,
            atom_spec=str(extra_info.get("atom_spec", "") or ""))

    out = _assemble(strict_ok, report, verdict, mode=_MODE, is_composite=is_comp,
                    question=question, evidence_text=solution_str, gt=str(ground_truth or ""))
    out = _out_fields(out, bench_ok, report, verdict)
    out["num_turns"] = float(extra_info.get("num_turns", 0) or 0)
    return out


async def score_record(rec: dict, judge_all: bool = False) -> dict:
    """Offline scoring for the validation pilot. rec fields:
        uid, question, gt, source_files, [atom_values], [atom_spec], [is_composite],
        [truncated], and EITHER `trajectory` (structured steps) or `solution_str`.
    `judge_all=True` calls the judge even on wrong answers (to validate the judge itself)."""
    steps = rec.get("trajectory")
    traj_for_ground = steps if steps is not None else (rec.get("solution_str") or "")
    traj_text = render_trajectory(steps) if steps is not None else (rec.get("solution_str") or "")
    if rec.get("pred"):
        pred = rec.get("pred")
    elif steps is not None:
        assistant_text = "\n".join(str(st.get("reasoning") or "") for st in steps)
        pred = _extractor.extract(assistant_text) or ""
    else:
        pred = _committed_answer(traj_text)
    gt = str(rec.get("gt", ""))
    strict_ok = bool(pred) and strict_correct(gt, pred).valid
    try:
        bench_ok = bool(pred) and score_answer(gt, pred, _BENCH_TOL) > 0
    except Exception:  # noqa: BLE001
        bench_ok = False

    gold = {"source_files": rec.get("source_files", "") or "",
            "atom_values": rec.get("atom_values"),
            "is_composite": rec.get("is_composite"),
            "truncated": rec.get("truncated")}
    report = grounding.grounding_report(traj_for_ground, gold)
    is_comp = bool(report.get("is_composite"))

    verdict = None
    if _MODE == "grounded" and (judge_all or strict_ok):
        verdict = await _judge_with_retry(
            rec.get("question", ""), gt, gold["source_files"], traj_text,
            atom_spec=str(rec.get("atom_spec", "") or ""))

    if steps is not None:
        # Offline evidence for quote verification: TOOL OUTPUTS ONLY -- the agent's
        # prose is not evidence (the F-class fabricated-citation hazard).
        evidence = "\n".join(str(r.get("result") or "") for st in steps
                             for r in (st.get("tool_results") or []))
    else:
        evidence = traj_text
    graded = _assemble(strict_ok, report, verdict, mode=_MODE, is_composite=is_comp,
                       question=str(rec.get("question", "") or ""), evidence_text=evidence,
                       gt=str(rec.get("gt", "") or ""))
    return {
        "uid": rec.get("uid"),
        "question": rec.get("question"),
        "gt": gt,
        "pred": pred,
        "strict_correct": strict_ok,
        "benchmark_correct": bench_ok,
        "source_files": gold["source_files"],
        "reward": float(graded["score"]),
        "reward_status": graded.get("status"),
        "verifier_ok": graded.get("verifier_ok"),
        "reward_reason": graded.get("reward_reason"),
        "grounding": report,
        "judge_verdict": verdict,
    }


if __name__ == "__main__":
    # Offline formula check (no judge server needed). See tests/test_reward_contract.py.
    grounded = [
        {"reasoning": "find 1940 defense", "tool_calls": [{"name": "grep_documents",
         "args": {"pattern": "National defense", "file_name": "treasury_bulletin_1941_01.txt"}}],
         "tool_results": [{"name": "grep_documents",
         "result": "treasury_bulletin_1941_01.txt:14: National defense ... 2,602"}]},
    ]
    empty = [
        {"reasoning": "grep the gold file but find nothing", "tool_calls": [{"name": "grep_documents",
         "args": {"pattern": "zzz", "file_name": "treasury_bulletin_1941_01.txt"}}],
         "tool_results": [{"name": "grep_documents", "result": "(no matches found)"}]},
    ]
    gold = {"source_files": "treasury_bulletin_1941_01.txt"}
    rep_g = grounding.grounding_report(grounded, gold)
    rep_e = grounding.grounding_report(empty, gold)
    ok = {"status": "ok", "route_score": 0.95, "answer_supported_by_retrieved_cells": True,
          "retrieved_all_components": True}
    print("correct + grounded (route .95)        ->", _assemble(True, rep_g, ok))
    print("correct + empty retrieval + judge DOWN ->", _assemble(True, rep_e, None))
    print("correct + judge says unsupported       ->",
          _assemble(True, rep_g, {"status": "ok", "route_score": 0.9, "answer_supported_by_retrieved_cells": False}))
    print("correct + judge verdict UNKNOWN        ->", _assemble(True, rep_g, {"status": "unknown", "reason": "NaN"}))
    print("WRONG answer                           ->", _assemble(False, rep_g, ok))
