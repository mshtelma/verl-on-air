#!/usr/bin/env python3
"""LLM-as-judge reward for the GSM8K tool-agent (fully-async / disaggregated).

Wired in via verl's rate-limited reward manager (built for external-API judges)::

    reward.reward_manager.name=rate_limited
    reward.custom_reward_function.path=/app/scripts/reward/judge_reward.py
    reward.custom_reward_function.name=compute_score
    +reward.max_concurrent=32          # DEFAULT IS 1 (serial) -> must raise it
    +reward.max_rpm=...  +reward.max_tpm=...  +reward.timeout=60

CONTRACT (verified against verl/experimental/reward_loop/reward_manager/limited.py:378)
--------------------------------------------------------------------------------------
The manager awaits this directly when it is `async def` (which it is), passing
KEYWORD args only:

    async compute_score(data_source=str, solution_str=str, ground_truth=Any,
                        extra_info=dict, **kwargs) -> dict | float

`solution_str` is the FULLY-DECODED trajectory of the tool-agent rollout — the
model's interleaved reasoning, its <tool_call> blocks, the calculator's tool
responses, and the final `#### <answer>` — so the judge grades the whole episode,
not just a bare answer. Return a dict with a "score" key (the reward GRPO
optimises); every other key is logged to MLflow as its own curve.

ENDPOINT RESOLUTION — robust to Ray not propagating env into reward actors
--------------------------------------------------------------------------
This module is imported inside verl's reward-loop workers, which are Ray actors
on the training nodes. In air/53 the judge lives on OTHER nodes and the dispatcher
publishes its URL to a shared UC rendezvous file, then `export JUDGE_BASE_URL`s it
into the training driver. But Ray does NOT reliably carry a driver `export` into
actor processes, and — critically — reading `JUDGE_BASE_URL` at *import* time (as an
earlier version did) froze it to the localhost default before the real URL arrived.
That silently pointed every judge call at 127.0.0.1:8000, which nothing answers, so
the broad except below fell back to the rule score with judge_ok=0.0 and the run
looked green while the judge sat idle (air/53 run3).

Fix: every knob is read at CALL time, and the URL is resolved from the rendezvous
FILE whose path is rebuilt from container-level vars (RENDEZVOUS_ROOT from the job
YAML; MASTER_ADDR/MASTER_PORT from the runtime) that reach EVERY process regardless
of how Ray builds an actor's runtime_env. The resolved URL is logged once so what
the worker actually used is always visible.

DESIGN — judge, validated by ground truth
-----------------------------------------
GSM8K has an exact numeric answer, so we can compute a RULE score (exact match)
for free every step. We use it two ways:
  1. As a SAFETY NET — the rate-limited manager silently returns 0.0 on any judge
     exception/timeout, which would quietly zero the reward and collapse training
     if the judge endpoint hiccups. Instead we fall back to the rule score, so a
     transient judge outage degrades to plain RLVR rather than to noise.
  2. As a VALIDATION signal — we log `acc` (rule), `judge_score`, and
     `judge_agree` (do they agree?) every step. Watching judge-vs-gold agreement
     is how we prove the judge reward is sound BEFORE trusting the same machinery
     on a task that has no ground truth. This is the whole point of starting the
     LLM-judge experiment on a dataset that also has a gold answer.

REWARD_SOURCE (env) selects what is actually optimised:
  judge (default) — the LLM judge's 0..1 score.
  rule            — pure RLVR exact-match (a control / baseline).
  blend           — JUDGE_BLEND_ALPHA*judge + (1-alpha)*rule.

The judge is a self-hosted vLLM OpenAI-compatible server; point JUDGE_BASE_URL at
it (or let resolution find the rendezvous file). It is reference-GUIDED (given the
gold answer) and scores correctness of the final answer plus soundness of the
reasoning/tool use, returning strict JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

# --- static defaults (NEVER capture os.environ at import; read at call time) --
_DEFAULT_JUDGE_URL = "http://127.0.0.1:8000/v1"

_JUDGE_SYSTEM = (
    "You are a strict grader for math problems (grade-school through "
    "competition-level). You are given the question, a reference final answer, and "
    "a student's full working (which may include tool calls to a calculator and the "
    "tool's results). Decide whether the student's FINAL answer is MATHEMATICALLY "
    "EQUIVALENT to the reference (e.g. 1/2, 0.5 and \\frac{1}{2} are equivalent; "
    "2\\sqrt{2} and \\sqrt{8} are equivalent; x=3 and 3 are equivalent) — the "
    "reference may be a LaTeX expression — and assess the soundness of the "
    "reasoning. Respond with ONLY a JSON object and nothing else, of the form: "
    '{"correct": true|false, "score": <float 0..1>, "reason": "<short>"}. '
    "Use score 1.0 for a correct (equivalent) final answer with sound reasoning; "
    "~0.7 for a correct answer with flawed/lucky reasoning; ~0.2 for a wrong answer "
    "that is close or on the right track; 0.0 for a wrong answer or no clear answer."
)


# --- endpoint resolution ------------------------------------------------------
_resolved_url: str | None = None   # cache of a NON-default resolution
_logged_default = False            # so a default fallback logs at most once
_fail_logged = 0                   # count of judge-call failures logged loudly


def _resolve_judge_url() -> str:
    """Resolve the judge OpenAI base URL at call time.

    Priority (first hit wins):
      1. ``JUDGE_BASE_URL`` env — honour any explicit override (incl. localhost).
      2. ``JUDGE_ENDPOINT_FILE`` env — read the rendezvous file it points at.
      3. Reconstruct ``$RENDEZVOUS_ROOT/${MASTER_ADDR}_${MASTER_PORT}/judge_endpoint``
         from container-level vars and read it. These reach every process (they
         are set before Ray starts), so this path survives Ray dropping the
         dispatcher's ``export JUDGE_BASE_URL``.
      4. Localhost default.

    A successful (non-default) resolution is cached and logged once. A default
    fallback is NOT cached — so if the rendezvous file appears a moment later,
    a subsequent call still finds the real endpoint.
    """
    global _resolved_url, _logged_default
    if _resolved_url is not None:
        return _resolved_url

    url: str | None = None
    source = ""

    env_url = os.environ.get("JUDGE_BASE_URL")
    if env_url:
        url, source = env_url, "env:JUDGE_BASE_URL"
    else:
        cand = os.environ.get("JUDGE_ENDPOINT_FILE")
        src = "env:JUDGE_ENDPOINT_FILE"
        if not cand:
            root = os.environ.get("RENDEZVOUS_ROOT")
            addr = os.environ.get("MASTER_ADDR")
            port = os.environ.get("MASTER_PORT")
            if root and addr and port:
                cand = os.path.join(root, f"{addr}_{port}", "judge_endpoint")
                src = f"rendezvous:{cand}"
        if cand:
            try:
                with open(cand) as fh:
                    file_url = fh.read().strip()
                if file_url:
                    url, source = file_url, src
            except OSError:
                pass

    if not url:
        # Misconfigured (or genuine single-node localhost). Don't cache: the
        # rendezvous file may still appear. Log once so it is not silent.
        if not _logged_default:
            print(
                "[judge] WARNING: no JUDGE_BASE_URL and no readable rendezvous "
                f"endpoint (RENDEZVOUS_ROOT={os.environ.get('RENDEZVOUS_ROOT')!r} "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')!r} "
                f"MASTER_PORT={os.environ.get('MASTER_PORT')!r}); "
                f"falling back to {_DEFAULT_JUDGE_URL} — judge calls will fail if "
                "nothing serves there.",
                flush=True,
            )
            _logged_default = True
        return _DEFAULT_JUDGE_URL

    _resolved_url = url
    print(f"[judge] reward worker resolved JUDGE_BASE_URL={url}  (source={source})", flush=True)
    return url


# --- answer extraction / rule score ------------------------------------------
# Handles BOTH task families with one code path:
#   GSM8K  -> `#### <number>`,  ground_truth a bare number  -> numeric compare
#   MATH   -> `\boxed{<expr>}`, ground_truth a LaTeX expr   -> latex-equivalence
# The rule/`acc` is a VALIDATION signal (REWARD_SOURCE=judge optimises the judge),
# but on the ~judge-timeout fallback it also becomes the reward, and it is our
# ground-truth "is the answer actually right" curve — so it must be correct on MATH,
# whose answers (\frac, \sqrt, 0.5==1/2) do not exact-match. Numeric answers still
# take the exact float path, so GSM8K behaviour is unchanged.
_HASH_RE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")
_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except (ValueError, AttributeError):
        return None


def _last_boxed(s: str) -> str | None:
    """Content of the LAST \\boxed{...}/\\fbox{...}, brace-balanced (so
    \\boxed{\\frac{1}{2}} -> `\\frac{1}{2}`, not `\\frac{1`). None if absent."""
    key = "\\boxed"
    i = s.rfind(key)
    if i < 0:
        key = "\\fbox"
        i = s.rfind(key)
        if i < 0:
            return None
    j = i + len(key)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    depth, start = 0, j
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:j]
        j += 1
    return None


# --- LaTeX answer normalization + equivalence (standard Hendrycks MATH is_equiv) --
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    if len(substr) >= 2:
                        a, b = substr[0], substr[1]
                        if b != "{":
                            new_str += "{" + a + "}{" + b + "}" + substr[2:]
                        else:
                            new_str += "{" + a + "}" + substr[1:]
                    else:
                        new_str += substr
                except Exception:  # noqa: BLE001
                    return string
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        int(a)
        int(b)
        return "\\frac{" + a + "}{" + b + "}"
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ ...}" trailing units, as in the Hendrycks normalizer
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    string = string.strip("{}")
    return string


def _is_equiv(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return _strip_string(a) == _strip_string(b)
    except Exception:  # noqa: BLE001
        return a.strip() == b.strip()


def _math_equiv(pred: str | None, gt: Any) -> bool:
    if pred is None or gt is None:
        return False
    p, g = str(pred).strip(), str(gt).strip()
    if not p or not g:
        return False
    fp, fg = _to_float(p), _to_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-4        # exact numeric path (GSM8K unchanged)
    return _is_equiv(p, g)                 # LaTeX-equivalence path (MATH)


def _extract_pred_str(solution_str: str) -> str | None:
    """The model's final answer AS A STRING: prefer `#### N`, then the last
    \\boxed{...}, then the last number in the text."""
    hits = _HASH_RE.findall(solution_str)
    if hits:
        return hits[-1]
    boxed = _last_boxed(solution_str)
    if boxed is not None:
        return boxed
    nums = _NUM_RE.findall(solution_str)
    return nums[-1] if nums else None


def _rule_score(solution_str: str, ground_truth: Any) -> float:
    pred = _extract_pred_str(solution_str)
    return 1.0 if _math_equiv(pred, ground_truth) else 0.0


# --- judge client (shared aiohttp session, keyed by event loop) --------------
_sessions: dict[Any, Any] = {}


async def _get_session():
    import aiohttp

    loop = asyncio.get_event_loop()
    sess = _sessions.get(loop)
    if sess is None or sess.closed:
        timeout = float(os.environ.get("JUDGE_TIMEOUT", "60"))
        sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
        _sessions[loop] = sess
    return sess


def _judge_user_prompt(question: str, trajectory: str, reference: Any) -> str:
    max_chars = int(os.environ.get("JUDGE_TRAJECTORY_CHARS", "8000"))  # bound judge input
    if len(trajectory) > max_chars:
        # keep the tail — the final answer and last reasoning are what matter most
        trajectory = "...(truncated)...\n" + trajectory[-max_chars:]
    return (
        f"[Question]\n{question}\n\n"
        f"[Reference final answer]\n{reference}\n\n"
        f"[Student's full working]\n{trajectory}\n\n"
        "Grade it now. Return ONLY the JSON object."
    )


def _parse_judge(content: str) -> float | None:
    """Extract a 0..1 score from the judge's reply, tolerant of code fences /
    stray prose around the JSON."""
    content = content.strip()
    # Try the first {...} block as JSON.
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "score" in obj:
                return max(0.0, min(1.0, float(obj["score"])))
            if "correct" in obj:
                return 1.0 if bool(obj["correct"]) else 0.0
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback: a bare float in [0,1].
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", content)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return None


async def _call_judge(question: str, trajectory: str, reference: Any) -> float | None:
    # All knobs read at CALL time (see module docstring: import-time capture was the bug).
    model = os.environ.get("JUDGE_MODEL", "judge")
    api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")  # vLLM ignores it; header still sent
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))  # room to THINK + emit JSON
    temperature = float(os.environ.get("JUDGE_TEMPERATURE", "0"))
    # GLM/Qwen chat templates enable a <think> phase by default; a judge wants a fast,
    # terse JSON verdict, so disable it (the parser tolerates either).
    disable_thinking = os.environ.get("JUDGE_DISABLE_THINKING", "1") == "1"

    session = await _get_session()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _judge_user_prompt(question, trajectory, reference)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        # Honored by GLM/Qwen chat templates on both vLLM and SGLang.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}"}
    url = _resolve_judge_url().rstrip("/") + "/chat/completions"
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    choice = data["choices"][0]
    msg = choice.get("message", {}) or {}
    content = msg.get("content") or ""
    # Reasoning models (GLM-5.3) split output: <think> -> reasoning_content, the
    # final answer -> content. With --reasoning-parser the JSON verdict lands in
    # content, but if the model ran out of tokens mid-think content can be empty,
    # so fall back to parsing reasoning_content too.
    reasoning = msg.get("reasoning_content") or ""
    if os.environ.get("JUDGE_DEBUG"):
        print(f"[judge-debug] url={url} finish_reason={choice.get('finish_reason')} "
              f"len(content)={len(content)} len(reasoning)={len(reasoning)}\n"
              f"[judge-debug] content={content[:600]!r}\n"
              f"[judge-debug] reasoning={reasoning[:400]!r}", flush=True)
    score = _parse_judge(content)
    if score is None and reasoning:
        score = _parse_judge(reasoning)
    return score


async def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    extra_info = extra_info or {}
    question = str(extra_info.get("question", "") or "")
    reward_source = os.environ.get("REWARD_SOURCE", "judge").lower()  # judge | rule | blend

    rule = _rule_score(solution_str, ground_truth)
    n_tool_calls = float(solution_str.count("<tool_call>"))
    num_turns = float(extra_info.get("num_turns", 0) or 0)

    judge_ok = 1.0
    judge = None
    if reward_source in ("judge", "blend"):
        try:
            judge = await _call_judge(question, solution_str, ground_truth)
        except Exception as e:  # noqa: BLE001 - never let a judge hiccup crash the step
            # Loud on the first few failures (NOT gated on JUDGE_DEBUG): run3's
            # bug hid here for a whole run. Include the resolved URL so a
            # misconfigured endpoint is diagnosable straight from the logs.
            global _fail_logged
            if _fail_logged < 5 or os.environ.get("JUDGE_DEBUG"):
                print(f"[judge] call FAILED ({type(e).__name__}: {e}) "
                      f"url={_resolve_judge_url()} -> falling back to rule score", flush=True)
                _fail_logged += 1
            judge = None
        if judge is None:
            judge_ok = 0.0  # call failed or reply unparseable -> fall back to rule

    if judge is None:
        judge_for_log = rule  # so the logged judge curve stays meaningful on fallback
        score = rule
    else:
        judge_for_log = judge
        if reward_source == "judge":
            score = judge
        elif reward_source == "blend":
            alpha = float(os.environ.get("JUDGE_BLEND_ALPHA", "0.5"))
            score = alpha * judge + (1.0 - alpha) * rule
        else:  # "rule"
            score = rule

    judge_bin = 1.0 if judge_for_log >= 0.5 else 0.0
    return {
        "score": float(score),                       # <- optimised by GRPO
        "judge_score": float(judge_for_log),         # raw judge 0..1 (=rule on fallback)
        "acc": float(rule),                          # ground-truth exact match (validation)
        "judge_agree": float(judge_bin == rule),     # judge vs gold agreement
        "judge_ok": judge_ok,                        # 1.0 if the judge answered, else 0.0
        "n_tool_calls": n_tool_calls,                # observable agentic signal
        "num_turns": num_turns,
    }


# ---------------------------------------------------------------------------
# Local sanity check:
#   python3 scripts/reward/judge_reward.py              # offline: extraction+rule
#   JUDGE_BASE_URL=http://host:8000/v1 JUDGE_MODEL=... \
#     python3 scripts/reward/judge_reward.py --live     # also hits the judge once
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    traj_correct = (
        "<think>18 eggs, eats 3, bakes 4, sells the rest.</think>"
        "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"18 - 3 - 4\"}}</tool_call>"
        "The remainder is 11, at $2 each.\n"
        "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"11 * 2\"}}</tool_call>"
        "So she makes $22.\n#### 22"
    )
    traj_wrong = "<think>guessing</think> I think it is #### 30"

    print("== offline: extraction + rule score ==")
    cases = [
        # GSM8K numeric path (unchanged behaviour)
        ("gsm8k correct #### 22", traj_correct, "22", 1.0),
        ("gsm8k wrong   #### 30", traj_wrong, "22", 0.0),
        ("gsm8k boxed fallback", r"the answer is \boxed{22}", "22", 1.0),
        # MATH latex-equivalence path
        ("math frac ==", r"so the answer is \boxed{\frac{1}{2}}", r"\frac{1}{2}", 1.0),
        ("math dfrac==frac", r"final: \boxed{\dfrac{1}{2}}", r"\frac{1}{2}", 1.0),
        ("math 0.5==1/2", r"hence \boxed{0.5}", r"\frac{1}{2}", 1.0),
        ("math 2sqrt2==sqrt-spacing", r"we get \boxed{2\sqrt{2}}", r"2\sqrt2", 1.0),
        ("math nested boxed", r"thus \boxed{\frac{3}{4} + \frac{1}{4}}", r"\frac{3}{4}+\frac{1}{4}", 1.0),
        ("math x=3 == 3", r"so \boxed{3}", r"x=3", 1.0),
        ("math wrong", r"answer \boxed{3}", r"\frac{1}{2}", 0.0),
        ("math no-answer", r"I am not sure how to proceed.", r"\frac{1}{2}", 0.0),
    ]
    n_fail = 0
    for label, traj, gt, want in cases:
        got = _rule_score(traj, gt)
        ok = got == want
        n_fail += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {label:24s} rule={got} (want {want}) "
              f"pred={_extract_pred_str(traj)!r}")
    print(f"== {len(cases) - n_fail}/{len(cases)} passed ==")
    if n_fail and "--live" not in sys.argv:
        sys.exit(1)

    if "--live" in sys.argv:
        os.environ["JUDGE_DEBUG"] = "1"  # dump the raw judge reply so a judge_ok=0 is diagnosable
        print(f"\n== live: calling judge at {_resolve_judge_url()} (model={os.environ.get('JUDGE_MODEL', 'judge')}) ==")
        out = asyncio.run(
            compute_score(
                data_source="openai/gsm8k",
                solution_str=traj_correct,
                ground_truth="22",
                extra_info={"question": "Janet's ducks lay 18 eggs...", "num_turns": 5},
            )
        )
        print(json.dumps(out, indent=2))
