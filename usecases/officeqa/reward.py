"""Miles' binary answer-and-evidence reward, using the co-located GLM Flash judge.

Only runtime-owned records are scored. Infrastructure faults raise the run's abort
channel and retain a fixed numeric metric schema, since verl swallows exceptions.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import run_control  # noqa: E402

import answer_check
import path_report

METRICS = (
    "score", "answer_correct", "report_valid", "abstention", "terminal_seen", "reference_valid",
    "value_valid", "judge_called", "judge_supported", "judge_unsupported", "judge_path_score",
    "judge_error", "infrastructure_error", "late_inputs", "observations", "delivered_observations",
    "compute_observations", "path_steps", "compute_steps", "compute_steps_with_inputs",
)
VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "path_status": {"type": "string", "enum": ["supported", "unsupported", "unknown"]},
        "path_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["path_status", "path_score", "issues"],
}
_sessions: dict = {}
_tokenizer = None


class JudgeError(RuntimeError):
    def __init__(self, kind: str, detail: str, *, retryable=False):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.retryable = kind, retryable


def resolve_judge_url() -> str:
    url = os.environ.get("JUDGE_BASE_URL")
    if not url:
        explicit = os.environ.get("JUDGE_ENDPOINT_FILE")
        rdv = run_control.rendezvous_dir()
        path = Path(explicit) if explicit else (rdv / "judge_endpoint" if rdv else None)
        if path and path.is_file():
            url = path.read_text().strip()
    if not url or not url.startswith(("http://", "https://")):
        raise JudgeError("configuration", "no valid judge endpoint/rendezvous")
    return url.rstrip("/")


def parse_verdict(content, finish_reason) -> dict:
    if finish_reason != "stop" or not isinstance(content, str):
        raise JudgeError("verdict", f"missing complete final-content verdict (finish={finish_reason})")
    try:
        obj = path_report.loads_strict(content)
    except (TypeError, ValueError) as error:
        raise JudgeError("verdict", str(error)) from error
    if not isinstance(obj, dict) or set(obj) != {"path_status", "path_score", "issues"}:
        raise JudgeError("verdict", "unexpected verdict fields")
    score = obj["path_score"]
    if (isinstance(score, bool) or (score is not None and
            (not isinstance(score, (int, float)) or not math.isfinite(score)))):
        raise JudgeError("verdict", "non-finite/non-numeric path score")
    if obj["path_status"] != "unknown" and score is None:
        raise JudgeError("verdict", "resolved verdict requires a numeric path score")
    if not isinstance(obj["issues"], list) or any(not isinstance(issue, str) for issue in obj["issues"]):
        raise JudgeError("verdict", "issues must be strings")
    verdict = path_report.parse_support_verdict(content)
    if verdict["status"] != "ok":
        raise JudgeError("verdict", verdict.get("reason", "unresolved support verdict"))
    return verdict


def check_cited_values(report, ledger):
    # Miles checks lookup values even if the policy declares dependencies.
    steps = [dataclasses.replace(step, depends_on=[]) for step in report.steps]
    return path_report.check_claimed_values(dataclasses.replace(report, steps=steps), ledger)


def judge_token_count(messages) -> int:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        path = os.environ.get("JUDGE_TOKENIZER_PATH") or os.environ.get("JUDGE_MODEL_PATH")
        if not path:
            raise JudgeError("configuration", "judge tokenizer path is required")
        _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ids = _tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                        add_generation_prompt=True, reasoning_effort="low")
    return len(ids)


def support_messages(record, report, ledger) -> tuple[list[dict], dict]:
    request = path_report.build_support_request(
        record["question"], record.get("question_requirements", ""), report, ledger)
    rows = sorted(ledger.observations.values(), key=lambda obs: obs.order)
    cited = {step.observation for step in report.steps}

    def row(obs):
        return {"observation_id": obs.observation_id, "tool": obs.tool, "args": obs.args,
                "outcome": obs.outcome, "delivered": obs.delivered, "delivered_text": obs.delivered_text or ""}

    # All cited evidence comes first, intact. Uncited history can be omitted to fit.
    history = [row(obs) for obs in rows if obs.observation_id in cited]
    request["tool_history"] = history
    limit = int(os.environ.get("JUDGE_MAX_MODEL_LEN", "16384")) - int(os.environ.get("JUDGE_MAX_TOKENS", "4096")) - 64

    def rendered():
        system, user = path_report.render_support_prompt(request)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    messages = rendered()
    tokens = judge_token_count(messages)
    if tokens > limit:
        raise JudgeError("context", f"all cited evidence needs {tokens} prompt tokens; budget={limit}")
    for obs in rows:
        if obs.observation_id in cited:
            continue
        history.append(row(obs))
        candidate = rendered()
        count = judge_token_count(candidate)
        if count > limit:
            history.pop()
        else:
            messages, tokens = candidate, count
    receipt = {"prompt_tokens": tokens, "prompt_budget": limit, "cited_observations": len(cited),
               "omitted_observations": len(rows) - len(history), "cited_evidence_clipped": False,
               "request_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
    return messages, receipt


async def get_session():
    import aiohttp
    loop = asyncio.get_running_loop()
    for old in [old for old in _sessions if old.is_closed()]:
        _sessions.pop(old, None)
    if loop not in _sessions or _sessions[loop].closed:
        _sessions[loop] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(os.environ.get("JUDGE_TIMEOUT", "90"))))
    return _sessions[loop]


async def close_sessions():
    session = _sessions.pop(asyncio.get_running_loop(), None)
    if session and not session.closed:
        await session.close()


async def call_judge(messages) -> dict:
    import aiohttp
    payload = {
        "model": os.environ.get("JUDGE_MODEL", "judge"), "messages": messages, "temperature": 0,
        "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "4096")),
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "response_format": {"type": "json_schema", "json_schema": {"name": "support", "schema": VERDICT_SCHEMA, "strict": True}},
    }
    session = await get_session()
    url = resolve_judge_url() + "/chat/completions"
    for attempt in range(int(os.environ.get("JUDGE_RETRIES", "1")) + 1):
        try:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    raise JudgeError("transport", f"HTTP {response.status}: {(await response.text())[:250]}",
                                     retryable=response.status == 429 or response.status >= 500)
                data = await response.json(content_type=None)
            try:
                choice = data["choices"][0]
                content = (choice.get("message") or {}).get("content")
                verdict = parse_verdict(content, choice.get("finish_reason"))
            except (KeyError, IndexError, TypeError, AttributeError) as error:
                raise JudgeError("verdict", "malformed completion response") from error
            return {**verdict, "endpoint": url, "usage": data.get("usage"), "raw_content": content}
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            failure = JudgeError("transport", f"{type(error).__name__}: {error}", retryable=True)
        except JudgeError as error:
            failure = error
        if not failure.retryable or attempt >= int(os.environ.get("JUDGE_RETRIES", "1")):
            raise failure
        await asyncio.sleep(1 + attempt)
    raise AssertionError("unreachable")


def write_trace(record, result, details):
    directory = os.environ.get("OQ_TRACE_DIR")
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(record.get("episode_id", "missing") if isinstance(record, dict) else "missing").encode()).hexdigest()[:16]
    path = root / f"{identity}-{uuid4().hex}.json"
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps({"run_id": os.environ.get("RUN_ID"), "git_sha": os.environ.get("GIT_SHA"),
                                     "record": record, "metrics": result, **details}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


async def compute_score(data_source="", solution_str="", ground_truth=None, extra_info=None, **kwargs):
    result = dict.fromkeys(METRICS, 0.0)
    details = {"category": "malformed", "reason": "no terminal submission", "verdict": None}
    record = None
    started = time.monotonic()
    try:
        raw = (extra_info or {}).get("officeqa_record")
        record = path_report.loads_strict(raw) if isinstance(raw, str) else raw
        if not isinstance(record, dict) or not isinstance(record.get("observations"), list):
            raise RuntimeError("runtime-owned officeqa_record is missing or malformed")
        expected_question = (extra_info or {}).get("question")
        if expected_question and expected_question != record.get("question"):
            raise RuntimeError("capture question does not match reward metadata")
        record["question_requirements"] = (extra_info or {}).get("question_requirements", "")
        ledger = path_report.EpisodeLedger.from_record(record)
        result.update(terminal_seen=float(bool(record.get("terminal_seen"))), observations=float(len(ledger.observations)),
                      delivered_observations=float(sum(obs.delivered for obs in ledger.observations.values())),
                      compute_observations=float(sum(obs.is_compute for obs in ledger.observations.values())))
        report = path_report.parse_report(record.get("terminal_text", "") if record.get("terminal_seen") else "")
        result.update(report_valid=float(report.ok), abstention=float(report.is_abstention), path_steps=float(len(report.steps)))
        refs = path_report.check_references(report, ledger)
        result["late_inputs"] = float(sum(issue.code == "impossible_chronology" for issue in refs.issues))
        kept = [issue for issue in refs.issues if issue.code != "impossible_chronology"]
        refs = path_report.RefCheck(not kept, kept)
        values = check_cited_values(report, ledger)
        result.update(reference_valid=float(report.ok and refs.valid), value_valid=float(report.ok and values.valid))
        computes = [step for step in report.steps if (obs := ledger.get(step.observation)) is not None and obs.is_compute]
        result.update(compute_steps=float(len(computes)), compute_steps_with_inputs=float(sum(bool(step.depends_on) for step in computes)))
        correct, answer_reason = False, "no answer"
        if report.ok:
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                raise RuntimeError("missing gold answer")
            correct, answer_reason = answer_check.check(report.answer, ground_truth, tolerance=0.0)
            result["answer_correct"] = float(correct)
        if report.is_abstention:
            details.update(category="abstention", reason="explicit DATA NOT AVAILABLE")
        elif not report.ok:
            details["reason"] = "; ".join(report.errors)
        elif not refs.valid:
            details.update(category="invalid_path", reason=", ".join(refs.codes))
        elif not values.valid:
            details.update(category="fabricated", reason=", ".join(values.codes))
        else:
            details.update(category="wrong", reason=answer_reason)
            if correct:
                messages, receipt = await asyncio.to_thread(support_messages, record, report, ledger)
                details["judge_request"] = receipt
                result["judge_called"] = 1.0
                verdict = await asyncio.wait_for(call_judge(messages), timeout=float(os.environ.get("JUDGE_DEADLINE_S", "180")))
                details["verdict"] = verdict
                supported = verdict["path_status"] == path_report.SUPPORTED
                result.update(score=float(supported), judge_supported=float(supported), judge_unsupported=float(not supported),
                              judge_path_score=float(verdict["path_score"]))
                details.update(category="correct" if supported else "unsupported", reason="; ".join(verdict["issues"]))
    except Exception as error:  # verl requires the same keys on failures too.
        result["score"] = 0.0
        result["infrastructure_error"] = 1.0
        result["judge_error"] = float(isinstance(error, (JudgeError, asyncio.TimeoutError)))
        details.update(category="infrastructure", reason=f"{type(error).__name__}: {error}")
        run_control.request_abort(details["reason"], "usecases/officeqa/reward.py")
    details["elapsed_s"] = time.monotonic() - started
    try:
        await asyncio.to_thread(write_trace, record, result, details)
    except Exception as error:
        result.update(score=0.0, infrastructure_error=1.0)
        run_control.request_abort(f"trace write failed: {error}", "usecases/officeqa/reward.py")
    print(json.dumps({"officeqa_reward": details["category"], "score": result["score"],
                      "episode_id": record.get("episode_id") if isinstance(record, dict) else None,
                      "reason": details["reason"][:160]}), flush=True)
    return result
