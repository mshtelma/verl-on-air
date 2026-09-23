#!/usr/bin/env python3
"""The contract every use case's eval.py follows, so a number in an eval artifact can be trusted.

An eval that swallows an inference or retrieval outage and scores it as a wrong answer produces a
valid-looking accuracy -- and a different outage rate between the baseline and the checkpoint run
manufactures a "learning delta" in either direction. So an eval built on this module:

  * checks READINESS first -- the served model is listed at /v1/models, and the use case's own
    probe (e.g. one real retrieval) works -- or exits 2 before running a single question;
  * gives every question a STATUS: `scored` (the model's doing -- right, wrong or no answer), or an
    infrastructure failure that is NOT scored: infra_inference, infra_retrieval, infra_tool,
    infra_harness, context_limit;
  * retries transient failures (connection, timeout, HTTP 429/5xx) but not deterministic ones;
  * is VALID only if it ran the expected question set (EVAL_EXPECT_N) and infrastructure failures
    stayed within EVAL_MAX_INFRA_ERRORS (default 0). An invalid run still writes its artifact,
    marked `"valid": false` with the reasons, and exits 1;
  * writes artifacts atomically and never over an existing file (EVAL_OVERWRITE=1 to force); each
    carries the served model's identity (EVAL_MODEL_IDENTITY_FILE, written by serve_and_eval.sh),
    the dataset fingerprint, a digest of the question ids, and the eval policy;
  * writes each question's record as it finishes, one closed file per question under
    <EVAL_OUT>.parts/ -- on a UC Volume a file is durable once closed -- so a timeout keeps the
    work already done.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

INFRA_KINDS = ("infra_inference", "infra_retrieval", "infra_tool", "infra_harness", "context_limit")
_CONTEXT_MARKERS = ("maximum context length", "context length", "too long", "prompt is too long")


class InfraError(Exception):
    """A failure that is not the model's doing. `kind` is one of INFRA_KINDS."""

    def __init__(self, kind: str, detail: str):
        assert kind in INFRA_KINDS, kind
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "")
    return default if raw.strip() == "" else int(raw)


# --- HTTP to the served model ------------------------------------------------------------------
def classify_http(status: int, body: str) -> tuple[str, bool]:
    """(kind, retryable) for an HTTP error from the inference server."""
    if status == 429 or status >= 500:
        return "infra_inference", True
    if status == 400 and any(m in body.lower() for m in _CONTEXT_MARKERS):
        return "context_limit", False
    return "infra_inference", False  # any other 4xx is a harness bug, never the model's answer


async def post_json(session, url: str, payload: dict, *, retries: int | None = None,
                    backoff_s: float = 0.5) -> dict:
    """POST and return the JSON body; transient failures are retried, then raise InfraError."""
    import aiohttp

    retries = _env_int("EVAL_HTTP_RETRIES", 4) if retries is None else retries
    for attempt in range(retries + 1):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    kind, retryable = classify_http(resp.status, body)
                    if retryable and attempt < retries:
                        await asyncio.sleep(backoff_s * (attempt + 1))
                        continue
                    raise InfraError(kind, f"HTTP {resp.status} from {url}: {body[:300]}")
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < retries:
                await asyncio.sleep(backoff_s * (attempt + 1))
                continue
            raise InfraError("infra_inference", f"{type(e).__name__}: {e} ({url})") from None
    raise AssertionError("unreachable")


async def check_served_model(session, base_url: str, served: str) -> None:
    """Readiness: the name we will request must be one the server actually serves."""
    try:
        async with session.get(f"{base_url}/models") as resp:
            data = await resp.json(content_type=None) if resp.status == 200 else {}
    except Exception as e:  # noqa: BLE001
        raise InfraError("infra_inference", f"{base_url}/models unreachable: {e}") from None
    ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
    if served not in ids:
        raise InfraError("infra_inference", f"served model {served!r} not listed at {base_url}/models (got {ids})")


def fatal_not_ready(what: str, err: Exception) -> None:
    print(f"[eval] FATAL: {what} is not ready ({err}) -- no question was run.", file=sys.stderr, flush=True)
    raise SystemExit(2)


# --- identities --------------------------------------------------------------------------------
def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def ids_digest(ids: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(i) for i in ids).encode()).hexdigest()


def model_identity() -> dict[str, Any] | None:
    """What serve_and_eval.sh verified and served (verify_checkpoint.py identity JSON)."""
    p = os.environ.get("EVAL_MODEL_IDENTITY_FILE")
    if not p or not Path(p).is_file():
        return None
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- artifacts ---------------------------------------------------------------------------------
def overwrite_allowed() -> bool:
    return os.environ.get("EVAL_OVERWRITE", "0").strip().lower() in ("1", "true", "yes")


def refuse_overwrite(*paths: str | None) -> None:
    """Exit 2 if an artifact would replace an earlier eval's -- they are evidence, not scratch."""
    if overwrite_allowed():
        return
    for p in paths:
        if p and (Path(p).exists() or Path(f"{p}.parts").exists()):
            print(f"[eval] FATAL: {p} already exists -- name this run's artifact differently, or "
                  "set EVAL_OVERWRITE=1 to replace it.", file=sys.stderr, flush=True)
            raise SystemExit(2)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)


class PartsWriter:
    """One closed JSON file per finished question under <out>.parts/ (durable on a UC Volume)."""

    def __init__(self, out: str | None):
        self.dir = Path(f"{out}.parts") if out else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, key: Any, record: dict) -> None:
        if self.dir:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))
            write_json_atomic(self.dir / f"{safe}.json", record)


def verdict(results: list[dict], *, n_loaded: int, n_expected: int | None) -> dict[str, Any]:
    """Validity of a finished run: the question set and the infrastructure error budget."""
    infra = {k: sum(1 for r in results if r.get("status") == k) for k in INFRA_KINDS}
    n_infra = sum(infra.values())
    reasons = []
    if n_loaded == 0:
        reasons.append("no questions were loaded")
    if n_expected is not None and n_loaded != n_expected:
        reasons.append(f"loaded {n_loaded} questions, expected {n_expected} (EVAL_EXPECT_N)")
    budget = _env_int("EVAL_MAX_INFRA_ERRORS", 0)
    if n_infra > budget:
        reasons.append(f"{n_infra} question(s) hit infrastructure failures (budget "
                       f"EVAL_MAX_INFRA_ERRORS={budget}): {', '.join(f'{k}={v}' for k, v in infra.items() if v)}")
    return {"valid": not reasons, "invalid_reasons": reasons, "n_expected": n_expected,
            "n_loaded": n_loaded, "n_scored": sum(1 for r in results if r.get("status") == "scored"),
            "infra_errors": infra}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def group_variance(results: list[dict], *, group_key: str = "group", reward_key: str = "reward") -> dict[str, Any]:
    """The GRPO signal of a use case: sample each prompt n times at the training temperature, score
    with the training reward, and ask how many prompt groups have ANY reward variance -- a group
    whose n rewards are all equal contributes a zero advantage, i.e. no task-reward gradient.

    Only fully scored groups count (an infrastructure failure is not a sample). -> fractions of
    all-equal / all-correct (every reward = the max) / all-wrong (every reward 0) / mixed groups,
    the Wilson 95% CI of the mixed ("effective") fraction, and the mean within-group std."""
    groups: dict[Any, list[float]] = {}
    broken: set[Any] = set()
    for r in results:
        g = r[group_key]
        if r.get("status", "scored") != "scored":
            broken.add(g)
            continue
        groups.setdefault(g, []).append(float(r[reward_key]))
    full = {g: v for g, v in groups.items() if g not in broken}
    sizes = sorted({len(v) for v in full.values()})
    n = len(full)
    all_equal = sum(1 for v in full.values() if max(v) == min(v))
    all_zero = sum(1 for v in full.values() if max(v) == 0)
    top = max((max(v) for v in full.values()), default=0.0)
    all_top = sum(1 for v in full.values() if top > 0 and min(v) == top)
    mixed = n - all_equal
    stds = [statistics.pstdev(v) for v in full.values()]
    lo, hi = wilson(mixed, n)
    return {"groups": n, "groups_with_infra_errors": len(broken), "samples_per_group": sizes,
            "all_equal_fraction": all_equal / n if n else None,
            "all_wrong_fraction": all_zero / n if n else None,
            "all_correct_fraction": all_top / n if n else None,
            "effective_fraction": mixed / n if n else None, "effective_fraction_ci95": [lo, hi],
            "mean_within_group_std": sum(stds) / n if n else None}


def model_label(identity: dict[str, Any] | None) -> str | None:
    """What was evaluated, in one string: <run>/global_step_N for a training checkpoint, else the
    served model (Hub id or path) -- never the serving alias, which is the same for every eval."""
    if not identity:
        return None
    if identity.get("step") is not None and identity.get("run_dir"):
        return f"{Path(identity['run_dir']).name}/global_step_{identity['step']}"
    return identity.get("hub_model_id") or identity.get("input") or identity.get("hf_dir")


def header(*, dataset: dict[str, Any], question_ids: list[Any], policy: dict[str, Any],
           started_at: float) -> dict[str, Any]:
    ident = model_identity()
    return {
        "eval_contract": 1,
        "model": model_label(ident),
        "served_model_alias": os.environ.get("EVAL_MODEL", "eval"),
        "model_identity": ident,
        "dataset": dataset,
        "question_ids_sha256": ids_digest(question_ids),
        "eval_policy": policy,
        "code": {"git_sha": os.environ.get("GIT_SHA"), "run_id": os.environ.get("RUN_ID"),
                 # what the job file named, and the tag the image says it was built as
                 "image": os.environ.get("VOA_IMAGE"), "image_tag": os.environ.get("VERL_ON_AIR_IMAGE_TAG")},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def report_and_exit_code(v: dict[str, Any]) -> int:
    if v["valid"]:
        print(f"[eval] VALID: {v['n_scored']}/{v['n_loaded']} questions scored", flush=True)
        return 0
    print("[eval] INVALID -- this artifact must not be compared or reported:", file=sys.stderr, flush=True)
    for r in v["invalid_reasons"]:
        print(f"  - {r}", file=sys.stderr, flush=True)
    return 1
