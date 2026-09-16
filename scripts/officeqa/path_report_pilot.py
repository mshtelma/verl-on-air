#!/usr/bin/env python3
"""Isolated runner for the OfficeQA path-report feasibility test (pilot Section 7).

This wires the pure checks in ``path_report.py`` into an offline scorer with an immutable,
per-run output directory. It is deliberately SMALL and inference/offline-only:

  * ``--self-check``  -- run built-in fixtures end-to-end through the full pipeline with a
                         STUB judge. No network, no model, no GPU, no file writes. Proves
                         the wiring (pilot Section 7 step 1, "Check no network/model call
                         is needed").
  * ``--episodes F --out DIR``  -- score a JSONL of runtime-owned capture records offline.
                         The semantic judge is DISABLED (deterministic-only, verdict
                         UNKNOWN) unless ``--judge-base-url`` is given; the HTTP judge is
                         imported lazily and only then. Writes an immutable run directory.

Explicitly NOT here (needs separate GPU/budget authorization -- see docs/officeqa_rgate_handoff.md):
  * the actor rollout / fresh collection (reuses scripts/eval_officeqa_agentic.py seams in a
    later phase under an opt-in mode); this runner refuses ``--collect`` on purpose.
  * any trainer, reward manager, parameter sync, checkpoint, or R-gate statistics.

These scores are OFFLINE PREVIEWS, never optimizer inputs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import path_report as pr  # noqa: E402


# ---------------------------------------------------------------------------
# deterministic answer-correctness from the OfficeQA answer key (truth-by-design;
# replaces the postponed human answer label). The official unit-aware scorer is
# imported lazily and only when an answer key is provided.
# ---------------------------------------------------------------------------
_SCORE_ANSWER = None


def _get_score_answer():
    global _SCORE_ANSWER
    if _SCORE_ANSWER is None:
        rw = os.path.abspath(os.path.join(_HERE, "..", "reward"))
        if rw not in sys.path:
            sys.path.insert(0, rw)
        from officeqa_reward import score_answer  # stdlib-only (re); the official unit-aware scorer
        _SCORE_ANSWER = score_answer
    return _SCORE_ANSWER


def load_answer_key(path: str) -> dict:
    """uid -> gold answer, from officeqa CSV (uid,answer,...) or a JSONL (uid/episode_id + answer/gt)."""
    key: dict[str, str] = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                uid = str(r.get("uid") or r.get("episode_id") or "").strip()
                ans = str(r.get("answer") or r.get("gt") or "").strip()
                if uid and ans:
                    key[uid] = ans
    else:
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = (r.get("uid") or "").strip()
                ans = (r.get("answer") or "").strip()
                if uid and ans:
                    key[uid] = ans
    return key


def _key_answer(answer_key: dict | None, episode_id: str):
    if not answer_key:
        return None
    return answer_key.get(episode_id) or answer_key.get(str(episode_id))


def _resolve_answer_correct(rec: dict, report, episode_id: str,
                            answer_key: dict | None, tol: float):
    """Explicit record label wins (a reviewer could still set it); otherwise derive it
    deterministically from the answer key. Abstentions/malformed reports stay None."""
    explicit = rec.get("answer_correct", None)
    if explicit is not None:
        return bool(explicit)
    if not report.ok or report.is_abstention or not report.answer:
        return None
    gold = _key_answer(answer_key, episode_id)
    if gold is None:
        return None
    try:
        return bool(_get_score_answer()(str(gold), str(report.answer), tol) > 0)
    except Exception:  # noqa: BLE001 - a scorer fault must not fabricate a label
        return None


# ---------------------------------------------------------------------------
# scoring one captured episode (runtime-owned record -> preview)
# ---------------------------------------------------------------------------
def score_episode(rec: dict, judge_fn: Callable[[dict], str] | None, *,
                  max_retries: int = 2,
                  max_bytes: int = pr.DEFAULT_MAX_REPORT_BYTES,
                  max_steps: int = pr.DEFAULT_MAX_STEPS,
                  max_history_bytes: int = pr.DEFAULT_MAX_HISTORY_BYTES,
                  answer_key: dict | None = None,
                  answer_tol: float = 0.0) -> dict:
    """Score one capture record. The record is the RUNTIME-OWNED history; the terminal
    report text is taken from ``terminal_text`` (or the last of ``assistant_events``).

    Two DETERMINISTIC gates run before the judge (truth-by-design): reference resolution and
    the value-fabrication check. The judge is called only when the report is valid AND passes
    both. Answer-correctness is auto-labeled from ``answer_key`` when present (else None)."""
    episode_id = str(rec.get("episode_id") or "")
    terminal = rec.get("terminal_text")
    if terminal is None:
        terminal = pr.extract_terminal_text(rec.get("assistant_events") or [])

    report = pr.parse_report(terminal, max_bytes=max_bytes, max_steps=max_steps)
    ledger = pr.EpisodeLedger.from_record(rec)
    foreign = frozenset(rec.get("foreign_observation_ids") or [])
    ref = pr.check_references(report, ledger, foreign_ids=foreign)
    values = pr.check_claimed_values(report, ledger)   # deterministic value-fabrication gate

    # answer-correctness: explicit record label wins; else derive from the key (deterministic).
    answer_correct = _resolve_answer_correct(rec, report, episode_id, answer_key, answer_tol)

    # Only bother the judge when the report is valid AND passes BOTH deterministic gates.
    verdict: dict
    if report.ok and ref.valid and values.valid:
        request = pr.build_support_request(
            str(rec.get("question") or ""), str(rec.get("question_requirements") or ""),
            report, ledger, max_history_bytes=max_history_bytes)
        verdict = pr.judge_with_retries(request, judge_fn, max_retries=max_retries)
    else:
        why = ("references unresolved" if (report.ok and not ref.valid)
               else "fabricated value" if (report.ok and not values.valid)
               else "report invalid")
        verdict = {"status": "skipped", "reason": why}

    v_for_preview = verdict if verdict.get("status") == "ok" else (None if verdict.get("status") == "skipped" else verdict)
    preview = pr.candidate_preview(report=report, ref=ref, verdict=v_for_preview,
                                   answer_correct=answer_correct, values=values)

    ac_source = ("record" if rec.get("answer_correct") is not None
                 else "answer_key" if (answer_key is not None and _key_answer(answer_key, episode_id) is not None)
                 else None)
    return {
        "episode_id": episode_id,
        "provenance_sha256": pr.provenance_hash(rec),
        "answer": report.answer,
        "report_kind": report.kind,
        "report_errors": report.errors,
        "report_limit": report.limit,
        # three distinct labels (Section 5.3): kept separate, never conflated
        "answer_correct": answer_correct,                         # separate answer-correctness
        "answer_correct_source": ac_source,                       # record | answer_key | None
        "ref_valid": ref.valid if report.ok else None,            # deterministic path integrity
        "ref_issues": [i.__dict__ for i in ref.issues],
        "value_valid": values.valid if report.ok else None,       # deterministic value faithfulness
        "value_issues": [i.__dict__ for i in values.issues],
        "support_verdict": verdict,                               # semantic support (judge)
        "candidate": preview.to_dict(),
        "labels": preview.labels,
    }


def score_episodes(records: list[dict], judge_fn: Callable[[dict], str] | None,
                   *, concurrency: int = 1, **kw) -> tuple[list[dict], dict]:
    """Score every record and return (results, summary), results in INPUT ORDER.

    ``concurrency <= 1`` scores one episode at a time (the proven path). ``concurrency > 1``
    runs the per-episode ``score_episode`` calls on a thread pool: each episode is independent
    and the judge call is blocking HTTP, so threads give real overlap (88 hard reports drop
    from ~sequential-judge-bound to ~1/concurrency of that). The judge closure builds a fresh
    request per call, so concurrent use is safe. Progress prints to stderr so a long scoring
    run is not silent (the sequential path was, which made a timeout invisible)."""
    n = len(records)
    if concurrency <= 1:
        results = []
        for i, r in enumerate(records):
            results.append(score_episode(r, judge_fn, **kw))
            print(f"[score] {i + 1}/{n} {r.get('episode_id')}", file=sys.stderr, flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * n
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(score_episode, r, judge_fn, **kw): i for i, r in enumerate(records)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                print(f"[score] {done}/{n} (episode {records[futs[fut]].get('episode_id')})",
                      file=sys.stderr, flush=True)
    expected = [str(r.get("episode_id") or "") for r in records]
    integrity = pr.check_result_integrity(results, expected_ids=expected)
    summary = _summarize(results, integrity)
    return results, summary


def _summarize(results: list[dict], integrity: dict) -> dict:
    counts = integrity["counts"]
    kinds: dict[str, int] = {}
    for r in results:
        kinds[r["report_kind"]] = kinds.get(r["report_kind"], 0) + 1
    scored = [r["candidate"]["score"] for r in results if r["candidate"]["status"] == pr.SCORED]
    # deterministic gates (truth-by-design; no model): among VALID reports, how many failed each
    valid = [r for r in results if r["report_kind"] == "valid"]
    ref_fail = sum(1 for r in valid if r.get("ref_valid") is False)
    value_fail = sum(1 for r in valid if r.get("value_valid") is False)
    # answer-correctness (auto-labeled where a key was given): counts kept apart from support
    ac = [r.get("answer_correct") for r in results]
    return {
        "n_episodes": len(results),
        "candidate_status_counts": counts,
        "report_kind_counts": kinds,
        "n_scored": len(scored),
        "score_mean": (sum(scored) / len(scored)) if scored else None,
        # deterministic gate outcomes (no judge involved)
        "n_valid_reports": len(valid),
        "n_ref_invalid": ref_fail,
        "n_value_fabricated": value_fail,
        # answer-correctness label distribution (separate from faithfulness/support)
        "answer_correct_true": sum(1 for x in ac if x is True),
        "answer_correct_false": sum(1 for x in ac if x is False),
        "answer_correct_unlabeled": sum(1 for x in ac if x is None),
        "integrity_ok": integrity["ok"],
        "integrity_problems": integrity["problems"],
    }


# ---------------------------------------------------------------------------
# immutable per-run output (pilot Section 8) -- never overwrite legacy artifacts
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_manifest() -> dict:
    out = {}
    for name in ("path_report.py", "path_report_pilot.py"):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            out[name] = _sha256_file(p)
    return out


def write_run(out_dir: str, manifest: dict, results: list[dict], summary: dict) -> None:
    """Write an immutable run directory; refuse to overwrite a non-empty one."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"refusing to overwrite non-empty run dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# lazily-constructed HTTP judge (only when --judge-base-url is given)
# ---------------------------------------------------------------------------
def make_http_judge(base_url: str, model: str, *, timeout: float | None = None,
                    max_tokens: int | None = None, reasoning_effort: str | None = None,
                    temperature: float | None = None) -> Callable[[dict], str]:
    """Return a judge_fn that POSTs the rendered support prompt to an OpenAI-compatible
    /chat/completions endpoint. Imported lazily so the offline/self-check paths need no
    network stack. NOT exercised by --self-check.

    GLM-5.3 dialect (see docs/officeqa_rl_plan history + the RL memory): thinking is always
    on, so we set a generous ``max_tokens`` and pass ``chat_template_kwargs.reasoning_effort``
    ("high" -- the default "max" overran the budget and 38% never emitted the closing JSON),
    and fall back to ``reasoning_content`` if ``content`` is empty. All knobs read from env
    (JUDGE_TIMEOUT / JUDGE_MAX_TOKENS / JUDGE_REASONING_EFFORT / JUDGE_TEMPERATURE) so the air
    recipe configures them without touching call sites."""
    import urllib.request  # stdlib, lazy

    timeout = float(os.environ.get("JUDGE_TIMEOUT", timeout if timeout is not None else 600.0))
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", max_tokens if max_tokens is not None else 10240))
    temperature = float(os.environ.get("JUDGE_TEMPERATURE", temperature if temperature is not None else 0.0))
    reasoning_effort = (reasoning_effort or os.environ.get("JUDGE_REASONING_EFFORT", "high")).strip()

    def judge_fn(request: dict) -> str:
        system, user = pr.render_support_prompt(request)
        body: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            body["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""

    return judge_fn


# ---------------------------------------------------------------------------
# self-check fixtures (CPU-only; a deterministic stub judge)
# ---------------------------------------------------------------------------
def _self_check_records() -> list[dict]:
    base_obs = [
        {"observation_id": "obs_1", "order": 1, "tool": "read_document",
         "delivered_text": "1941 total 1.47 trillion", "generated_by_request": 0, "delivered_to_request": 1},
        {"observation_id": "obs_2", "order": 2, "tool": "read_document",
         "delivered_text": "1942 total 1.30 trillion", "generated_by_request": 1, "delivered_to_request": 2},
        {"observation_id": "obs_3", "order": 3, "tool": "compute", "args": {"code": "print(1.47-1.30)"},
         "delivered_text": "0.17", "generated_by_request": 2, "delivered_to_request": 3},
    ]
    valid_report = (
        '{"answer":"0.17 trillion dollars","path":['
        '{"id":"a","observation":"obs_1","claim":"1941 total 1.47"},'
        '{"id":"b","observation":"obs_2","claim":"1942 total 1.30"},'
        '{"id":"c","observation":"obs_3","depends_on":["a","b"],"claim":"1.47-1.30=0.17"}]}')
    return [
        {"episode_id": "ok_supported", "question": "decrease?", "answer_correct": True,
         "terminal_text": valid_report, "observations": base_obs},
        {"episode_id": "wrong_value", "question": "decrease?", "answer_correct": True,
         "terminal_text": valid_report, "observations": base_obs},        # judge will reject (fixture)
        {"episode_id": "fabricated_value", "question": "decrease?", "answer_correct": True,
         # leaf claim asserts 9999 (absent from every delivered obs) -> deterministic value gate, NOT the judge
         "terminal_text": '{"answer":"0.17","path":[{"id":"a","observation":"obs_1","claim":"1941 total = 9999 trillion"}]}',
         "observations": base_obs},
        {"episode_id": "fake_ref", "question": "decrease?", "answer_correct": True,
         "terminal_text": '{"answer":"0.17","path":[{"id":"a","observation":"obs_999","claim":"x"}]}',
         "observations": base_obs},
        {"episode_id": "malformed", "question": "decrease?", "answer_correct": True,
         "terminal_text": '{"answer":"0.17","path":[]}', "observations": base_obs},
        {"episode_id": "abstain", "question": "decrease?", "answer_correct": None,
         "terminal_text": '{"answer":"DATA NOT AVAILABLE","path":[]}', "observations": base_obs},
        {"episode_id": "wrong_answer", "question": "decrease?", "answer_correct": False,
         "terminal_text": valid_report, "observations": base_obs},
    ]


def _self_check() -> int:
    def stub_judge(request: dict) -> str:
        # The stub is keyed on report content so we can exercise both branches WITHOUT a model.
        # (Real semantic judging is the GPU phase; this only proves the wiring.)
        answer = str(request.get("answer") or "")
        # a deterministic "reviewer": reject when the tool history contradicts a claim we planted
        return ('{"path_status":"unsupported","path_score":0,"issues":["planted mismatch"]}'
                if "WRONGVALUE" in answer else
                '{"path_status":"supported","path_score":0.85,"issues":[]}')

    records = _self_check_records()
    # planted semantic failure: mutate the "wrong_value" answer so the stub rejects it.
    for r in records:
        if r["episode_id"] == "wrong_value":
            r["terminal_text"] = r["terminal_text"].replace("0.17 trillion dollars", "WRONGVALUE 2.0")
    results, summary = score_episodes(records, stub_judge, max_retries=1)

    expected = {
        "ok_supported": pr.SCORED,
        "wrong_value": pr.ZERO,        # semantic reject by (stub) judge
        "fabricated_value": pr.ZERO,   # DETERMINISTIC value gate (no judge)
        "fake_ref": pr.ZERO,           # deterministic invalid path
        "malformed": pr.ZERO,          # interface failure
        "abstain": pr.ABSTENTION,
        "wrong_answer": pr.ZERO,       # wrong final answer despite supported path
    }
    ok = summary["integrity_ok"]
    print("self-check results:")
    by_id = {}
    for r in results:
        by_id[r["episode_id"]] = r
        got = r["candidate"]["status"]
        want = expected[r["episode_id"]]
        flag = "OK " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  [{flag}] {r['episode_id']:16s} -> {got:10s} (want {want})  {r['candidate']['reason'][:60]}")

    # the fabricated_value case must be caught by the DETERMINISTIC gate, not the judge
    fv = by_id["fabricated_value"]
    det_fv = (fv["value_valid"] is False and fv["support_verdict"].get("status") == "skipped")
    print(f"\n  [{'OK ' if det_fv else 'BAD'}] fabricated_value rejected deterministically "
          f"(value_valid={fv['value_valid']}, judge={fv['support_verdict'].get('status')})")

    # answer-correctness auto-labeled from an answer key (truth-by-design; no judge, no human)
    key = {"ep_key": "507"}
    rep_ok = pr.parse_report('{"answer":"507 million dollars","path":[{"id":"a","observation":"obs_1","claim":"VA 1934 = 507"}]}')
    rep_bad = pr.parse_report('{"answer":"999","path":[{"id":"a","observation":"obs_1","claim":"x"}]}')
    rep_abs = pr.parse_report('{"answer":"DATA NOT AVAILABLE","path":[]}')
    ak = [
        ("answer-key correct -> True", _resolve_answer_correct({"episode_id": "ep_key"}, rep_ok, "ep_key", key, 0.0) is True),
        ("answer-key wrong -> False", _resolve_answer_correct({"episode_id": "ep_key"}, rep_bad, "ep_key", key, 0.0) is False),
        ("abstention -> None", _resolve_answer_correct({"episode_id": "ep_key"}, rep_abs, "ep_key", key, 0.0) is None),
        ("explicit record label wins", _resolve_answer_correct({"episode_id": "ep_key", "answer_correct": False}, rep_ok, "ep_key", key, 0.0) is False),
        ("no key -> None", _resolve_answer_correct({"episode_id": "z"}, rep_ok, "z", None, 0.0) is None),
    ]
    for name, passed in ak:
        ok = ok and passed
        print(f"  [{'OK ' if passed else 'BAD'}] {name}")

    ok = ok and det_fv
    print("\nsummary:", json.dumps(summary, indent=2))
    if not ok:
        print("\nSELF-CHECK FAILED")
        return 1
    print("\nSELF-CHECK PASSED (CPU-only; no network/model/GPU used)")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true", help="run built-in CPU fixtures and exit")
    ap.add_argument("--episodes", help="JSONL of runtime-owned capture records to score offline")
    ap.add_argument("--out", help="immutable output directory for this run")
    ap.add_argument("--judge-base-url", default=None,
                    help="OpenAI-compatible base URL; if omitted the semantic judge is DISABLED")
    ap.add_argument("--judge-model", default="glm-5.3", help="judge model name (only with --judge-base-url)")
    ap.add_argument("--max-retries", type=int, default=2, help="bounded UNKNOWN retries per episode")
    ap.add_argument("--answer-key", default=None,
                    help="officeqa CSV (uid,answer,...) or JSONL; auto-labels answer-correctness deterministically")
    ap.add_argument("--answer-tol", type=float, default=0.0,
                    help="tolerance for the answer-key match (default 0.0 = exact/headline)")
    ap.add_argument("--max-history-bytes", type=int, default=pr.DEFAULT_MAX_HISTORY_BYTES,
                    help="cap on the tool-history bytes handed to the judge; raise it for long "
                         "trajectories so the judge sees the FULL history (it truncates the tail, "
                         "which is where the report usually cites). Must fit the judge context.")
    ap.add_argument("--score-concurrency", type=int, default=1,
                    help="score this many episodes concurrently (judge HTTP is I/O-bound); default "
                         "1 = sequential. Results and order are identical; large batches finish far "
                         "faster (e.g. 88 hard reports).")
    ap.add_argument("--collect", action="store_true", help="(refused) actor rollout is a separate GPU phase")
    args = ap.parse_args(argv)

    if args.collect:
        print("ERROR: --collect (actor rollout / fresh collection) is not part of this CPU runner.\n"
              "It requires separate GPU/budget authorization and reuses scripts/eval_officeqa_agentic.py\n"
              "under a dedicated opt-in mode. See docs/officeqa_rgate_handoff.md Section 3.", file=sys.stderr)
        return 2

    if args.self_check:
        return _self_check()

    if not args.episodes or not args.out:
        ap.error("provide --self-check, or both --episodes and --out")

    with open(args.episodes, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    judge_fn = None
    judge_cfg = "disabled (deterministic-only)"
    if args.judge_base_url:
        judge_fn = make_http_judge(args.judge_base_url, args.judge_model)
        judge_cfg = f"{args.judge_base_url} model={args.judge_model}"

    answer_key = None
    answer_key_cfg = "none (answer-correctness unlabeled unless present in records)"
    if args.answer_key:
        answer_key = load_answer_key(args.answer_key)
        answer_key_cfg = f"{os.path.abspath(args.answer_key)} ({len(answer_key)} uids, tol={args.answer_tol})"

    results, summary = score_episodes(records, judge_fn, max_retries=args.max_retries,
                                      answer_key=answer_key, answer_tol=args.answer_tol,
                                      max_history_bytes=args.max_history_bytes,
                                      concurrency=args.score_concurrency)
    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "episodes_file": os.path.abspath(args.episodes),
        "episodes_sha256": _sha256_file(args.episodes),
        "n_episodes": len(records),
        "judge": judge_cfg,
        "answer_key": answer_key_cfg,
        "answer_tol": args.answer_tol,
        "max_retries": args.max_retries,
        "score_concurrency": args.score_concurrency,
        "limits": {"max_report_bytes": pr.DEFAULT_MAX_REPORT_BYTES, "max_steps": pr.DEFAULT_MAX_STEPS,
                   "max_history_bytes": args.max_history_bytes},
        "code_sha256": _code_manifest(),
        "note": "OFFLINE PREVIEW scores; not optimizer inputs; not an R-gate result. "
                "answer-correctness auto-labeled from the answer key; support from the validated judge.",
    }
    write_run(args.out, manifest, results, summary)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {len(results)} results to {args.out}")
    return 0 if summary["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
