#!/usr/bin/env python3
"""Validate the OfficeQA support judge (GLM-5.3 TP16) against truth-by-construction fixtures.

Runs each investigator fixture (see judge_validation_fixtures.py) through the EXACT production
judge path -- `path_report.build_support_request` -> the served judge -> `parse_support_verdict`
-- and compares the judge's `path_status` to the construction label. The two dangerous errors
are called out explicitly:
  * FALSE ACCEPT  = a must-reject path (wrong row / period / value / missing coverage / code
                    mismatch) the judge marked `supported` -> would poison training.
  * FALSE REJECT  = a genuinely-supported path (correct lookup/compute, valid alternative,
                    published aggregate) the judge marked `unsupported` -> would kill signal.
`unknown` (verifier fault) is reported separately, never silently counted as a rejection.

Modes:
  --self-check                 : oracle stub judge (CPU only) -> proves the harness wiring.
  --judge-base-url URL [--out DIR] : run the REAL served judge and write an immutable report.

The judge is shown ONLY question/answer/report/tool-history -- never the label, family or
answer-correctness. These fixtures validate the VERIFIER; they are not actor episodes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import path_report as pr                          # noqa: E402
import path_report_pilot as prp                   # noqa: E402  (make_http_judge)
import judge_validation_fixtures as jvf            # noqa: E402

SUPPORTED, UNSUPPORTED, UNKNOWN = pr.SUPPORTED, pr.UNSUPPORTED, "unknown"


def evaluate_fixture(fx: dict, judge_fn: Callable[[dict], str] | None, *, max_retries: int = 2) -> dict:
    """Run one fixture and classify the judge's decision vs its construction label."""
    report = pr.parse_report(fx["terminal_text"])
    ledger = pr.EpisodeLedger.from_record(fx)
    ref = pr.check_references(report, ledger)
    values = pr.check_claimed_values(report, ledger)   # deterministic value-fabrication gate
    expected = fx["expected_path_status"]
    out = {"episode_id": fx["episode_id"], "family": fx["family"], "expected": expected,
           "rationale": fx.get("rationale", ""), "deterministic": bool(fx.get("deterministic"))}

    if fx.get("deterministic"):
        # must be rejected BEFORE the judge -- by EITHER deterministic gate (references or values)
        det_reject = (not ref.valid) or (not values.valid)
        out["got"] = ("invalid_ref" if not ref.valid
                      else "fabricated_value" if not values.valid else "RESOLVED")
        out["ref_codes"] = ref.codes
        out["value_codes"] = values.codes
        out["det_ok"] = det_reject
        out["match"] = det_reject
        out["false_accept"] = not det_reject           # a deterministic defect that slips through
        out["false_reject"] = False
        out["judge_unknown"] = False
        return out

    if not report.ok:
        out["got"] = "report_malformed"
        out.update(match=False, false_accept=False, false_reject=(expected == SUPPORTED), judge_unknown=False,
                   note="; ".join(report.errors[:3]))
        return out
    if not ref.valid:
        # a semantic fixture should have resolvable refs; if not, that's a fixture/authoring bug
        out["got"] = "unexpected_invalid_ref"
        out.update(match=False, false_accept=False, false_reject=False, judge_unknown=False, ref_codes=ref.codes)
        return out
    if not values.valid:
        # a semantic fixture whose value is absent from the trace should have been authored as
        # `deterministic` (value gate). Flag the authoring mismatch rather than silently pass it.
        out["got"] = "unexpected_fabricated_value"
        out.update(match=False, false_accept=False, false_reject=False, judge_unknown=False,
                   value_codes=values.codes)
        return out

    request = pr.build_support_request(fx.get("question", ""), fx.get("question_requirements", ""), report, ledger)
    verdict = pr.judge_with_retries(request, judge_fn, max_retries=max_retries)
    if verdict.get("status") == "ok":
        got = verdict["path_status"]
        out.update(got=got, path_score=verdict.get("path_score"), issues=verdict.get("issues", []),
                   attempts=verdict.get("attempts"))
    else:
        got = UNKNOWN
        out.update(got=UNKNOWN, verdict_kind=verdict.get("kind"), verdict_reason=verdict.get("reason"),
                   attempts=verdict.get("attempts"))
    out["match"] = (got == expected)
    out["false_accept"] = (expected == UNSUPPORTED and got == SUPPORTED)
    out["false_reject"] = (expected == SUPPORTED and got == UNSUPPORTED)
    out["judge_unknown"] = (got == UNKNOWN)
    return out


def run(fixtures: list[dict], judge_fn: Callable[[dict], str] | None, *, max_retries: int = 2):
    results = [evaluate_fixture(fx, judge_fn, max_retries=max_retries) for fx in fixtures]
    n_fa = sum(r["false_accept"] for r in results)
    n_fr = sum(r["false_reject"] for r in results)
    n_unk = sum(r["judge_unknown"] for r in results)
    n_match = sum(r["match"] for r in results)
    det = [r for r in results if r["deterministic"]]
    det_ok = all(r["det_ok"] for r in det) if det else True
    # confusion over semantic cases
    confusion: dict = {}
    for r in results:
        if r["deterministic"]:
            continue
        key = f"{r['expected']}->{r['got']}"
        confusion[key] = confusion.get(key, 0) + 1
    summary = {
        "n_fixtures": len(results), "n_match": n_match,
        "false_accepts": n_fa, "false_rejects": n_fr, "unknowns": n_unk,
        "deterministic_ok": det_ok,
        "confusion": confusion,
        # a useful judge must discriminate AND resolve: any false accept/reject, unresolved
        # deterministic leak, or UNKNOWN on a by-construction-decidable fixture fails the smoke.
        "pass": (n_fa == 0 and n_fr == 0 and det_ok and n_unk == 0),
        "attention": (n_unk > 0),
    }
    return results, summary


def _print_report(results: list[dict], summary: dict) -> None:
    print("case                 expected     got          verdict")
    print("-" * 78)
    for r in results:
        flag = "OK " if r["match"] else ("!! " if (r["false_accept"] or r.get("got") == "RESOLVED") else "xx ")
        extra = ""
        if r.get("path_score") is not None:
            extra = f"score={r['path_score']}"
        elif r["got"] == UNKNOWN:
            extra = f"kind={r.get('verdict_kind')}"
        elif r["deterministic"]:
            extra = f"codes={r.get('ref_codes')}"
        print(f"[{flag}] {r['episode_id']:18s} {r['expected']:12s} {str(r['got']):14s} {extra}")
    print("-" * 78)
    print(f"confusion (semantic): {summary['confusion']}")
    print(f"false_accepts={summary['false_accepts']}  false_rejects={summary['false_rejects']}  "
          f"unknowns={summary['unknowns']}  deterministic_ok={summary['deterministic_ok']}")
    print(f"\nJUDGE VALIDATION: {'PASS' if summary['pass'] else 'FAIL'}")
    if not summary["pass"]:
        for r in results:
            if r["false_accept"]:
                print(f"  FALSE ACCEPT  {r['episode_id']} ({r['family']}): judge said supported for a must-reject path")
            if r["false_reject"]:
                print(f"  FALSE REJECT  {r['episode_id']} ({r['family']}): judge said unsupported for a valid path")
            if r["judge_unknown"]:
                print(f"  UNKNOWN       {r['episode_id']} ({r['family']}): judge returned no usable verdict "
                      f"(kind={r.get('verdict_kind')})")
            if r["deterministic"] and not r["det_ok"]:
                print(f"  LEAK          {r['episode_id']}: a fake reference RESOLVED (deterministic layer failed)")


def _oracle_judge(fixtures: list[dict]) -> Callable[[dict], str]:
    """A CPU stub that returns each fixture's construction label, keyed on (question, answer,
    report) -- exactly the inputs the real judge sees -- so two fixtures that share a report but
    pose different questions (e.g. a correct lookup vs. the same lookup used to answer a different
    question) are distinguished, as the real judge would. Deterministic fixtures never reach it."""
    def _key(req: dict) -> str:
        return json.dumps([req.get("question", ""), req.get("answer", ""), req.get("report", [])],
                          sort_keys=True)

    key_to_status: dict[str, str] = {}
    for fx in fixtures:
        if fx.get("deterministic"):
            continue
        report = pr.parse_report(fx["terminal_text"])
        ledger = pr.EpisodeLedger.from_record(fx)
        req = pr.build_support_request(fx.get("question", ""), fx.get("question_requirements", ""), report, ledger)
        key_to_status[_key(req)] = fx["expected_path_status"]

    def judge_fn(request: dict) -> str:
        st = key_to_status.get(_key(request), UNKNOWN)
        if st == SUPPORTED:
            return '{"path_status":"supported","path_score":0.9,"issues":[]}'
        if st == UNSUPPORTED:
            return '{"path_status":"unsupported","path_score":0,"issues":["planted defect"]}'
        return '{"path_status":"unknown","path_score":null}'
    return judge_fn


def _self_check() -> int:
    fixtures = jvf.all_fixtures()
    results, summary = run(fixtures, _oracle_judge(fixtures), max_retries=1)
    _print_report(results, summary)
    ok = summary["pass"] and not summary["attention"]
    print("\nSELF-CHECK " + ("PASSED (CPU-only; oracle judge)" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--fixtures", help="JSONL of fixtures (default: built-in judge_validation_fixtures)")
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"),
                    help="OpenAI-compatible base URL of the served judge")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "judge"))
    ap.add_argument("--max-retries", type=int, default=int(os.environ.get("OQ_JV_MAX_RETRIES", "2")))
    ap.add_argument("--out", default=os.environ.get("OQ_JV_OUT"), help="immutable output directory")
    args = ap.parse_args(argv)

    if args.self_check:
        return _self_check()

    fixtures = jvf.all_fixtures() if not args.fixtures else \
        [json.loads(l) for l in open(args.fixtures, encoding="utf-8") if l.strip()]

    if not args.judge_base_url:
        ap.error("provide --self-check, or --judge-base-url (and optionally --out) for a real run")

    judge_fn = prp.make_http_judge(args.judge_base_url, args.judge_model)
    results, summary = run(fixtures, judge_fn, max_retries=args.max_retries)
    _print_report(results, summary)

    if args.out:
        if os.path.exists(args.out) and os.listdir(args.out):
            raise SystemExit(f"refusing to overwrite non-empty run dir: {args.out}")
        os.makedirs(args.out, exist_ok=True)
        manifest = {
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "judge": f"{args.judge_base_url} model={args.judge_model}",
            "max_retries": args.max_retries, "n_fixtures": len(fixtures),
            "fixtures_source": args.fixtures or "judge_validation_fixtures.py (built-in)",
            "note": "investigator truth-by-construction fixtures for VERIFIER validation; not actor episodes.",
        }
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nwrote judge-validation report -> {args.out}")

    # exit non-zero on any discrimination failure so the job status reflects it
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
