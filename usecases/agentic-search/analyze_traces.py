#!/usr/bin/env python3
"""Offline diagnostic for the agentic-search eval traces -- a HYPOTHESIS GENERATOR, not evidence.

Runs purely on saved trajectory traces (eval.py's EVAL_TRACE_OUT JSONL) -- no GPU, no serving, no
network. For every question it asks whether a gold answer string ever *surfaced* in a tool output,
and splits EM exactly by that:

    EM = P(surfaced) x P(correct | surfaced)  +  P(not surfaced) x P(correct | not surfaced)

Both terms are reported; they sum to EM by construction. "Surfaced" is an answer-string proxy: the
gold string appeared as a contiguous normalized token run in some tool output. It is NOT
supporting-passage recall and does not show that a multi-hop chain was retrieved -- with
--supporting (or --supporting-from-musique) the script also reports the fraction of each
question's supporting paragraph titles that a search result or article read surfaced, and EM given
that the WHOLE chain surfaced. A miss is bucketed as

  NOT_SURFACED     the answer string never appeared in a tool output. Better queries or tool choice
                   (policy) can fix this as well as a better index -- it is not "retrieval-bound".
  SURFACED_MISSED  it appeared, and the model answered something else.
  FORMAT           cover_em=1, em=0: the answer contains the gold but is not exactly it.
  PARTIAL          token F1 >= 0.5 against the gold.

None of this says which component limits EM, nor anything about within-group reward variance in
training: use it to choose the next experiment, then measure that.

Usage:
  analyze_traces.py BASE_traces.jsonl [OTHER_traces.jsonl ...] [--label base --label step20 ...]
                    [--out DIR] [--supporting FILE | --supporting-from-musique]

The first file is the reference: every other file is paired with it question by question (by uid,
and the question text must agree), with the discordant pairs and bucket transitions reported.
--label names the files (default: the file stem; duplicate labels are refused). --supporting FILE
is JSONL of {"question": ..., "supporting_titles": [...]} (or "uid" instead of "question"); trace
records may also carry "supporting_titles" themselves. Writes DIR/trace_diagnostic.json (DIR
defaults to the current directory and is created).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the EXACT normalizer the reward/eval use, so "surfaced" is consistent with EM.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from reward import normalize_answer  # noqa: E402

BUCKETS = ["CORRECT", "FORMAT", "PARTIAL", "SURFACED_MISSED", "NOT_SURFACED"]
# a search result line "[3] Some Title  score=0.812" (tool.py) and a read_article header
_RESULT_TITLE = re.compile(r"^\[\d+\] (.+?)(?:  score=[-0-9.e]+)?$", re.M)
_ARTICLE_TITLE = re.compile(r"^Article: (.+)$", re.M)


class DiagnosticError(SystemExit):
    """A usage or input problem: a clear message and exit code 2."""

    def __init__(self, msg: str):
        super().__init__(2)
        self.msg = msg


def _toks(s: str) -> list[str]:
    return normalize_answer(s).split()


def _contiguous_contains(hay: list[str], needle: list[str]) -> bool:
    """Is `needle` a contiguous sublist of `hay`? (token-level, avoids short-substring FPs)."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return False
    first = needle[0]
    for i in range(n - m + 1):
        if hay[i] == first and hay[i:i + m] == needle:
            return True
    return False


def _tool_outputs(rec: dict) -> list[str]:
    return [tr.get("result", "") for st in rec.get("trajectory", []) or []
            for tr in st.get("tool_results", []) or []
            if isinstance(tr.get("result"), str) and not tr["result"].startswith("Error:")]


def _surfaced_titles(outputs: list[str]) -> set[str]:
    titles = set()
    for out in outputs:
        titles |= {t.strip().lower() for t in _RESULT_TITLE.findall(out)}
        titles |= {t.strip().lower() for t in _ARTICLE_TITLE.findall(out)}
    return titles


def _gold_surfaced(rec: dict, hay_toks: list[str]) -> bool:
    return any(_contiguous_contains(hay_toks, _toks(g)) for g in rec.get("gt", []) or [])


def _bucket(rec: dict, surfaced: bool) -> str:
    em = float(rec.get("em", 0) or 0)
    cover = float(rec.get("cover_em", 0) or 0)
    f1 = float(rec.get("f1", 0) or 0)
    if em >= 1.0:
        return "CORRECT"
    if cover >= 1.0:
        return "FORMAT"          # pred contains gold: over-complete / alias / string
    if f1 >= 0.5:
        return "PARTIAL"         # substantial overlap, softer near-miss
    return "SURFACED_MISSED" if surfaced else "NOT_SURFACED"


def load_traces(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise DiagnosticError(f"no such trace file: {path}")
    recs = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise DiagnosticError(f"{path}:{i}: not a JSON record ({e})") from None
    if not recs:
        raise DiagnosticError(f"{path} holds no trace records -- was EVAL_TRACE_OUT set for that eval?")
    uids = [str(r.get("uid")) for r in recs]
    dup = [u for u, c in Counter(uids).items() if c > 1]
    if dup:
        raise DiagnosticError(f"{path}: duplicate uids (e.g. {dup[0]}) -- not one eval's traces")
    return recs


def load_supporting(path: str) -> dict[str, list[str]]:
    """JSONL {"question"|"uid": ..., "supporting_titles": [...]} -> key -> titles."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            key = r.get("question") or r.get("uid")
            if key is None:
                raise DiagnosticError(f"{path}: a record has neither 'question' nor 'uid'")
            out[str(key).strip()] = list(r.get("supporting_titles") or [])
    return out


def supporting_from_musique(cache_dir: str | None = None) -> dict[str, list[str]]:
    """question -> supporting paragraph titles, from MuSiQue validation at prep_data.py's pinned
    revision (needs `datasets` and the Hub; imported only when asked for)."""
    import prep_data  # noqa: PLC0415 -- heavy (datasets, engine/lib); only for this option
    ds = prep_data.SOURCES["musique"].load(cache_dir=cache_dir)["validation"]
    return {(r["question"] or "").strip(): [p["title"] for p in r["paragraphs"] if p.get("is_supporting")]
            for r in ds}


def _record(rec: dict, supporting: dict[str, list[str]] | None) -> dict:
    outputs = _tool_outputs(rec)
    surfaced = _gold_surfaced(rec, _toks(" ".join(outputs)))
    sup = rec.get("supporting_titles")
    if sup is None and supporting is not None:
        sup = supporting.get(str(rec.get("question", "")).strip(), supporting.get(str(rec.get("uid"))))
    row = {"uid": str(rec.get("uid")), "question": rec.get("question"), "gt": rec.get("gt"),
           "pred": rec.get("pred"), "hop": str(rec.get("hop_type") or "?"),
           "correct": float(rec.get("em", 0) or 0) >= 1.0, "cover": float(rec.get("cover_em", 0) or 0) >= 1.0,
           "surfaced": surfaced, "bucket": _bucket(rec, surfaced),
           "no_answer": rec.get("pred") in (None, "", "None")}
    if sup:
        wanted = {t.strip().lower() for t in sup}
        got = wanted & _surfaced_titles(outputs)
        row["supporting_titles"] = sorted(wanted)
        row["supporting_surfaced"] = sorted(got)
        row["support_recall"] = len(got) / len(wanted)
    return row


def _pct(a: int, b: int) -> float | None:
    return 100.0 * a / b if b else None


def analyze(recs: list[dict], supporting: dict[str, list[str]] | None = None) -> dict:
    rows = [_record(r, supporting) for r in recs]
    n = len(rows)
    n_c = sum(r["correct"] for r in rows)
    s = [r for r in rows if r["surfaced"]]
    ns = [r for r in rows if not r["surfaced"]]
    # the two terms of EM = P(S) P(C|S) + P(~S) P(C|~S), each as a share of all questions
    term_s = sum(r["correct"] for r in s) / n
    term_ns = sum(r["correct"] for r in ns) / n
    assert abs(term_s + term_ns - n_c / n) < 1e-12
    by_hop: dict[str, dict] = {}
    for hop in sorted({r["hop"] for r in rows}):
        h = [r for r in rows if r["hop"] == hop]
        by_hop[hop] = {"n": len(h), "em_pct": _pct(sum(r["correct"] for r in h), len(h)),
                       "surfaced_pct": _pct(sum(r["surfaced"] for r in h), len(h))}
    out = {
        "n": n, "em_pct": 100.0 * n_c / n, "cover_pct": _pct(sum(r["cover"] for r in rows), n),
        "no_answer": sum(r["no_answer"] for r in rows),
        "decomposition": {
            "p_surfaced": len(s) / n, "p_correct_given_surfaced": (sum(r["correct"] for r in s) / len(s)) if s else None,
            "p_not_surfaced": len(ns) / n,
            "p_correct_given_not_surfaced": (sum(r["correct"] for r in ns) / len(ns)) if ns else None,
            "term_surfaced": term_s, "term_not_surfaced": term_ns,
        },
        "buckets": {b: sum(r["bucket"] == b for r in rows) for b in BUCKETS},
        "by_hop": by_hop,
        "examples": {b: [{k: r[k] for k in ("uid", "question", "gt", "pred")}
                         for r in rows if r["bucket"] == b][:3] for b in BUCKETS if b != "CORRECT"},
    }
    with_sup = [r for r in rows if "support_recall" in r]
    if with_sup:
        full = [r for r in with_sup if r["support_recall"] >= 1.0]
        out["supporting"] = {
            "n_with_titles": len(with_sup),
            "mean_support_recall": sum(r["support_recall"] for r in with_sup) / len(with_sup),
            "p_all_supporting_surfaced": len(full) / len(with_sup),
            "em_pct_given_all_supporting": _pct(sum(r["correct"] for r in full), len(full)),
        }
    out["_rows"] = rows
    return out


def pair(ref: dict, other: dict) -> dict:
    """The reference vs another trace set, question by question."""
    a = {r["uid"]: r for r in ref["_rows"]}
    b = {r["uid"]: r for r in other["_rows"]}
    common = sorted(set(a) & set(b), key=lambda u: (len(u), u))
    clash = [u for u in common if (a[u]["question"] or "").strip() != (b[u]["question"] or "").strip()]
    if clash:
        raise DiagnosticError(f"uid {clash[0]} names different questions in the two files -- "
                              "they are not evaluations of the same question set")
    only_ref = [u for u in common if a[u]["correct"] and not b[u]["correct"]]
    only_other = [u for u in common if b[u]["correct"] and not a[u]["correct"]]
    transitions = Counter(f"{a[u]['bucket']}->{b[u]['bucket']}" for u in common if a[u]["bucket"] != b[u]["bucket"])
    return {"n_paired": len(common), "unpaired_ref": len(set(a) - set(b)), "unpaired_other": len(set(b) - set(a)),
            "both_correct": sum(a[u]["correct"] and b[u]["correct"] for u in common),
            "only_ref_correct": len(only_ref), "only_other_correct": len(only_other),
            "surfaced_ref_only": sum(a[u]["surfaced"] and not b[u]["surfaced"] for u in common),
            "surfaced_other_only": sum(b[u]["surfaced"] and not a[u]["surfaced"] for u in common),
            "bucket_transitions": dict(transitions.most_common()),
            "examples_only_ref": only_ref[:5], "examples_only_other": only_other[:5]}


def _labels(paths: list[str], labels: list[str] | None) -> list[str]:
    if labels:
        if len(labels) != len(paths):
            raise DiagnosticError(f"{len(labels)} --label(s) for {len(paths)} trace file(s)")
        out = labels
    else:
        out = [Path(p).name.removesuffix(".jsonl").removesuffix("_traces") for p in paths]
    dup = [x for x, c in Counter(out).items() if c > 1]
    if dup:
        raise DiagnosticError(f"label {dup[0]!r} names two files; pass --label for each")
    return out


def _f(x: float | None, w: int = 6) -> str:
    return f"{x:{w}.1f}" if x is not None else f"{'-':>{w}}"


def report(res: dict[str, dict], pairs: dict[str, dict]) -> None:
    order = list(res)
    print("=" * 96)
    print("EM DECOMPOSITION  EM = P(S)*P(C|S) + P(~S)*P(C|~S),  S = a gold answer string surfaced in a tool output")
    print("(a proxy, not supporting-passage recall -- a hypothesis generator, not a causal account)")
    print("=" * 96)
    print(f"{'run':>14} | {'n':>4} {'EM%':>6} | {'P(S)%':>6} {'P(C|S)%':>8} {'term S':>7} | "
          f"{'P(C|~S)%':>8} {'term ~S':>7} | {'no_ans':>6}")
    print("-" * 96)
    for k in order:
        r, d = res[k], res[k]["decomposition"]
        pcs = 100 * d["p_correct_given_surfaced"] if d["p_correct_given_surfaced"] is not None else None
        pcn = 100 * d["p_correct_given_not_surfaced"] if d["p_correct_given_not_surfaced"] is not None else None
        print(f"{k:>14} | {r['n']:4d} {r['em_pct']:6.1f} | {100 * d['p_surfaced']:6.1f} {_f(pcs, 8)} "
              f"{100 * d['term_surfaced']:7.1f} | {_f(pcn, 8)} {100 * d['term_not_surfaced']:7.1f} | {r['no_answer']:6d}")

    print(f"\nBUCKETS (counts)\n{'run':>14} | " + " ".join(f"{c:>15}" for c in BUCKETS))
    for k in order:
        print(f"{k:>14} | " + " ".join(f"{res[k]['buckets'][c]:>15}" for c in BUCKETS))
    print("NOT_SURFACED can be a query/tool-choice (policy) miss as much as an index miss.")

    for k in order:
        sup = res[k].get("supporting")
        if sup:
            print(f"\n--- {k}: supporting paragraphs ({sup['n_with_titles']} questions with titles) --- "
                  f"mean title recall {100 * sup['mean_support_recall']:.1f}%, whole chain surfaced "
                  f"{100 * sup['p_all_supporting_surfaced']:.1f}%, EM given whole chain "
                  f"{_f(sup['em_pct_given_all_supporting'], 0)}%")
        print(f"--- {k}: by hop (n / EM% / surfaced%) ---")
        for hop, h in res[k]["by_hop"].items():
            print(f"  {hop:>6}: n={h['n']:4d}  EM={_f(h['em_pct'], 5)}%  surfaced={_f(h['surfaced_pct'], 5)}%")

    for k, p in pairs.items():
        print(f"\n--- paired {order[0]} vs {k}: {p['n_paired']} questions "
              f"(unpaired: {p['unpaired_ref']} / {p['unpaired_other']}) ---")
        print(f"  both correct {p['both_correct']}, only {order[0]} {p['only_ref_correct']}, only {k} "
              f"{p['only_other_correct']};  surfaced only in {order[0]} {p['surfaced_ref_only']}, only in {k} "
              f"{p['surfaced_other_only']}")
        for t, c in list(p["bucket_transitions"].items())[:8]:
            print(f"    {t:<36} {c}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="eval trace JSONL files; the first is the reference")
    ap.add_argument("--label", action="append", help="a name per trace file, in order")
    ap.add_argument("--out", default=".", help="directory for trace_diagnostic.json (default: cwd)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--supporting", help='JSONL {"question"|"uid", "supporting_titles": [...]}')
    g.add_argument("--supporting-from-musique", action="store_true",
                   help="supporting titles from MuSiQue validation at the pinned revision (needs the Hub)")
    ap.add_argument("--cache", default=None, help="HF cache dir for --supporting-from-musique")
    args = ap.parse_args(argv)
    try:
        if not args.traces:
            raise DiagnosticError("give at least one trace file (eval.py's EVAL_TRACE_OUT)")
        labels = _labels(args.traces, args.label)
        supporting = (load_supporting(args.supporting) if args.supporting
                      else supporting_from_musique(args.cache) if args.supporting_from_musique else None)
        res = {lab: analyze(load_traces(p), supporting) for lab, p in zip(labels, args.traces)}
        pairs = {lab: pair(res[labels[0]], res[lab]) for lab in labels[1:]}
    except DiagnosticError as e:
        print(f"analyze_traces: {e.msg}", file=sys.stderr)
        return 2
    report(res, pairs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "trace_diagnostic.json"
    doc = {"note": "hypothesis generator: 'surfaced' is an answer-string proxy, not supporting-passage recall",
           "files": dict(zip(labels, args.traces)), "reference": labels[0],
           "runs": {k: {kk: vv for kk, vv in v.items() if kk != "_rows"} for k, v in res.items()},
           "paired": pairs}
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    os.replace(tmp, dest)
    print(f"\n[wrote {dest}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
