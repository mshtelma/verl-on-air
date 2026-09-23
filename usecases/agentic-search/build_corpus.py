#!/usr/bin/env python3
"""Build the passage corpus the agentic-search agent retrieves from -> corpus parquet + manifest.

The corpus is the UNION of the source datasets' own per-question contexts -- every gold and
distractor paragraph MuSiQue and HotpotQA ship with their questions -- across the listed splits,
de-duplicated by content. The shipped job uses MuSiQue + HotpotQA over train + validation, read at
the commits pinned in prep_data.SOURCES (the same ones the questions come from).

This is a TRANSDUCTIVE, curated retrieval setting, not open-domain QA: the passages that answer
the evaluation questions are in the corpus by construction (their validation split is included),
alongside the datasets' adversarial distractors and every other question's passages. That is not
label leakage -- passages carry no answers -- but it is the setting every published number was
measured in, and it must stay fixed between a baseline and a trained checkpoint.

A requested source that fails to load FAILS the build: without MuSiQue there are still enough
HotpotQA passages to look fine, and the task has silently changed. `--allow-partial` writes the
corpus anyway, WITHOUT any passage of a failed source, and marks its manifest `complete: false`;
create_vs_index.py refuses such a corpus unless told otherwise.

Outputs, side by side:
    <out>                     parquet (id, title, text); id = md5(title + text), stable across rebuilds
    <stem>.manifest.json      sources + revisions; per source and split: questions, paragraphs,
                              unique/new passages, supporting-passage coverage; the passage count
                              and content_sha256 over (id, title, text) -- which names the index

Kept `datasets`-only (no Spark) so it runs in the verl image via air, like the other prep jobs.

Env / CLI:
    QA_CORPUS_DATASETS comma list of sources (default musique,hotpotqa)
    QA_CORPUS_OUT      output parquet path (default Volume data/qa_musique/corpus_big.parquet)
    QA_CORPUS_SPLITS   comma splits to union (default train,validation)
    QA_HF_CACHE        HF datasets cache dir (default /local_disk0/hf_cache)
    HF_TOKEN           optional HuggingFace token (the data is public)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402
from prep_data import SOURCES  # noqa: E402

MIN_CHARS = 20   # shorter "paragraphs" are empty/junk context rows


def _passage_id(title: str, text: str) -> str:
    # Hash BOTH title+text so distinct passages that share a title (common across MuSiQue/HotpotQA
    # excerpts) are all kept -- more distractors = harder retrieval. read_article filters by title
    # and returns every passage under it, so multi-passage titles are fine.
    return "p_" + hashlib.md5((title.strip().lower() + "\n" + text.strip()).encode("utf-8")).hexdigest()[:20]


def _norm(text: str) -> str:
    return " ".join(str(text).split())


# Each source yields, per question: (question id, [(title, text, is_supporting), ...]).
Context = Iterator[tuple[str, list[tuple[str, str, bool]]]]


def _musique_contexts(rows) -> Context:
    """MuSiQue: 20 paragraphs per question (gold + adversarial distractors), flagged is_supporting."""
    for r in rows:
        yield str(r["id"]), [(p.get("title"), p.get("paragraph_text"), bool(p.get("is_supporting")))
                             for p in r.get("paragraphs") or []]


def _hotpotqa_contexts(rows) -> Context:
    """HotpotQA distractor: 10 paragraphs (2 gold + 8 distractors); gold titles in supporting_facts."""
    for r in rows:
        ctx = r.get("context") or {}
        titles, sents = ctx.get("title") or [], ctx.get("sentences") or []
        gold = set((r.get("supporting_facts") or {}).get("title") or [])
        yield str(r["id"]), [(t, " ".join(s for s in (sents[i] if i < len(sents) else []) if str(s).strip()),
                              t in gold) for i, t in enumerate(titles)]


_CONTEXTS = {"musique": _musique_contexts, "hotpotqa": _hotpotqa_contexts}


def _collect(name: str, ds, splits: list[str]) -> tuple[dict[str, dict], dict]:
    """One source's unique passages (first-seen order) and its per-split audit. Raises on any
    problem, so a source contributes all of its passages or none."""
    passages: dict[str, dict] = {}
    audit: dict[str, dict] = {}
    for split in splits:
        if split not in ds:
            raise dm.SourceError(f"{name} has no split {split!r} (it has {sorted(ds)})")
        a = dict(questions=0, paragraphs=0, supporting=0, supporting_kept=0, questions_fully_supported=0)
        for _qid, paras in _CONTEXTS[name](ds[split]):
            a["questions"] += 1
            whole = True
            for title, text, supporting in paras:
                title, text = str(title or "").strip(), _norm(text or "")
                a["paragraphs"] += 1
                a["supporting"] += supporting
                if title and len(text) >= MIN_CHARS:
                    pid = _passage_id(title, text)
                    passages.setdefault(pid, {"id": pid, "title": title, "text": text})
                    a["supporting_kept"] += supporting
                elif supporting:
                    whole = False    # a gold paragraph too short to index: that question lost evidence
            a["questions_fully_supported"] += whole
        audit[split] = a
    return passages, audit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=os.environ.get("QA_CORPUS_DATASETS", "musique,hotpotqa"))
    p.add_argument("--out", default=os.environ.get("QA_CORPUS_OUT",
                   "/Volumes/main/mshtelma/verl/data/qa_musique/corpus_big.parquet"))
    p.add_argument("--cache", default=os.environ.get("QA_HF_CACHE", "/local_disk0/hf_cache"))
    p.add_argument("--splits", default=os.environ.get("QA_CORPUS_SPLITS", "train,validation"))
    p.add_argument("--allow-partial", action="store_true",
                   help="write the corpus even if a source fails (without it); the manifest says so")
    args = p.parse_args(argv)

    names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [n for n in names if n not in _CONTEXTS]
    if unknown or not names or not splits:
        raise SystemExit(f"unknown corpus source(s) {unknown}; choose from {sorted(_CONTEXTS)} "
                         f"(and at least one split)")

    # Union the sources, de-duping by content id (title+text): distinct passages survive, exact
    # cross-dataset duplicates collapse. Bigger union = more distractors.
    dedup: dict[str, dict] = {}
    built, failed = [], []
    for name in names:
        src = SOURCES[name]
        try:
            passages, audit = _collect(name, src.load(cache_dir=args.cache), splits)
        except Exception as e:  # noqa: BLE001 - recorded, and fatal unless --allow-partial
            failed.append(src.record(name=name, error=f"{type(e).__name__}: {e}"))
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        n0 = len(dedup)
        for pid, row in passages.items():
            dedup.setdefault(pid, row)
        built.append(src.record(name=name, splits=audit, unique_passages=len(passages),
                                new_passages=len(dedup) - n0))
        cover = {s: f"{a['supporting_kept']}/{a['supporting']}" for s, a in audit.items()}
        print(f"[{name}] +{len(dedup) - n0} new unique passages (total {len(dedup)}); "
              f"supporting passages kept {cover}", flush=True)
    if failed and not args.allow_partial:
        raise SystemExit(f"{len(failed)} corpus source(s) failed: {[f['name'] for f in failed]} -- the "
                         f"corpus would silently be a different task. Fix the source, or pass "
                         f"--allow-partial to build a corpus marked incomplete.")
    rows = list(dedup.values())
    if not rows:
        raise SystemExit("no passages built")

    out = Path(os.path.expanduser(args.out))
    manifest = out.with_name(f"{out.stem}.manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)   # never beside a corpus it does not describe
    dm.write_parquet(out, rows)
    dm.write_manifest(
        manifest, tool="usecases/agentic-search/build_corpus.py", sources=built, failed_sources=failed,
        complete=not failed, splits=splits, min_chars=MIN_CHARS,
        setting="transductive: the union of the sources' own gold + distractor paragraphs for the "
                "listed splits, including the evaluation split",
        outputs=[dm.output_record(out, len(rows),
                                  content_sha256=dm.content_digest(rows, ["id", "title", "text"]))])
    avg = sum(len(r["text"]) for r in rows) / len(rows)
    print(f"wrote {len(rows)} unique passages -> {out}  (avg {avg:.0f} chars)"
          + ("" if not failed else f"  INCOMPLETE: without {[f['name'] for f in failed]}"))
    for r in rows[:3]:
        print(f"  {r['id']}  {r['title']!r}: {r['text'][:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
