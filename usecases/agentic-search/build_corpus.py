#!/usr/bin/env python3
"""Build the curated Wikipedia retrieval corpus for the agentic search task -> corpus.parquet.

MVP source = HotpotQA `distractor` `context`: each question ships 10 paragraphs (2 gold + 8
distractor), each a Wikipedia intro paragraph with its title. Taking the UNION of these paragraphs
across train+validation and de-duplicating by title yields exactly the "curated Wikipedia subset
(gold + distractors + neighbor hop)" we chose -- self-contained (no 21M-passage wiki-18 download),
and it is the multi-hop retrieval showcase.

Output: a Parquet file with columns (id, title, text) written to a Volume. A separate step
(create_vs_index.py) loads it into a Unity Catalog Delta table (CDF on) and builds the Vector Search
Delta-Sync index (managed embeddings). Kept `datasets`-only (no Spark) so it runs in the verl image
via air, exactly like the other prep jobs.

NQ fast-follow: NQ answers live in full Wikipedia, so `--datasets nq` requires seeding this corpus
from the wiki-18 subset relevant to the NQ questions (not yet implemented) -- HotpotQA stands alone.

Env / CLI:
    QA_CORPUS_OUT    output parquet path (default Volume data/qa_search/corpus.parquet)
    QA_HF_CACHE      HF datasets cache dir (default /local_disk0/hf_cache)
    QA_CORPUS_SPLITS comma HotpotQA splits to union (default train,validation)
    HF_TOKEN         HuggingFace token (from the air secret)
"""
from __future__ import annotations

import argparse
import hashlib
import os

import datasets


def _passage_id(title: str, text: str) -> str:
    # Hash BOTH title+text so distinct passages that share a title (common across MuSiQue/2Wiki/
    # HotpotQA excerpts) are all kept -- more distractors = harder retrieval. read_article filters
    # by title and returns every passage under it, so multi-passage titles are fine.
    return "p_" + hashlib.md5((title.strip().lower() + "\n" + text.strip()).encode("utf-8")).hexdigest()[:20]


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _hotpotqa_passages(cache: str, splits: list[str]):
    """Yield (title, text) from HotpotQA `distractor` context across the given splits."""
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir=cache, trust_remote_code=True)
    for split in splits:
        if split not in ds:
            continue
        for r in ds[split]:
            ctx = r.get("context") or {}
            titles = ctx.get("title") or []
            sents = ctx.get("sentences") or []
            for i, title in enumerate(titles):
                title = str(title).strip()
                if not title:
                    continue
                paragraph = _norm(" ".join(s for s in (sents[i] if i < len(sents) else []) if str(s).strip()))
                if len(paragraph) >= 20:
                    yield title, paragraph


def _musique_passages(cache: str, splits: list[str]):
    """Yield (title, text) from MuSiQue paragraphs (20 per question: gold + adversarial distractors)."""
    ds = datasets.load_dataset("dgslibisey/MuSiQue", cache_dir=cache)
    smap = {"validation": "validation", "val": "validation", "train": "train"}
    for split in splits:
        hf = smap.get(split, split)
        if hf not in ds:
            continue
        for r in ds[hf]:
            for p in r.get("paragraphs") or []:
                title = str(p.get("title") or "").strip()
                text = _norm(p.get("paragraph_text") or "")
                if title and len(text) >= 20:
                    yield title, text


def _twowiki_passages(cache: str, splits: list[str]):
    """Yield (title, text) from 2WikiMultihopQA context (title + joined sentences)."""
    ds = datasets.load_dataset("xanhho/2WikiMultihopQA", cache_dir=cache, trust_remote_code=True)
    for split in splits:
        if split not in ds:
            continue
        for r in ds[split]:
            for c in r.get("context") or []:
                title = str(c.get("title") or "").strip()
                text = _norm(" ".join(s for s in (c.get("content") or []) if str(s).strip()))
                if title and len(text) >= 20:
                    yield title, text


_PASSAGE_SOURCES = {
    "hotpotqa": _hotpotqa_passages,
    "musique": _musique_passages,
    "2wiki": _twowiki_passages,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=os.environ.get("QA_CORPUS_DATASETS", "hotpotqa"))
    p.add_argument("--out", default=os.environ.get("QA_CORPUS_OUT",
                   "/Volumes/main/mshtelma/verl/data/qa_search/corpus.parquet"))
    p.add_argument("--cache", default=os.environ.get("QA_HF_CACHE", "/local_disk0/hf_cache"))
    p.add_argument("--splits", default=os.environ.get("QA_CORPUS_SPLITS", "train,validation"))
    args = p.parse_args()

    names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "nq" in names:
        raise SystemExit("NQ corpus needs the wiki-18 subset seed (not yet implemented)")

    # Union the requested datasets' contexts, de-duping by content id (title+text) so distinct
    # passages survive but exact cross-dataset duplicates collapse. Bigger union = more distractors.
    dedup: dict[str, dict] = {}
    per_src: dict[str, int] = {}
    for name in names:
        gen = _PASSAGE_SOURCES.get(name)
        if gen is None:
            raise SystemExit(f"unknown corpus dataset {name!r}; choose from {sorted(_PASSAGE_SOURCES)}")
        n0 = len(dedup)
        try:
            # One flaky source (e.g. an HF mirror that changed format) must NOT discard the passages
            # already collected from the others -- warn and continue so the corpus still builds.
            for title, text in gen(args.cache, splits):
                pid = _passage_id(title, text)
                if pid not in dedup:
                    dedup[pid] = {"id": pid, "title": title, "text": text}
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] SKIPPED after error: {type(e).__name__}: {e}")
            continue
        per_src[name] = len(dedup) - n0
        print(f"[{name}] +{per_src[name]} new unique passages (total {len(dedup)})")
    rows = list(dedup.values())
    if not rows:
        raise SystemExit("no passages built")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(out)
    avg = sum(len(r["text"]) for r in rows) / len(rows)
    print(f"wrote {len(rows)} unique passages -> {out}  (avg {avg:.0f} chars)")
    for r in rows[:3]:
        print(f"  {r['id']}  {r['title']!r}: {r['text'][:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
