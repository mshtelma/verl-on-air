#!/usr/bin/env python3
"""Prepare the agentic search/RAG tool-agent dataset for verl (Search-R1 style) -> train/test parquet.

Emits verl's standard tool-agent parquet schema, so the fully-async
ToolAgentLoop + rule-based EM reward (usecases/agentic-search/reward.py) consume it unchanged:

    data_source   "musique" | "hotpotqa"
    agent_name    "tool_agent"
    prompt        [{role: system, content: SYSTEM_PROMPT}, {role: user, content: "Question: ..."}]
    ability       "qa_search"
    reward_model  {"style": "rule", "ground_truth": [gold, ...]}   # golden answers list
    extra_info    {split, index, source_id, data_source, question, answers, hop_type, level}

The model is told to use vector_search / keyword_search / read_article and commit its final answer in
<answer> ... </answer> (the only thing the reward reads). Ground truth lives ONLY in reward_model,
never in the prompt. `source_id` is the dataset's own question id, stable across rebuilds.

The questions are MuSiQue-Ans (the shipped job); HotpotQA is also available. Both are read at the
pinned commits in SOURCES -- the same ones build_corpus.py builds the passage corpus from -- and a
DATA_MANIFEST.json beside the outputs records those revisions, the row counts before and after
each filter, the sampling, and each output's sha256 (engine/lib/data_manifest.py).

Env / CLI:
    QA_DATASETS         comma list: musique[,hotpotqa]   (default musique)
    QA_TOOL_OUT_DIR     output dir (default Volume data/qa_musique)
    QA_PREP_TRAIN_LIMIT cap train rows (default 0 = all)
    QA_PREP_VAL_LIMIT   cap val rows   (default 500)
    QA_HF_CACHE         HF datasets cache dir (default /local_disk0/hf_cache)
    HF_TOKEN            optional HuggingFace token (the data is public)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

# Question AND passage sources (build_corpus.py reads the same pins), each at the commit its main
# branch has pointed to since before the first published run (last changes: 2023-06, 2025-08).
SOURCES = {
    "musique": dm.Source("dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"),
    "hotpotqa": dm.Source("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab", "distractor"),
}

# Shared with the eval harness (eval.py imports SYSTEM_PROMPT from here) so train ==
# eval: identical tool contract + identical <answer> commitment.
SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching an encyclopedic corpus with "
    "tools. You have three tools:\n"
    "  - vector_search(query, top_k): semantic search; returns passages, each with its article title.\n"
    "  - keyword_search(query, top_k): hybrid keyword+semantic search; use it when exact names, "
    "titles, numbers, or rare terms must match.\n"
    "  - read_article(title): read the full text of one article by its EXACT title (as shown in a "
    "search result).\n\n"
    "Work step by step. Search for what you need; for a multi-hop question, find the first entity, "
    "read its article to discover the next entity, then search/read again. Ground every claim in "
    "retrieved passages -- do NOT answer from memory alone. When you have enough evidence, give your "
    "final answer and NOTHING else, wrapped in <answer> and </answer> tags. The answer must be a "
    "short span -- a name, entity, number, date, or yes/no -- with no extra words. "
    "For example: <answer> Beijing </answer>."
)

USER_TEMPLATE = "Question: {question}"


def _to_verl(*, data_source: str, source_id: str, question: str, golden_answers: list[str], split: str,
             index: int, hop_type: str = "", level: str = "") -> dict:
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question)},
        ],
        "ability": "qa_search",
        "reward_model": {"style": "rule", "ground_truth": golden_answers},
        "extra_info": {
            "split": split,
            "index": index,
            "source_id": source_id,
            "data_source": data_source,
            "question": question,
            "answers": golden_answers,
            "hop_type": hop_type,
            "level": level,
        },
    }


def _clean_answers(answers) -> list[str]:
    if isinstance(answers, str):
        answers = [answers]
    seen, out = set(), []
    for a in answers or []:
        a = str(a).strip()
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out


def _hotpotqa_row(r: dict) -> dict | str:
    q, ga = (r.get("question") or "").strip(), _clean_answers(r.get("answer"))
    if not q or not ga:
        return "no_question_or_answer"
    return dict(question=q, golden_answers=ga, hop_type=r.get("type", ""), level=r.get("level", ""))


def _musique_row(r: dict) -> dict | str:
    # MuSiQue-Ans: harder 2-4 hop questions, built so single-hop retrieval shortcuts fail.
    # golden = answer + answer_aliases; keep only answerable items. hop_type = N-hop from the
    # decomposition length (useful for a difficulty breakdown in eval).
    if r.get("answerable") is False:
        return "unanswerable"
    q = (r.get("question") or "").strip()
    ga = _clean_answers([r.get("answer")] + list(r.get("answer_aliases") or []))
    if not q or not ga:
        return "no_question_or_answer"
    hops = len(r.get("question_decomposition") or []) or 0
    return dict(question=q, golden_answers=ga, hop_type=(f"{hops}hop" if hops else ""))


_ROWS = {"musique": _musique_row, "hotpotqa": _hotpotqa_row}


def _convert(name: str, ds, split_name: str, hf_split: str, limit: int) -> tuple[list[dict], dict]:
    """The first `limit` rows (0 = all) of `hf_split`, in source order, as verl rows -> (rows, stats)."""
    rows = ds[hf_split]
    selected = rows.select(range(min(limit, len(rows)))) if limit and limit > 0 else rows
    out, dropped = [], Counter()
    for i, r in enumerate(selected):
        got = _ROWS[name](r)
        if isinstance(got, str):
            dropped[got] += 1
            continue
        out.append(_to_verl(data_source=name, source_id=str(r["id"]), split=split_name, index=i, **got))
    return out, {"hf_split": hf_split, "source_rows": len(rows), "selected": len(selected),
                 "kept": len(out), "dropped": dict(dropped)}


def _check(train_rows: list[dict], val_rows: list[dict]) -> None:
    """Every row answerable and uniquely identified; no question in both splits."""
    ids = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        keys = [(r["data_source"], r["extra_info"]["source_id"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{split}: duplicate source ids -- the source is not what it should be")
        bad = [k for k, r in zip(keys, rows) if not r["reward_model"]["ground_truth"]
               or not all(isinstance(a, str) and a for a in r["reward_model"]["ground_truth"])]
        if bad:
            raise SystemExit(f"{split}: rows without a usable gold answer: {bad[:5]}")
        ids[split] = set(keys)
    both = ids["train"] & ids["val"]
    if both:
        raise SystemExit(f"{len(both)} questions are in both train and val, e.g. {sorted(both)[:3]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=os.environ.get("QA_DATASETS", "musique"),
                   help=f"comma list of {sorted(SOURCES)}")
    p.add_argument("--out-dir", default=os.environ.get("QA_TOOL_OUT_DIR",
                   "/Volumes/main/mshtelma/verl/data/qa_musique"))
    p.add_argument("--train-limit", type=int, default=int(os.environ.get("QA_PREP_TRAIN_LIMIT", "0")))
    p.add_argument("--val-limit", type=int, default=int(os.environ.get("QA_PREP_VAL_LIMIT", "500")))
    p.add_argument("--cache", default=os.environ.get("QA_HF_CACHE", "/local_disk0/hf_cache"))
    args = p.parse_args(argv)

    names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown or not names:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(SOURCES)}")
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    sources = []
    for name in names:
        ds = SOURCES[name].load(cache_dir=args.cache)
        tr, tr_stats = _convert(name, ds, "train", "train", args.train_limit)
        va, va_stats = _convert(name, ds, "val", "validation", args.val_limit)
        print(f"[{name}] train={len(tr)} val={len(va)}  dropped train={tr_stats['dropped']} "
              f"val={va_stats['dropped']}")
        train_rows += tr
        val_rows += va
        sources.append(SOURCES[name].record(name=name, splits={"train": tr_stats, "val": va_stats}))
    if not train_rows or not val_rows:
        raise SystemExit(f"no rows produced (train={len(train_rows)}, val={len(val_rows)})")
    _check(train_rows, val_rows)

    # Re-index globally so extra_info.index is unique across merged sources.
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / dm.DIR_MANIFEST).unlink(missing_ok=True)   # never beside files it does not describe
    outputs = []
    for fname, rows in (("train.parquet", train_rows), ("test.parquet", val_rows)):
        path = dm.write_parquet(out_dir / fname, rows)
        uids = ({"uid": f"{r['data_source']}:{r['extra_info']['source_id']}"} for r in rows)
        outputs.append(dm.output_record(path, len(rows), question_ids_sha256=dm.content_digest(uids, ["uid"])))
        print(f"wrote {len(rows)} -> {path}")
    def rows_of(split: str, limit: int) -> str:
        return (f"each source's {split} split: {f'the first {limit} rows' if limit > 0 else 'all rows'}, "
                f"in source order, no shuffle")

    dm.write_manifest(
        out_dir / dm.DIR_MANIFEST, tool="usecases/agentic-search/prep_data.py", sources=sources,
        sampling={"train.parquet": rows_of("train", args.train_limit),
                  "test.parquet": rows_of("validation", args.val_limit)},
        filters=["drop unanswerable (MuSiQue answerable=False)", "drop rows with no question or no gold answer"],
        outputs=outputs)
    ex = train_rows[0]
    print(f"sample: data_source={ex['data_source']} id={ex['extra_info']['source_id']} "
          f"gt={ex['reward_model']['ground_truth']!r}\n  Q={ex['extra_info']['question'][:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
