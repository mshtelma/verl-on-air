#!/usr/bin/env python3
"""Prepare the agentic search/RAG (HotpotQA + NQ) tool-agent dataset for verl (Search-R1 style).

Emits verl's standard tool-agent parquet schema, so the fully-async
ToolAgentLoop + rule-based EM reward (usecases/agentic-search/reward.py) consume it unchanged:

    data_source   "hotpotqa" | "nq"
    agent_name    "tool_agent"
    prompt        [{role: system, content: SYSTEM_PROMPT}, {role: user, content: "Question: ..."}]
    ability       "qa_search"
    reward_model  {"style": "rule", "ground_truth": [gold, ...]}   # golden answers list
    extra_info    {split, index, data_source, question, answers, hop_type, level}

The model is told to use vector_search / keyword_search / read_article and commit its final answer in
<answer> ... </answer> (the only thing the reward reads). Ground truth lives ONLY in reward_model,
never in the prompt.

MVP = HotpotQA (its bundled `context` is a self-contained curated Wikipedia subset with distractors,
and it is the multi-hop showcase); ``--datasets hotpotqa,nq`` also folds in Natural Questions once the
wiki-18 subset backs NQ retrieval (build_corpus_and_index.py --datasets nq).

Env / CLI:
    QA_DATASETS         comma list: hotpotqa[,nq]      (default hotpotqa)
    QA_TOOL_OUT_DIR     output dir (default Volume data/qa_search)
    QA_PREP_TRAIN_LIMIT cap train rows (default 0 = all)
    QA_PREP_VAL_LIMIT   cap val rows   (default 500)
    QA_HF_CACHE         HF datasets cache dir (default /local_disk0/hf_cache)
    HF_TOKEN            HuggingFace token (from the air secret)
"""
from __future__ import annotations

import argparse
import os

import datasets

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


def _to_verl(*, data_source: str, question: str, golden_answers: list[str], split: str, index: int,
             hop_type: str = "", level: str = "") -> dict:
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


def _hotpotqa(cache: str, train_limit: int, val_limit: int) -> tuple[list[dict], list[dict]]:
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir=cache, trust_remote_code=True)
    def conv(split_name: str, hf_split: str, limit: int) -> list[dict]:
        rows = ds[hf_split]
        if limit and limit > 0:
            rows = rows.select(range(min(limit, len(rows))))
        out = []
        for i, r in enumerate(rows):
            ga = _clean_answers(r.get("answer"))
            q = (r.get("question") or "").strip()
            if not q or not ga:
                continue
            out.append(_to_verl(data_source="hotpotqa", question=q, golden_answers=ga, split=split_name,
                                index=i, hop_type=r.get("type", ""), level=r.get("level", "")))
        return out
    return conv("train", "train", train_limit), conv("val", "validation", val_limit)


def _musique(cache: str, train_limit: int, val_limit: int) -> tuple[list[dict], list[dict]]:
    # MuSiQue-Ans: harder 2-4 hop questions, built so single-hop retrieval shortcuts fail.
    # golden = answer + answer_aliases; keep only answerable items. hop_type = N-hop from the
    # decomposition length (useful for a difficulty breakdown in eval).
    ds = datasets.load_dataset("dgslibisey/MuSiQue", cache_dir=cache)
    def conv(split_name: str, hf_split: str, limit: int) -> list[dict]:
        rows = ds[hf_split]
        if limit and limit > 0:
            rows = rows.select(range(min(limit, len(rows))))
        out = []
        for i, r in enumerate(rows):
            if r.get("answerable") is False:
                continue
            ga = _clean_answers([r.get("answer")] + list(r.get("answer_aliases") or []))
            q = (r.get("question") or "").strip()
            if not q or not ga:
                continue
            hops = len(r.get("question_decomposition") or []) or 0
            out.append(_to_verl(data_source="musique", question=q, golden_answers=ga, split=split_name,
                                index=i, hop_type=(f"{hops}hop" if hops else "")))
        return out
    return conv("train", "train", train_limit), conv("val", "validation", val_limit)


def _twowiki(cache: str, train_limit: int, val_limit: int) -> tuple[list[dict], list[dict]]:
    ds = datasets.load_dataset("xanhho/2WikiMultihopQA", cache_dir=cache, trust_remote_code=True)
    def conv(split_name: str, hf_split: str, limit: int) -> list[dict]:
        rows = ds[hf_split]
        if limit and limit > 0:
            rows = rows.select(range(min(limit, len(rows))))
        out = []
        for i, r in enumerate(rows):
            ga = _clean_answers([r.get("answer")])
            q = (r.get("question") or "").strip()
            if not q or not ga:
                continue
            out.append(_to_verl(data_source="2wiki", question=q, golden_answers=ga, split=split_name,
                                index=i, hop_type=str(r.get("type", ""))))
        return out
    return conv("train", "train", train_limit), conv("val", "validation", val_limit)


def _nq(cache: str, train_limit: int, val_limit: int) -> tuple[list[dict], list[dict]]:
    # NQ-open: {question, answer:[...]}. Retrieval for NQ needs the wiki-18 subset in the VS index
    # (build_corpus_and_index.py --datasets nq); the parquet itself is corpus-agnostic.
    ds = datasets.load_dataset("google-research-datasets/nq_open", cache_dir=cache)
    def conv(split_name: str, hf_split: str, limit: int) -> list[dict]:
        rows = ds[hf_split]
        if limit and limit > 0:
            rows = rows.select(range(min(limit, len(rows))))
        out = []
        for i, r in enumerate(rows):
            ga = _clean_answers(r.get("answer"))
            q = (r.get("question") or "").strip()
            if not q or not ga:
                continue
            out.append(_to_verl(data_source="nq", question=q, golden_answers=ga, split=split_name, index=i))
        return out
    return conv("train", "train", train_limit), conv("val", "validation", val_limit)


_LOADERS = {"hotpotqa": _hotpotqa, "musique": _musique, "2wiki": _twowiki, "nq": _nq}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=os.environ.get("QA_DATASETS", "hotpotqa"),
                   help="comma list: hotpotqa[,nq]")
    p.add_argument("--out-dir", default=os.environ.get("QA_TOOL_OUT_DIR",
                   "/Volumes/main/mshtelma/verl/data/qa_search"))
    p.add_argument("--train-limit", type=int, default=int(os.environ.get("QA_PREP_TRAIN_LIMIT", "0")))
    p.add_argument("--val-limit", type=int, default=int(os.environ.get("QA_PREP_VAL_LIMIT", "500")))
    p.add_argument("--cache", default=os.environ.get("QA_HF_CACHE", "/local_disk0/hf_cache"))
    args = p.parse_args()

    names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for name in names:
        if name not in _LOADERS:
            raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(_LOADERS)}")
        tr, va = _LOADERS[name](args.cache, args.train_limit, args.val_limit)
        print(f"[{name}] train={len(tr)} val={len(va)}")
        train_rows += tr
        val_rows += va
    if not train_rows:
        raise SystemExit("no training rows produced")

    # Re-index globally so extra_info.index is unique across merged sources.
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i

    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    train_path = os.path.join(os.path.expanduser(args.out_dir), "train.parquet")
    val_path = os.path.join(os.path.expanduser(args.out_dir), "test.parquet")
    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_path)
    print(f"wrote {len(train_rows)} train -> {train_path}")
    print(f"wrote {len(val_rows)} val   -> {val_path}")
    ex = train_rows[0]
    print(f"sample: data_source={ex['data_source']} gt={ex['reward_model']['ground_truth']!r}\n"
          f"  Q={ex['extra_info']['question'][:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
