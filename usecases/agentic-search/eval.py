#!/usr/bin/env python3
"""Agentic search/RAG eval -- the HELD-OUT benchmark + OOB baseline probe for the RL run.

Mirrors eval_officeqa_agentic.py (the proven agentic-eval harness) but for NQ+HotpotQA:
  * prompts rendered with the model's OWN tokenizer chat template, tools=[...] (same template the
    ToolAgentLoop rollout uses);
  * the model's raw output parsed with verl's ``qwen3_coder`` ToolParser (regex fallback);
  * tool calls run the SAME plain impls the rollout uses (usecases/agentic-search/tool.py:
    _vector_search / _keyword_search / _read_article over the Vector Search index);
  * the final answer is extracted from ``<answer>...</answer>`` and scored with the SAME rule-based
    metric as the training reward (usecases/agentic-search/reward.py: EM / cover-EM / F1).

Run on the BASE model (baseline / OOB headroom probe) and the RL checkpoint with identical settings;
the delta is the demo result. The baseline probe answers the gate question: does the 35B land in the
learnable ~30-55% EM band before we spend a training run?

Serving: OpenAI-compatible vLLM endpoint (EVAL_BASE_URL, default http://127.0.0.1:8000/v1) via
/completions (raw text) so we control the prompt string exactly; engine/serve/serve_and_eval.sh brings the
server up first. The search tools query Databricks Vector Search (QA_VS_ENDPOINT / QA_VS_INDEX).

Knobs (env): EVAL_BASE_URL, EVAL_MODEL, EVAL_MAX_TURNS (8), EVAL_MAX_TOKENS (512), EVAL_MAX_CONT (2),
EVAL_TEMPERATURE (0), EVAL_CONCURRENCY (32), EVAL_LIMIT (0=all), EVAL_OUT, EVAL_TRACE_OUT,
MODEL_PATH (tokenizer), QA_VAL_PARQUET (held-out set), QA_REWARD_METRIC (headline em|cover_em).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

try:
    import aiohttp
except ImportError:  # noqa: BLE001
    aiohttp = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from prep_data import SYSTEM_PROMPT  # noqa: E402  (shared with the training data prep)
from reward import (  # noqa: E402  (the SAME scorer as the training reward)
    _f1, _gold_list, cover_em_check, em_check, extract_answer, normalize_answer,
)
import tool as _qst  # noqa: E402  (for pre-warm)
from tool import _keyword_search, _read_article, _vector_search  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SERVED_MODEL = os.environ.get("EVAL_MODEL", "eval")
VAL_PARQUET = os.environ.get("QA_VAL_PARQUET", "/Volumes/main/mshtelma/verl/data/qa_search/test.parquet")
MAX_TURNS = int(os.environ.get("EVAL_MAX_TURNS", "8"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "512"))
MAX_CONT = int(os.environ.get("EVAL_MAX_CONT", "2"))
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
OUT = os.environ.get("EVAL_OUT", "")
TRACE_OUT = os.environ.get("EVAL_TRACE_OUT", "")
REQ_TIMEOUT = float(os.environ.get("EVAL_REQ_TIMEOUT", "900"))
_HTTP_RETRIES = int(os.environ.get("EVAL_HTTP_RETRIES", "4"))
HEADLINE = os.environ.get("QA_REWARD_METRIC", "em").strip().lower()   # em | cover_em

# Final-turn nudge: commit an answer instead of exhausting the budget mid-search.
FINAL_NUDGE = (
    "You have reached your final step and must NOT call any more tools. Using ONLY the passages "
    "already retrieved above, give your answer now on a single line as <answer>value</answer>. "
    "The answer must be a short span with no extra words."
)

# OpenAI tool schemas (rendered into the chat template; mirror qa_search_tools sigs).
TOOLS = [
    {"type": "function", "function": {
        "name": "vector_search",
        "description": ("Semantic search over the Wikipedia corpus; returns the most relevant "
                        "passages, each with its article title."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "top_k": {"type": "integer", "description": "How many passages to return (default 5; max 20)."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "keyword_search",
        "description": ("Hybrid keyword+semantic search over the Wikipedia corpus; best when exact "
                        "terms (proper nouns, titles, numbers) must match. Returns passages with titles."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for; include exact names/terms."},
            "top_k": {"type": "integer", "description": "How many passages to return (default 5; max 20)."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_article",
        "description": ("Read the full passage(s) of one Wikipedia article by its EXACT title (as "
                        "shown in a search result). Use to follow a multi-hop link."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "The exact article title, e.g. 'Christopher Nolan'."},
        }, "required": ["title"]}}},
]

TOOL_IMPLS = {
    "vector_search": lambda a: _vector_search(a.get("query", ""), a.get("top_k", 5)),
    "keyword_search": lambda a: _keyword_search(a.get("query", ""), a.get("top_k", 5)),
    "read_article": lambda a: _read_article(a.get("title", "")),
}


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _load_parser(tok):
    """verl qwen3_coder ToolParser; regex fallback so a parser change never silently zeros tools."""
    try:
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema
        try:
            schemas = [OpenAIFunctionToolSchema.model_validate(t) for t in TOOLS]
        except Exception:  # noqa: BLE001
            schemas = None
        qp = ToolParser.get_tool_parser("qwen3_coder", tok)

        def parse(text: str):
            calls = []
            for s in qp._get_function_calls(text):
                p = qp._parse_xml_function_call(s, schemas)
                if p is None:
                    continue
                args = p.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                calls.append((p.name, args or {}))
            return calls
        print("[eval] tool parser: verl qwen3_coder", flush=True)
        return parse
    except Exception as e:  # noqa: BLE001
        print(f"[eval] verl qwen3_coder unavailable ({type(e).__name__}: {e}); using regex fallback", flush=True)

    _CALL_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
    _PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)

    def parse_regex(text: str):
        calls = []
        for name, body in _CALL_RE.findall(text):
            args = {k: v.strip("\n") for k, v in _PARAM_RE.findall(body)}
            calls.append((name.strip(), args))
        return calls
    return parse_regex


def _render(tok, messages) -> str:
    try:
        return tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001
        safe = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls") and m.get("content"):
                m = {**m, "content": ""}
            safe.append(m)
        return tok.apply_chat_template(safe, tools=TOOLS, add_generation_prompt=True, tokenize=False)


_TOOLCALL_OPEN_RE = re.compile(r"<tool_call>|<function=", re.IGNORECASE)


async def _post(session, payload):
    last = None
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            async with session.post(f"{BASE_URL}/completions", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data["choices"][0]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:  # noqa: BLE001
            last = e
            if attempt < _HTTP_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise last


async def _complete(session, prompt: str):
    ch = await _post(session, {
        "model": SERVED_MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "stop": ["<|im_end|>", "</tool_call>"], "include_stop_str_in_output": True,
    })
    return ch["text"], ch.get("finish_reason")


async def _assistant_turn(session, base_prompt: str):
    acc = ""
    for cont in range(MAX_CONT + 1):
        text, fr = await _complete(session, base_prompt + acc)
        acc += text
        if fr != "length":
            return acc, cont, False
    return acc, MAX_CONT, True


async def _forced_answer(session, base_prompt: str) -> str:
    ch = await _post(session, {
        "model": SERVED_MODEL, "prompt": base_prompt + "<answer>", "max_tokens": 64,
        "temperature": TEMPERATURE, "stop": ["</answer>", "<|im_end|>"],
    })
    return "<answer>" + (ch.get("text") or "").strip() + "</answer>"


def _split_reasoning(text: str) -> str:
    m = _TOOLCALL_OPEN_RE.search(text)
    return (text[: m.start()] if m else text).strip()


async def _run_one(session, tok, parse, sem, ex) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        tool_counts = {k: 0 for k in TOOL_IMPLS}
        n_tool, n_tool_err, turns, n_cont, truncated = 0, 0, 0, 0, False
        assistant_texts, steps = [], []
        for turn in range(MAX_TURNS):
            turns = turn + 1
            if turn == MAX_TURNS - 1:
                messages.append({"role": "user", "content": FINAL_NUDGE})
                try:
                    forced = await _forced_answer(session, _render(tok, messages))
                except Exception as e:  # noqa: BLE001
                    forced = f"[error {type(e).__name__}: {e}]"
                assistant_texts.append(forced)
                steps.append({"turn": turn, "reasoning": forced, "tool_calls": [], "tool_results": []})
                break
            try:
                text, c, tr = await _assistant_turn(session, _render(tok, messages))
            except Exception as e:  # noqa: BLE001
                assistant_texts.append(f"[error {type(e).__name__}: {e}]")
                break
            n_cont += c
            truncated = tr
            assistant_texts.append(text)
            calls = parse(text)
            if not calls:
                steps.append({"turn": turn, "reasoning": _split_reasoning(text), "tool_calls": [], "tool_results": []})
                break
            messages.append({
                "role": "assistant", "content": _split_reasoning(text),
                "tool_calls": [{"id": f"c{turn}_{i}", "type": "function",
                                "function": {"name": n, "arguments": a}}
                               for i, (n, a) in enumerate(calls)],
            })
            step_results = []
            for i, (name, args) in enumerate(calls):
                impl = TOOL_IMPLS.get(name)
                if impl is None:
                    res = f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_IMPLS)}."
                    n_tool_err += 1
                else:
                    try:
                        res = await asyncio.get_event_loop().run_in_executor(
                            None, impl, args if isinstance(args, dict) else {})
                    except Exception as e:  # noqa: BLE001
                        res = f"Error: tool {name} failed: {type(e).__name__}: {e}"
                        n_tool_err += 1
                    else:
                        tool_counts[name] += 1
                        if isinstance(res, str) and res.startswith("Error:"):
                            n_tool_err += 1
                n_tool += 1
                step_results.append({"name": name, "result": str(res)})
                messages.append({"role": "tool", "content": str(res), "tool_call_id": f"c{turn}_{i}", "name": name})
            steps.append({"turn": turn, "reasoning": _split_reasoning(text),
                          "tool_calls": [{"name": n, "args": a} for n, a in calls],
                          "tool_results": step_results})

        full_output = "\n".join(assistant_texts)
        pred = extract_answer(full_output)
        golds_norm = [normalize_answer(g) for g in ex["gt"]]
        if pred is None:
            em = cover = f1 = 0.0
        else:
            pn = normalize_answer(pred)
            em = float(em_check(pn, golds_norm))
            cover = float(cover_em_check(pn, golds_norm))
            f1 = float(_f1(pn, golds_norm))
        hit = cover if HEADLINE == "cover_em" else em
        return {
            "uid": ex["uid"], "data_source": ex.get("data_source", ""), "hop_type": ex.get("hop_type", ""),
            "question": ex["question"], "gt": ex["gt"], "pred": pred, "correct": bool(hit > 0),
            "em": em, "cover_em": cover, "f1": f1,
            "n_tool": n_tool, "n_tool_err": n_tool_err, "tool_counts": tool_counts,
            "turns": turns, "n_cont": n_cont, "truncated": bool(truncated),
            "final_tail": full_output[-400:], "_trace": steps,
        }


def _load_qa():
    import datasets as _d
    ds = _d.Dataset.from_parquet(VAL_PARQUET)
    rows = []
    for i, r in enumerate(ds):
        ei = r.get("extra_info") or {}
        q = (ei.get("question") or "").strip()
        gts = _gold_list((r.get("reward_model") or {}).get("ground_truth"))
        if not q or not gts:
            continue
        rows.append({"uid": str(ei.get("index", i)), "question": q, "gt": gts,
                     "data_source": r.get("data_source", ""), "hop_type": ei.get("hop_type", "")})
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


async def _main_async():
    print(f"[eval] agentic search eval | model={SERVED_MODEL} url={BASE_URL} turns<={MAX_TURNS} "
          f"max_tok={MAX_TOKENS} temp={TEMPERATURE} conc={CONCURRENCY} headline={HEADLINE}", flush=True)
    tok = _load_tokenizer()
    parse = _load_parser(tok)
    rows = _load_qa()
    print(f"[eval] loaded {len(rows)} questions from {VAL_PARQUET}", flush=True)

    # Pre-warm the Vector Search client/index handle up front (one query) so establishing the
    # connection doesn't freeze the event loop mid-run (the officeqa lesson: lazy build -> vLLM
    # keep-alive drops -> ServerDisconnected). A failure here surfaces auth/config immediately.
    tw = time.time()
    warm = _vector_search("test connectivity", 1)
    print(f"[eval] pre-warmed VS in {time.time()-tw:.0f}s (endpoint={_qst.QA_VS_ENDPOINT!r} "
          f"index={_qst.QA_VS_INDEX!r}) -> {warm[:80]!r}", flush=True)

    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[_run_one(session, tok, parse, sem, r) for r in rows])
    dt = time.time() - t0

    from collections import Counter
    n = len(results)
    em = sum(r["em"] for r in results) / n if n else 0.0
    cover = sum(r["cover_em"] for r in results) / n if n else 0.0
    f1 = sum(r["f1"] for r in results) / n if n else 0.0
    no_ans = sum(1 for r in results if not r["pred"])
    answered = n - no_ans
    used_tool = sum(1 for r in results if r["n_tool"] > 0)
    mean_tool = sum(r["n_tool"] for r in results) / n if n else 0
    tool_err = sum(r["n_tool_err"] for r in results)
    tool_totals = Counter()
    for r in results:
        for k, v in r["tool_counts"].items():
            tool_totals[k] += v
    by_src = {}
    for r in results:
        by_src.setdefault(r["data_source"] or "?", []).append(r)

    print("\n==================== AGENTIC SEARCH (NQ+HotpotQA) RESULT ====================", flush=True)
    print(f"model={SERVED_MODEL}  n={n}  wall={dt:.0f}s", flush=True)
    print(f"EM={em:.4f}   cover_EM={cover:.4f}   F1={f1:.4f}   (headline={HEADLINE})", flush=True)
    print(f"answered={answered}/{n}  (no_answer={no_ans})", flush=True)
    for src in sorted(by_src):
        rs = by_src[src]
        e = sum(x["em"] for x in rs) / len(rs)
        print(f"  {src:>10s} ({len(rs):>3d}):  EM={e*100:5.1f}%", flush=True)
    print(f"agentic: used_tool={used_tool}/{n}  mean_tool_calls={mean_tool:.2f}  tool_errors={tool_err}", flush=True)
    print("tool usage: " + "  ".join(f"{k}={tool_totals[k]}" for k in TOOL_IMPLS), flush=True)
    print("---- 4 sample trajectories (tail) ----", flush=True)
    for r in results[:4]:
        print(f"  [{r['data_source']} {'OK' if r['correct'] else 'XX'}] gt={r['gt']!r} pred={r['pred']!r} "
              f"tools={r['n_tool']} turns={r['turns']}", flush=True)
    print("============================================================================", flush=True)

    if TRACE_OUT:
        os.makedirs(os.path.dirname(TRACE_OUT) or ".", exist_ok=True)
        keep = ("uid", "data_source", "hop_type", "question", "gt", "pred", "correct",
                "em", "cover_em", "f1", "n_tool", "turns", "truncated")
        with open(TRACE_OUT, "w") as fh:
            for r in results:
                rec = {k: r.get(k) for k in keep}
                rec["trajectory"] = r.get("_trace", [])
                fh.write(json.dumps(rec) + "\n")
        print(f"[eval] wrote {len(results)} trajectory traces -> {TRACE_OUT}", flush=True)
    for r in results:
        r.pop("_trace", None)

    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump({
                "model": SERVED_MODEL, "val_parquet": VAL_PARQUET, "n": n, "headline_metric": HEADLINE,
                "em": em, "cover_em": cover, "f1": f1, "answered": answered, "no_answer": no_ans,
                "by_data_source": {s: {"n": len(rs), "em": sum(x["em"] for x in rs) / len(rs)}
                                   for s, rs in by_src.items()},
                "mean_tool_calls": mean_tool, "used_tool": used_tool, "tool_totals": dict(tool_totals),
                "tool_errors": tool_err, "wall_s": dt, "results": results,
            }, fh, indent=2)
        print(f"[eval] wrote {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main_async())
