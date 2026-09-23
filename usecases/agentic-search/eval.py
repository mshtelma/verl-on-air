#!/usr/bin/env python3
"""Agentic search/RAG eval -- the HELD-OUT benchmark + OOB baseline probe for the RL run.

The agentic-eval harness for this use case:
  * prompts rendered with the model's OWN tokenizer chat template, tools=[...] (same template the
    ToolAgentLoop rollout uses);
  * the model's raw output parsed with verl's ``qwen3_coder`` ToolParser (regex fallback);
  * tool calls run the SAME plain impls the rollout uses (usecases/agentic-search/tool.py:
    vector_search_impl / keyword_search_impl / read_article_impl over the Vector Search index) --
    the raising form, so a retrieval OUTAGE is recorded as infrastructure, never as a wrong answer;
  * the final answer is extracted from ``<answer>...</answer>`` and scored with the SAME rule-based
    metric as the training reward (usecases/agentic-search/reward.py: EM / cover-EM / F1).

Follows engine/serve/eval_contract.py: readiness before any question (served model listed, one real
retrieval), a per-question status (only `scored` questions are graded), transient-only retries, a
validity verdict (expected question count + infrastructure error budget) in the artifact and the exit
code, atomic never-overwriting artifacts with the model identity + dataset fingerprint, and one
closed file per finished question under <EVAL_OUT>.parts/.

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
    _gold_list, _last_answer, score_segments,
)
import tool as _qst  # noqa: E402  (the rollout's own tool impls)

sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "engine", "serve"))
import eval_contract as ec  # noqa: E402
import data_manifest as dm  # noqa: E402  (engine/lib, on the path via prep_data)

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SERVED_MODEL = os.environ.get("EVAL_MODEL", "eval")
VAL_PARQUET = os.environ.get("QA_VAL_PARQUET", "/Volumes/main/mshtelma/verl/data/qa_musique/test.parquet")
MAX_TURNS = int(os.environ.get("EVAL_MAX_TURNS", "8"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "512"))
MAX_CONT = int(os.environ.get("EVAL_MAX_CONT", "2"))
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
OUT = os.environ.get("EVAL_OUT", "")
TRACE_OUT = os.environ.get("EVAL_TRACE_OUT", "")
REQ_TIMEOUT = float(os.environ.get("EVAL_REQ_TIMEOUT", "900"))
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

# The RAISING impls (tool.ToolInfraError on a backend failure), looked up at call time.
TOOL_IMPLS = {
    "vector_search": lambda a: _qst.vector_search_impl(a.get("query", ""), a.get("top_k", 5)),
    "keyword_search": lambda a: _qst.keyword_search_impl(a.get("query", ""), a.get("top_k", 5)),
    "read_article": lambda a: _qst.read_article_impl(a.get("title", "")),
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
    """One /completions call; raises ec.InfraError (transient failures retried inside)."""
    data = await ec.post_json(session, f"{BASE_URL}/completions", payload)
    try:
        return data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise ec.InfraError("infra_inference", f"malformed completion: {str(data)[:200]}") from None


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


async def _run_one(session, tok, parse, sem, ex, parts) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {ex['question']}"},
        ]
        tool_counts = {k: 0 for k in TOOL_IMPLS}
        n_tool, n_tool_err, turns, n_cont, truncated = 0, 0, 0, 0, False
        assistant_texts, steps = [], []
        status, infra_detail = "scored", ""
        try:
            for turn in range(MAX_TURNS):
                turns = turn + 1
                if turn == MAX_TURNS - 1:
                    messages.append({"role": "user", "content": FINAL_NUDGE})
                    forced = await _forced_answer(session, _render(tok, messages))
                    assistant_texts.append(forced)
                    steps.append({"turn": turn, "reasoning": forced, "tool_calls": [], "tool_results": []})
                    break
                text, c, tr = await _assistant_turn(session, _render(tok, messages))
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
                    if impl is None:  # the MODEL called a tool that does not exist: policy, scored
                        res = f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_IMPLS)}."
                        n_tool_err += 1
                    else:
                        try:
                            res = await asyncio.get_running_loop().run_in_executor(
                                None, impl, args if isinstance(args, dict) else {})
                        except _qst.ToolInfraError as e:
                            raise ec.InfraError("infra_retrieval", str(e)) from None
                        except Exception as e:  # noqa: BLE001 - a tool bug is ours, not the model's
                            raise ec.InfraError("infra_tool", f"{name}: {type(e).__name__}: {e}") from None
                        tool_counts[name] += 1
                        if isinstance(res, str) and res.startswith("Error:"):
                            n_tool_err += 1   # bad arguments etc. -- the model's doing
                    n_tool += 1
                    step_results.append({"name": name, "result": str(res)})
                    messages.append({"role": "tool", "content": str(res), "tool_call_id": f"c{turn}_{i}", "name": name})
                steps.append({"turn": turn, "reasoning": _split_reasoning(text),
                              "tool_calls": [{"name": n, "args": a} for n, a in calls],
                              "tool_results": step_results})
        except ec.InfraError as e:
            status, infra_detail = e.kind, e.detail
        except Exception as e:  # noqa: BLE001 - never let the harness score its own bug
            status, infra_detail = "infra_harness", f"{type(e).__name__}: {e}"

        full_output = "\n".join(assistant_texts)
        # The training reward's own scorer, on the same split it sees in training: what the model
        # wrote (assistant_texts) vs what the tools returned.
        tool_texts = [tr["result"] for st in steps for tr in st["tool_results"]]
        sc = score_segments(assistant_texts, tool_texts, ex["gt"])
        pred = _last_answer(assistant_texts) if status == "scored" else None
        em, cover, f1 = (sc["em"], sc["cover_em"], sc["f1"]) if status == "scored" else (0.0, 0.0, 0.0)
        hit = cover if HEADLINE == "cover_em" else em
        rec = {
            "uid": ex["uid"], "data_source": ex.get("data_source", ""), "hop_type": ex.get("hop_type", ""),
            "question": ex["question"], "gt": ex["gt"], "status": status, "infra_detail": infra_detail[:400],
            "pred": pred, "correct": bool(status == "scored" and hit > 0),
            "em": em, "cover_em": cover, "f1": f1,
            "gold_retrieved": sc["gold_retrieved"] if status == "scored" else 0.0,
            "n_tool": n_tool, "n_tool_err": n_tool_err, "tool_counts": tool_counts,
            "turns": turns, "n_cont": n_cont, "truncated": bool(truncated),
            "final_tail": full_output[-400:],
        }
        parts.write(ex["uid"], {**rec, "trajectory": steps})
        return {**rec, "_trace": steps}


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


def _policy() -> dict:
    """The eval policy recorded in every artifact -- part of what a number means."""
    import hashlib
    return {
        "max_turns": MAX_TURNS, "max_tokens_per_request": MAX_TOKENS, "max_continuations": MAX_CONT,
        "force_final_answer": True, "temperature": TEMPERATURE, "headline": HEADLINE,
        "tools": [t["function"]["name"] for t in TOOLS],
        "tool_schema_sha256": hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "search_top_k": os.environ.get("QA_SEARCH_TOP_K", "5"), "vs_index": _qst.QA_VS_INDEX,
    }


async def _main_async() -> int:
    print(f"[eval] agentic search eval | model={SERVED_MODEL} url={BASE_URL} turns<={MAX_TURNS} "
          f"max_tok={MAX_TOKENS} temp={TEMPERATURE} conc={CONCURRENCY} headline={HEADLINE}", flush=True)
    ec.refuse_overwrite(OUT, TRACE_OUT)
    started = time.time()
    tok = _load_tokenizer()
    parse = _load_parser(tok)
    rows = _load_qa()
    n_expected = ec._env_int("EVAL_EXPECT_N", LIMIT if LIMIT > 0 else None)
    print(f"[eval] loaded {len(rows)} questions from {VAL_PARQUET} (expected {n_expected})", flush=True)

    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # --- readiness: fail before the first question, not as N silently-wrong answers ---
        try:
            await ec.check_served_model(session, BASE_URL, SERVED_MODEL)
        except ec.InfraError as e:
            ec.fatal_not_ready("the served model", e)
        # Also pre-warms the Vector Search client/index handle (one query), so establishing the
        # connection doesn't freeze the event loop mid-run (lazy build -> vLLM keep-alive drops ->
        # ServerDisconnected).
        tw = time.time()
        try:
            warm = await asyncio.get_running_loop().run_in_executor(None, _qst.vector_search_impl,
                                                                    "test connectivity", 1)
        except _qst.ToolInfraError as e:
            ec.fatal_not_ready(f"retrieval (index {_qst.QA_VS_INDEX!r})", e)
        if not warm.startswith("[0]"):
            ec.fatal_not_ready(f"retrieval (index {_qst.QA_VS_INDEX!r})", RuntimeError(f"no passages: {warm[:120]!r}"))
        print(f"[eval] ready: model {SERVED_MODEL!r} served; retrieval answered in {time.time()-tw:.0f}s "
              f"(index={_qst.QA_VS_INDEX!r})", flush=True)

        t0 = time.time()
        sem = asyncio.Semaphore(CONCURRENCY)
        parts = ec.PartsWriter(OUT)
        results = await asyncio.gather(*[_run_one(session, tok, parse, sem, r, parts) for r in rows])
    dt = time.time() - t0

    from collections import Counter
    v = ec.verdict(results, n_loaded=len(rows), n_expected=n_expected)
    scored = [r for r in results if r["status"] == "scored"]
    n = len(scored)
    em = sum(r["em"] for r in scored) / n if n else 0.0
    cover = sum(r["cover_em"] for r in scored) / n if n else 0.0
    f1 = sum(r["f1"] for r in scored) / n if n else 0.0
    no_ans = sum(1 for r in scored if not r["pred"])
    answered = n - no_ans
    used_tool = sum(1 for r in scored if r["n_tool"] > 0)
    mean_tool = sum(r["n_tool"] for r in scored) / n if n else 0
    tool_err = sum(r["n_tool_err"] for r in scored)
    tool_totals = Counter()
    for r in scored:
        for k, c in r["tool_counts"].items():
            tool_totals[k] += c
    by_src = {}
    for r in scored:
        by_src.setdefault(r["data_source"] or "?", []).append(r)

    print("\n==================== AGENTIC SEARCH RESULT ====================", flush=True)
    print(f"model={SERVED_MODEL}  scored={n}/{len(rows)}  valid={v['valid']}  wall={dt:.0f}s", flush=True)
    print(f"EM={em:.4f}   cover_EM={cover:.4f}   F1={f1:.4f}   (headline={HEADLINE}, over scored questions)", flush=True)
    print(f"answered={answered}/{n}  (no_answer={no_ans})   infra={v['infra_errors']}", flush=True)
    for src in sorted(by_src):
        rs = by_src[src]
        e = sum(x["em"] for x in rs) / len(rs)
        print(f"  {src:>10s} ({len(rs):>3d}):  EM={e*100:5.1f}%", flush=True)
    print(f"agentic: used_tool={used_tool}/{n}  mean_tool_calls={mean_tool:.2f}  tool_errors={tool_err}", flush=True)
    print("tool usage: " + "  ".join(f"{k}={tool_totals[k]}" for k in TOOL_IMPLS), flush=True)
    print("===============================================================", flush=True)

    if TRACE_OUT:
        keep = ("uid", "data_source", "hop_type", "question", "gt", "status", "pred", "correct",
                "em", "cover_em", "f1", "n_tool", "turns", "truncated")
        tmp = f"{TRACE_OUT}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(TRACE_OUT) or ".", exist_ok=True)
        with open(tmp, "w") as fh:
            for r in results:
                fh.write(json.dumps({**{k: r.get(k) for k in keep}, "trajectory": r.get("_trace", [])}) + "\n")
        os.replace(tmp, TRACE_OUT)
        print(f"[eval] wrote {len(results)} trajectory traces -> {TRACE_OUT}", flush=True)
    for r in results:
        r.pop("_trace", None)

    if OUT:
        ec.write_json_atomic(OUT, {
            **ec.header(dataset={**dm.provenance(VAL_PARQUET), "limit": LIMIT},
                        question_ids=[r["uid"] for r in rows], policy=_policy(), started_at=started),
            **v,
            "model": SERVED_MODEL, "val_parquet": VAL_PARQUET, "n": n, "headline_metric": HEADLINE,
            "em": em, "cover_em": cover, "f1": f1, "answered": answered, "no_answer": no_ans,
            "by_data_source": {s_: {"n": len(rs), "em": sum(x["em"] for x in rs) / len(rs)}
                               for s_, rs in by_src.items()},
            "mean_tool_calls": mean_tool, "used_tool": used_tool, "tool_totals": dict(tool_totals),
            "tool_errors": tool_err, "wall_s": dt, "results": results,
        })
        print(f"[eval] wrote {OUT}", flush=True)
    return ec.report_and_exit_code(v)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main_async()))
