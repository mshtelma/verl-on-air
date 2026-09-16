#!/usr/bin/env python3
"""Agentic OfficeQA eval -- the HELD-OUT benchmark for the RL run.

Goal (user): show a REAL, meaningful improvement on a hard, uncontaminated,
genuinely-agentic benchmark. Competition math is correctness-saturated for this
model (~95-100% right whenever it answers, even on fresh 2026 sets), so we moved
to OfficeQA -- Databricks' grounded-reasoning benchmark over U.S. Treasury
Bulletins (1939-2025). Frontier LLMs score <34% *with* the corpus, so there is
large real headroom. The model must GROUND its answer in retrieved figures via a
small tool set (search / grep / read / list / compute), then emit a single
``<FINAL_ANSWER>value</FINAL_ANSWER>``. We score with the official OfficeQA
``score_answer`` (unit-aware exact/fuzzy match). Run on the BASE model (baseline)
and the RL checkpoint with identical settings; the delta is the result.

FAITHFUL TO TRAINING (not vLLM's server tool-parsing, which is uncertain for
Qwen3.5 on vLLM 0.24):
  * prompts are rendered with the model's OWN tokenizer chat template, tools=[...]
    (same template the ToolAgentLoop rollout uses);
  * the model's raw output is parsed with verl's ``qwen3_coder`` ToolParser
    (Qwen3XMLToolParser) -- the exact parser air/53/air/55 proved for this tokenizer;
  * tool calls run the SAME plain impls the rollout uses
    (scripts/tools/officeqa_tools.py: _search_documents / _grep_documents /
    _read_document / _list_documents / _compute);
  * the final answer is extracted with XMLTagExtractor(FINAL_ANSWER) and scored
    with the official reward.score_answer (same matcher as training-time reward).

Serving: talks to an OpenAI-compatible vLLM endpoint (EVAL_BASE_URL, default
http://127.0.0.1:8000/v1) via /completions (raw text) so we control the prompt
string exactly; scripts/serve_and_eval.sh brings the server up first.

Knobs (env): EVAL_BASE_URL, EVAL_MODEL (served name), EVAL_MAX_TURNS (10),
EVAL_MAX_TOKENS (2048/turn), EVAL_MAX_CONT (2), EVAL_TEMPERATURE (0),
EVAL_CONCURRENCY (32), EVAL_LIMIT (0=all; >0 = smoke), EVAL_DIFFICULTY
(""|easy|hard), EVAL_OUT (json results path), MODEL_PATH (tokenizer),
OFFICEQA_EVAL_CSV (dataset CSV: officeqa_full.csv | officeqa_pro.csv).
Tool data paths (OFFICEQA_CHUNKS / OFFICEQA_CORPUS_DIR) are read by officeqa_tools.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time

try:
    import aiohttp  # present in the serving image; absent in some local offline test envs
except ImportError:  # noqa: BLE001
    aiohttp = None

# reuse the EXACT training-time scorer + tool impls, from sibling script dirs
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from reward.answer_extract import XMLTagExtractor  # noqa: E402
from reward.officeqa_reward import score_answer  # noqa: E402
from tools import officeqa_tools as _oqt  # noqa: E402  (for pre-warm)
from tools.officeqa_tools import (  # noqa: E402
    _compute,
    _grep_documents,
    _list_documents,
    _read_document,
    _search_documents,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SERVED_MODEL = os.environ.get("EVAL_MODEL", "eval")
EVAL_CSV = os.environ.get(
    "OFFICEQA_EVAL_CSV", "/Volumes/main/mshtelma/verl/data/officeqa/officeqa_full.csv"
)
EVAL_DIFFICULTY = os.environ.get("EVAL_DIFFICULTY", "").strip().lower()  # ""|easy|hard
MAX_TURNS = int(os.environ.get("EVAL_MAX_TURNS", "16"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "2048"))
# A turn cut off at max_tokens (finish_reason=="length") is NOT done -- continue it
# up to this many times so a long chain-of-thought still reaches its tool call or
# <FINAL_ANSWER> instead of being scored as "no answer".
MAX_CONT = int(os.environ.get("EVAL_MAX_CONT", "2"))
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
OUT = os.environ.get("EVAL_OUT", "")
REQ_TIMEOUT = float(os.environ.get("EVAL_REQ_TIMEOUT", "900"))
_HTTP_RETRIES = int(os.environ.get("EVAL_HTTP_RETRIES", "4"))   # retry transient vLLM disconnects
TOLERANCES = [0.0, 0.01, 0.05]   # exact + fuzzy (report all; headline = exact 0.0)

# --- trajectory tracing (for the grounding judge / process reward) -----------
# EVAL_TRACE=1 captures a STRUCTURED trajectory per question (reasoning + each tool
# call's args + the tool output). It is (a) logged to MLflow as spans -- inspectable
# in the MLflow UI + queryable via mlflow.search_traces -- and (b) written as a JSONL
# bundle (EVAL_TRACE_OUT) the grounding-judge job consumes. Off by default so a plain
# eval is unchanged.
TRACE = os.environ.get("EVAL_TRACE", "0") == "1"
TRACE_OUT = os.environ.get("EVAL_TRACE_OUT", "")           # JSONL bundle path (e.g. on the Volume)
MLFLOW_TRACE = TRACE and os.environ.get("EVAL_MLFLOW_TRACE", "1") == "1"

_extractor = XMLTagExtractor(tag="FINAL_ANSWER")

SYSTEM_PROMPT = (
    "You are a meticulous financial-data research agent answering questions about "
    "U.S. Treasury Bulletins (monthly, 1939 onward). You MUST ground every answer in "
    "figures you actually retrieve -- never guess or use prior knowledge.\n\n"
    "Tools:\n"
    "- search_documents(query, top_k): find the file/table for a figure. Each result "
    "names its source file and YYYY-MM date. Treasury Bulletins report PRIOR periods, "
    "so a calendar-1940 figure is usually in a 1940 or 1941 bulletin.\n"
    "- grep_documents(pattern, file_name, year, regex): grep for an exact string over "
    "the files (like grep). The most reliable way to PIN a row label or value, e.g. "
    "grep_documents('National defense', year='1941'). Returns file:line matches.\n"
    "- read_document(file_name, start_line, num_lines): read the table in full "
    "(Markdown); jump to a line grep reported and read around it.\n"
    "- list_documents(year): list available bulletins.\n"
    "- compute(code): run Python (numpy/pandas available) for ALL arithmetic -- sum "
    "the monthly cells, differences, ratios, %-change -- so you never miscalculate. "
    "Each call is a fresh sandbox; pass self-contained code, e.g. "
    "print(sum([132,129,143])).\n\n"
    "Method: search to find the file -> grep_documents to locate the EXACT row (verify "
    "the label matches the requested category exactly; Treasury tables have "
    "near-identical labels) -> read_document to see the row's cells and the column "
    "headers/periods -> compute to calculate. Figures are 'in millions of dollars' "
    "unless the table says otherwise; answer in the units the QUESTION asks. Work "
    "efficiently and commit as soon as you have the figures.\n\n"
    "When confident, stop calling tools and output EXACTLY ONE line, nothing else:\n"
    "<FINAL_ANSWER>value</FINAL_ANSWER>\n"
    "Put ONLY the final value inside the tag. Strip $ and USD unless a dollar format "
    "is asked. If the corpus genuinely lacks the answer, output "
    "<FINAL_ANSWER>DATA NOT AVAILABLE</FINAL_ANSWER>."
)

# Injected as a user turn on the LAST allowed step so the model commits an answer
# instead of exhausting the budget mid-retrieval (smoke 326906562878044: all 24 hit
# the turn cap still searching, calculator used once total -> never reached answering).
FINAL_NUDGE = (
    "You have reached your final step and must NOT call any more tools. Using ONLY the "
    "figures already retrieved above, give your answer now on a single line as "
    "<FINAL_ANSWER>value</FINAL_ANSWER> (strip $ and commas per the instructions). If the "
    "needed figures were not found, answer <FINAL_ANSWER>DATA NOT AVAILABLE</FINAL_ANSWER>."
)

# OpenAI tool schemas (rendered into the chat template; mirror officeqa_tools sigs).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the Treasury Bulletin corpus and return the most relevant "
                "passages, each with its source document and YYYY-MM bulletin date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'national defense expenditures 1940'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (default 6; max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_documents",
            "description": (
                "Grep for an exact text pattern over the Treasury Bulletin files (like "
                "grep). The most reliable way to pin a row label/category/value. Returns "
                "'file:line: text'. Substring (default, case-insensitive) or regex."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex, e.g. 'National defense' or '2,602'."},
                    "file_name": {"type": "string", "description": "Optional file to search within."},
                    "year": {"type": "string", "description": "Optional 4-digit year filter (ignored if file_name set)."},
                    "regex": {"type": "boolean", "description": "If true, treat pattern as a regex (default false)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read a slice of one Treasury Bulletin document by exact file name "
                "(financial tables are Markdown). Lines are numbered; page on by "
                "increasing start_line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Document file name, e.g. 'treasury_bulletin_1940_06.txt'.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "0-based line to start from (default 0).",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "How many lines to return (default 100; capped at 400).",
                    },
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List available Treasury Bulletin documents, optionally filtered by 4-digit year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "string",
                        "description": "Optional 4-digit year, e.g. '1940'. Empty lists all (capped).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute",
            "description": (
                "Execute Python (numpy/pandas available) for calculations and table "
                "analysis; returns stdout and the last expression's value. Use for ALL "
                "arithmetic over retrieved figures. Fresh sandbox each call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Self-contained Python, e.g. 'print(sum([132,129,143,159,154,153,177,200,219,287,376,473]))'.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]

# Dispatch table: tool name -> callable(args_dict) -> str. The plain impls coerce
# arg types defensively (XML params arrive as strings).
TOOL_IMPLS = {
    "search_documents": lambda a: _search_documents(a.get("query", ""), a.get("top_k", 6)),
    "grep_documents": lambda a: _grep_documents(
        a.get("pattern", ""), a.get("file_name", ""), a.get("year", ""), a.get("regex", False)
    ),
    "read_document": lambda a: _read_document(
        a.get("file_name", ""), a.get("start_line", 0), a.get("num_lines", 100)
    ),
    "list_documents": lambda a: _list_documents(a.get("year", "")),
    "compute": lambda a: _compute(a.get("code", "")),
}


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _load_parser(tok):
    """verl qwen3_coder ToolParser (+ tool schemas for type-coercion). Returns a
    callable text -> list[(name, args_dict)]; falls back to a regex if verl's
    internal API shifts, so a parser change can never silently zero tool use."""
    fn = None
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
        fn = parse
        print("[eval] tool parser: verl qwen3_coder", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] verl qwen3_coder unavailable ({type(e).__name__}: {e}); using regex fallback", flush=True)

    if fn is not None:
        return fn

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
        return tok.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False
        )
    except Exception:  # noqa: BLE001
        # Some chat templates reject an assistant message carrying BOTH content and
        # tool_calls. Blank such content and retry -> loses only per-turn reasoning
        # (the proven MATH-eval form), so a template quirk never fakes a 0% score.
        safe = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls") and m.get("content"):
                m = {**m, "content": ""}
            safe.append(m)
        return tok.apply_chat_template(
            safe, tools=TOOLS, add_generation_prompt=True, tokenize=False
        )


# The model's turn ends either at end-of-turn (<|im_end|>) or right after a tool
# call closes (</tool_call>); we include the stop string so we can parse it.
_TOOLCALL_OPEN_RE = re.compile(r"<tool_call>|<function=", re.IGNORECASE)


async def _complete(session, prompt: str):
    payload = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stop": ["<|im_end|>", "</tool_call>"],
        "include_stop_str_in_output": True,
    }
    ch = await _post(session, payload)
    return ch["text"], ch.get("finish_reason")


async def _post(session, payload):
    """POST /completions with retry on transient vLLM disconnects/timeouts."""
    last = None
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            async with session.post(f"{BASE_URL}/completions", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data["choices"][0]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:  # transient vLLM disconnect/timeout
            last = e
            if attempt < _HTTP_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise last


async def _forced_answer(session, base_prompt: str) -> str:
    """Final-turn commit: prefix-seed the opening tag so the model MUST emit a
    value from whatever it has gathered (natural answering on earlier turns is
    unchanged). Without this the base model runs out of budget still calling tools
    and emits no answer (smoke 620600576193281: 21/24 no-answer at the turn cap)."""
    ch = await _post(session, {
        "model": SERVED_MODEL, "prompt": base_prompt + "<FINAL_ANSWER>",
        "max_tokens": 96, "temperature": TEMPERATURE,
        "stop": ["</FINAL_ANSWER>", "<|im_end|>"],
    })
    return "<FINAL_ANSWER>" + (ch.get("text") or "").strip() + "</FINAL_ANSWER>"


async def _assistant_turn(session, base_prompt: str):
    """One assistant turn, transparently continuing across max_tokens truncations
    (finish_reason == 'length') so a long chain still reaches its tool call or
    <FINAL_ANSWER>. Uses the raw /completions route: a continuation re-sends
    prompt+partial (no new generation header), resuming the SAME assistant message.
    Returns (accumulated_text, n_continuations, still_truncated)."""
    acc = ""
    for cont in range(MAX_CONT + 1):
        text, fr = await _complete(session, base_prompt + acc)
        acc += text
        if fr != "length":
            return acc, cont, False
    return acc, MAX_CONT, True


def _split_reasoning(text: str) -> str:
    """The assistant's natural-language reasoning is the text BEFORE its first
    tool-call marker; keep it as the assistant message content so the model
    retains its own notes across turns (the structured tool_calls carry the call
    itself, so this avoids duplicating the XML)."""
    m = _TOOLCALL_OPEN_RE.search(text)
    return (text[: m.start()] if m else text).strip()


async def _run_one(session, tok, parse, sem, ex) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        tool_counts = {k: 0 for k in TOOL_IMPLS}
        n_tool, n_tool_err, turns = 0, 0, 0
        n_cont, truncated = 0, False
        assistant_texts = []
        steps: list = []   # structured trajectory when TRACE (reasoning + calls + outputs)
        for turn in range(MAX_TURNS):
            turns = turn + 1
            if turn == MAX_TURNS - 1:
                # out of budget: force a committed answer (prefix-seed the tag), no more tools
                messages.append({"role": "user", "content": FINAL_NUDGE})
                try:
                    forced = await _forced_answer(session, _render(tok, messages))
                except Exception as e:  # noqa: BLE001
                    forced = f"[error {type(e).__name__}: {e}]"
                assistant_texts.append(forced)
                if TRACE:
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
                if TRACE:
                    steps.append({"turn": turn, "reasoning": _split_reasoning(text),
                                  "tool_calls": [], "tool_results": []})
                break
            messages.append({
                "role": "assistant",
                "content": _split_reasoning(text),
                "tool_calls": [
                    # arguments as a DICT: the Qwen3.5 chat template iterates
                    # .arguments.items(), so a JSON string raises "Can only get
                    # item pairs from a mapping" (seen in the MATH smoke).
                    {"id": f"c{turn}_{i}", "type": "function",
                     "function": {"name": n, "arguments": a}}
                    for i, (n, a) in enumerate(calls)
                ],
            })
            step_results: list = []
            for i, (name, args) in enumerate(calls):
                impl = TOOL_IMPLS.get(name)
                if impl is None:
                    res = f"Error: unknown tool {name!r}. Available: {', '.join(TOOL_IMPLS)}."
                    n_tool_err += 1
                else:
                    try:
                        # Run the (synchronous) tool off the event loop so retrieval /
                        # file reads never freeze it and drop the vLLM connections.
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
                # Record the FULL actor-visible tool output (the exact string fed back
                # to the model below). The old [:3000] clip made the saved trajectory
                # lossy vs. what the actor saw, so the grounding judge graded evidence it
                # could not inspect. The tools already bound their own output, so this is
                # actor-visible-lossless, not unbounded.
                step_results.append({"name": name, "result": str(res)})
                messages.append({"role": "tool", "content": str(res),
                                 "tool_call_id": f"c{turn}_{i}", "name": name})
            if TRACE:
                steps.append({"turn": turn, "reasoning": _split_reasoning(text),
                              "tool_calls": [{"name": n, "args": a} for n, a in calls],
                              "tool_results": step_results})

        full_output = "\n".join(assistant_texts)
        pred = _extractor.extract(full_output)
        scores = {}
        for tol in TOLERANCES:
            if not pred:
                scores[tol] = 0.0
            else:
                try:
                    scores[tol] = float(score_answer(ex["gt"], pred, tol))
                except Exception:  # noqa: BLE001 - empty/malformed pred or gt -> wrong
                    scores[tol] = 0.0
        result = {
            "uid": ex["uid"], "difficulty": ex["difficulty"], "question": ex["question"],
            "gt": ex["gt"], "pred": pred, "correct": bool(scores[0.0] > 0),
            "scores": scores, "source_files": ex.get("source_files", ""),
            "n_tool": n_tool, "n_tool_err": n_tool_err, "tool_counts": tool_counts,
            "turns": turns, "n_cont": n_cont, "truncated": bool(truncated),
            "final_tail": full_output[-400:],
        }
        if TRACE:
            result["_trace"] = steps
        return result


def _load_officeqa():
    rows = []
    with open(EVAL_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            q = (r.get("question") or "").strip()
            gt = (r.get("answer") or "").strip()
            if not q or not gt:
                continue
            diff = (r.get("difficulty") or "").strip().lower()
            if EVAL_DIFFICULTY and diff != EVAL_DIFFICULTY:
                continue
            rows.append({
                "uid": (r.get("uid") or "").strip(), "question": q, "gt": gt,
                "difficulty": diff, "source_files": (r.get("source_files") or "").strip(),
            })
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


def _emit_mlflow_traces(results) -> None:
    """Best-effort: log each question as an MLflow trace (root AGENT span + per-turn
    CHAIN spans + per-call TOOL spans) so trajectories are inspectable in the MLflow
    UI and queryable via mlflow.search_traces. Emitted sequentially AFTER the
    concurrent gather so span nesting is unambiguous. Any MLflow API drift degrades
    to a warning -- the JSONL bundle is the reliable artifact."""
    try:
        import mlflow
    except Exception as e:  # noqa: BLE001
        print(f"[eval] MLflow unavailable ({e}); skipped trace emission (bundle still written)", flush=True)
        return
    exp = os.environ.get("MLFLOW_EXPERIMENT") or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    try:
        if exp:
            mlflow.set_experiment(exp)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] mlflow.set_experiment({exp!r}) failed: {e}", flush=True)
    n_ok = 0
    for r in results:
        try:
            with mlflow.start_span(name="officeqa_question", span_type="AGENT") as root:
                root.set_inputs({"uid": r.get("uid"), "question": r.get("question"),
                                 "gt": r.get("gt"), "source_files": r.get("source_files")})
                try:
                    mlflow.update_current_trace(tags={
                        "uid": str(r.get("uid")), "difficulty": str(r.get("difficulty")),
                        "correct": str(r.get("correct"))})
                except Exception:  # noqa: BLE001 (older MLflow: tags optional)
                    pass
                for step in r.get("_trace", []):
                    calls = step.get("tool_calls") or []
                    res = step.get("tool_results") or []
                    with mlflow.start_span(name=f"turn_{step.get('turn')}", span_type="CHAIN") as ts:
                        ts.set_inputs({"reasoning": (step.get("reasoning") or "")[:1000]})
                        for j, c in enumerate(calls):
                            out = res[j].get("result") if j < len(res) else ""
                            with mlflow.start_span(name=f"tool:{c.get('name')}", span_type="TOOL") as tsp:
                                a = c.get("args")
                                tsp.set_inputs(a if isinstance(a, dict) else {"raw": a})
                                tsp.set_outputs({"result": out})
                root.set_outputs({"pred": r.get("pred"), "correct": r.get("correct")})
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            if n_ok == 0:
                print(f"[eval] MLflow trace emission failed ({type(e).__name__}: {e}); continuing", flush=True)
    print(f"[eval] emitted {n_ok}/{len(results)} MLflow traces (experiment={exp})", flush=True)


async def _main_async():
    from collections import Counter

    print(f"[eval] OfficeQA agentic eval | model={SERVED_MODEL} url={BASE_URL} "
          f"turns<={MAX_TURNS} max_tok={MAX_TOKENS} temp={TEMPERATURE} conc={CONCURRENCY}", flush=True)
    print(f"[eval] csv={EVAL_CSV} difficulty={EVAL_DIFFICULTY or 'all'}", flush=True)
    tok = _load_tokenizer()
    parse = _load_parser(tok)
    rows = _load_officeqa()
    print(f"[eval] loaded {len(rows)} questions", flush=True)

    # Pre-build the BM25 index + load chunks NOW (synchronous, ~10-30s) BEFORE any
    # async HTTP. Doing it lazily inside the first tool call froze the event loop
    # mid-run and vLLM dropped the idle keep-alive connections -> ServerDisconnected
    # on every turn-2 request (smoke run 187346983850485: 0/24, all "Server
    # disconnected"). Warming it up front keeps the loop responsive during the run.
    tw = time.time()
    _oqt._load_chunks()
    _oqt._get_bm25()
    print(f"[eval] pre-warmed chunks+BM25 in {time.time()-tw:.0f}s "
          f"(bm25s={_oqt._HAS_BM25}, chunks={len(_oqt._CHUNKS or [])})", flush=True)

    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[_run_one(session, tok, parse, sem, r) for r in rows])
    dt = time.time() - t0

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    acc = n_correct / n if n else 0.0
    no_ans = sum(1 for r in results if not r["pred"])
    answered = n - no_ans
    acc_ans = n_correct / answered if answered else 0.0
    used_tool = sum(1 for r in results if r["n_tool"] > 0)
    mean_tool = sum(r["n_tool"] for r in results) / n if n else 0
    tool_err = sum(r["n_tool_err"] for r in results)
    n_trunc = sum(1 for r in results if r.get("truncated"))
    tot_cont = sum(r.get("n_cont", 0) for r in results)
    tool_totals = Counter()
    for r in results:
        for k, v in r["tool_counts"].items():
            tool_totals[k] += v

    # accuracy at each tolerance, overall + by difficulty
    def acc_at(rs, tol):
        return (sum(x["scores"][tol] for x in rs) / len(rs)) if rs else 0.0
    by_diff = {}
    for r in results:
        by_diff.setdefault(r["difficulty"] or "unknown", []).append(r)

    print(f"\n==================== OfficeQA (agentic) RESULT ====================", flush=True)
    print(f"model={SERVED_MODEL}  n={n}  wall={dt:.0f}s", flush=True)
    print("ACCURACY  " + "  ".join(f"tol{int(t*100)}%={acc_at(results, t):.4f}" for t in TOLERANCES)
          + f"   (exact correct={n_correct}/{n})", flush=True)
    print(f"answered={answered}/{n}  ACCURACY_AMONG_ANSWERED(exact)={acc_ans:.4f} ({n_correct}/{answered})", flush=True)
    print("by difficulty (exact / 1% / 5%):", flush=True)
    for diff in sorted(by_diff):
        rs = by_diff[diff]
        print(f"  {diff:>8s} ({len(rs):>3d}):  " + " / ".join(f"{acc_at(rs, t)*100:5.1f}%" for t in TOLERANCES), flush=True)
    print(f"agentic: used_tool={used_tool}/{n}  mean_tool_calls={mean_tool:.2f}  tool_errors={tool_err}", flush=True)
    print("tool usage: " + "  ".join(f"{k}={tool_totals[k]}" for k in TOOL_IMPLS), flush=True)
    print(f"no_final_answer={no_ans}  turn_truncated={n_trunc}  total_continuations={tot_cont}", flush=True)
    print("---- 4 sample trajectories (tail) ----", flush=True)
    for r in results[:4]:
        print(f"  [{r['difficulty']} {'OK' if r['correct'] else 'XX'}] gt={r['gt']!r} pred={r['pred']!r} "
              f"tools={r['n_tool']} turns={r['turns']}", flush=True)
        print(f"      tail={r['final_tail']!r}", flush=True)
    print("===================================================================", flush=True)

    # --- trajectory trace bundle (grounding judge input) + MLflow tracing --------
    if TRACE and TRACE_OUT:
        os.makedirs(os.path.dirname(TRACE_OUT) or ".", exist_ok=True)
        keep = ("uid", "question", "difficulty", "gt", "pred", "correct", "scores",
                "source_files", "n_tool", "turns", "truncated")
        with open(TRACE_OUT, "w") as fh:
            for r in results:
                rec = {k: r.get(k) for k in keep}
                rec["trajectory"] = r.get("_trace", [])
                fh.write(json.dumps(rec) + "\n")
        print(f"[eval] wrote {len(results)} trajectory traces -> {TRACE_OUT}", flush=True)
    if MLFLOW_TRACE:
        _emit_mlflow_traces(results)
    for r in results:
        r.pop("_trace", None)   # keep EVAL_OUT compact (traces live in the bundle / MLflow)

    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump({
                "model": SERVED_MODEL, "csv": EVAL_CSV, "difficulty_filter": EVAL_DIFFICULTY,
                "n": n, "accuracy_exact": acc, "accuracy_by_tolerance": {str(t): acc_at(results, t) for t in TOLERANCES},
                "answered": answered, "accuracy_among_answered_exact": acc_ans,
                "by_difficulty": {d: {"n": len(rs), **{f"tol{int(t*100)}": acc_at(rs, t) for t in TOLERANCES}}
                                  for d, rs in by_diff.items()},
                "mean_tool_calls": mean_tool, "used_tool": used_tool, "tool_totals": dict(tool_totals),
                "tool_errors": tool_err, "no_answer": no_ans, "turn_truncated": n_trunc,
                "total_continuations": tot_cont, "wall_s": dt, "results": results,
            }, fh, indent=2)
        print(f"[eval] wrote {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main_async())
