#!/usr/bin/env python3
"""Agentic MATH-500 eval — the held-out benchmark for the RL run.

Goal (user): show a REAL improvement on a well-known held-out benchmark, proving
the GRPO-trained model learned something meaningful. This scores a model on
MATH-500 (HuggingFaceH4/MATH-500 — 500 problems from the MATH *test* split, so it
is disjoint from our MATH *train* L3-5 data) using the SAME agentic setup we train
with: multi-turn, the model may call the `calculator` tool, and we grade the final
\\boxed{} answer by mathematical equivalence. Run it on the BASE model (baseline)
and on the RL checkpoint (after) with identical settings; the delta is the result.

FAITHFUL TO TRAINING (not vLLM's server tool-parsing, which is uncertain for
Qwen3.5 on vLLM 0.24):
  * prompts are rendered with the model's OWN tokenizer chat template, tools=[calc]
    (same template the rollout uses);
  * the model's raw output is parsed with verl's `qwen3_coder` ToolParser
    (Qwen3XMLToolParser) — the exact parser air/53/air/55 proved for this tokenizer;
  * tool calls run usecases/math/tool.evaluate (the same safe AST calculator);
  * the final answer is scored with judge_reward._math_equiv / _extract_pred_str
    (the same matcher as the training-time `acc`).

Serving: talks to an OpenAI-compatible vLLM endpoint (EVAL_BASE_URL, default
http://127.0.0.1:8000/v1) via the /completions (raw text) route, so we control the
prompt string exactly. engine/serve/serve_and_eval.sh brings the server up first.

Knobs (env): EVAL_BASE_URL, EVAL_MODEL (served name), EVAL_MAX_TURNS (4),
EVAL_MAX_TOKENS (1024/turn), EVAL_TEMPERATURE (0), EVAL_CONCURRENCY (32),
EVAL_LIMIT (0=all 500; >0 = smoke), EVAL_OUT (json results path), MODEL_PATH
(tokenizer), MATH500_ID (HuggingFaceH4/MATH-500).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

# reuse the EXACT training-time scorer + tool, from sibling script dirs
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from reward import _math_equiv, _last_boxed  # noqa: E402
from tool import evaluate as calc_evaluate  # noqa: E402

# Eval extraction: an EXPLICIT final answer only (\boxed{} preferred, then #### N).
# NO lenient last-number fallback -- that manufactured spurious preds from truncated
# reasoning in the first smoke (e.g. "\sqrt{9}" -> "9"). No answer => None => wrong.
_HASH_ANS_RE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")


def _extract_answer(text: str):
    b = _last_boxed(text)
    if b is not None:
        return b
    m = _HASH_ANS_RE.findall(text)
    return m[-1] if m else None

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SERVED_MODEL = os.environ.get("EVAL_MODEL", "eval")
MATH500_ID = os.environ.get("MATH500_ID", "HuggingFaceH4/MATH-500")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "math500").lower()  # math500 | aime
MAX_TURNS = int(os.environ.get("EVAL_MAX_TURNS", "4"))
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "1024"))
# A turn cut off at max_tokens (finish_reason=="length") is NOT done -- continue it
# up to this many times so a long chain-of-thought still reaches its \boxed{} answer
# instead of being scored as "no answer" (the dominant baseline failure mode).
MAX_CONT = int(os.environ.get("EVAL_MAX_CONT", "3"))
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))
OUT = os.environ.get("EVAL_OUT", "")
REQ_TIMEOUT = float(os.environ.get("EVAL_REQ_TIMEOUT", "600"))

# Same system prompt the MATH training data uses (usecases/math/prep_data.py).
SYSTEM_PROMPT = (
    "You are a careful competition-math problem solver. Reason step by step. "
    "Whenever you need to do arithmetic, call the `calculator` tool with a single "
    "arithmetic expression (e.g. {\"expression\": \"12 * 7 + 3\"}) instead of "
    "computing it in your head, and use its result. You may call the tool several "
    "times. When you are confident, stop calling tools and give your final answer "
    "on its own line in the exact form \\boxed{<answer>} (put ONLY the final answer "
    "inside the box, e.g. \\boxed{\\frac{1}{2}} or \\boxed{24})."
)

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the exact numeric result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "One arithmetic expression to evaluate, e.g. '3 * (17 + 4) / 2'.",
                }
            },
            "required": ["expression"],
        },
    },
}


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _load_parser(tok):
    """verl qwen3_coder ToolParser (+ tool schema for type-coercion). Returns a
    callable text -> list[(name, args_dict)]; falls back to a regex if verl's
    internal API shifts, so a parser change can never silently zero tool use."""
    fn = None
    try:
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema
        try:
            schemas = [OpenAIFunctionToolSchema.model_validate(CALC_TOOL)]
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
                        args = {"expression": args}
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
    return tok.apply_chat_template(
        messages, tools=[CALC_TOOL], add_generation_prompt=True, tokenize=False
    )


async def _complete(session, prompt: str) -> str:
    payload = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        # stop at the end of the assistant turn so we can inspect/parse it, and at a
        # tool-call close so the model hands control back to run the tool.
        "stop": ["<|im_end|>", "</tool_call>"],
        "include_stop_str_in_output": True,
    }
    async with session.post(f"{BASE_URL}/completions", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    ch = data["choices"][0]
    return ch["text"], ch.get("finish_reason")


async def _assistant_turn(session, base_prompt: str):
    """One assistant turn, transparently continuing across max_tokens truncations
    (finish_reason == 'length') so a long chain-of-thought still reaches its
    \\boxed{} answer or tool call instead of being scored as 'no answer'. Uses the
    raw /completions route: a continuation just re-sends prompt+partial (no new
    generation header), so it resumes the SAME assistant message seamlessly.
    Returns (accumulated_text, n_continuations, still_truncated)."""
    acc = ""
    for cont in range(MAX_CONT + 1):
        text, fr = await _complete(session, base_prompt + acc)
        acc += text
        if fr != "length":
            return acc, cont, False
    return acc, MAX_CONT, True


async def _run_one(session, tok, parse, sem, ex) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
        ]
        n_tool, n_tool_err, turns, final_text = 0, 0, 0, ""
        n_cont, truncated = 0, False
        for turn in range(MAX_TURNS):
            turns = turn + 1
            try:
                text, c, tr = await _assistant_turn(session, _render(tok, messages))
            except Exception as e:  # noqa: BLE001
                final_text = f"[error {type(e).__name__}: {e}]"
                break
            n_cont += c
            truncated = tr
            final_text = text
            calls = parse(text)
            if not calls:
                break
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [
                    # arguments as a DICT: the Qwen3.5 chat template iterates
                    # .arguments.items(), so a JSON string raises "Can only get
                    # item pairs from a mapping" (first smoke, air/58 run 210783425286225).
                    {"id": f"c{turn}_{i}", "type": "function",
                     "function": {"name": n, "arguments": a}}
                    for i, (n, a) in enumerate(calls)
                ],
            })
            for i, (name, args) in enumerate(calls):
                if name == "calculator":
                    res = calc_evaluate(str(args.get("expression", "")))
                    if res.startswith("Error:"):
                        n_tool_err += 1
                else:
                    res = f"Error: unknown tool {name!r}"
                    n_tool_err += 1
                n_tool += 1
                messages.append({"role": "tool", "content": res,
                                 "tool_call_id": f"c{turn}_{i}", "name": name})
        pred = _extract_answer(final_text)
        correct = _math_equiv(pred, ex["gt"])
        return {
            "idx": ex["idx"], "level": ex["level"], "type": ex["type"],
            "gt": ex["gt"], "pred": pred, "correct": bool(correct),
            "n_tool": n_tool, "n_tool_err": n_tool_err, "turns": turns,
            "n_cont": n_cont, "truncated": bool(truncated),
            "final_tail": final_text[-300:],
        }


def _load_math500():
    import datasets
    ds = datasets.load_dataset(MATH500_ID, split="test")
    rows = []
    for i, r in enumerate(ds):
        problem = r.get("problem") or r.get("question") or ""
        ans = r.get("answer")
        gt = ans if (ans and "\\boxed" not in str(ans)) else _last_boxed(r.get("solution") or "")
        if not problem or not gt:
            continue
        lvl = r.get("level")
        lvl = int("".join(c for c in str(lvl) if c.isdigit()) or -1)
        rows.append({"idx": i, "problem": problem, "gt": str(gt).strip(),
                     "level": lvl, "type": r.get("subject") or r.get("type") or ""})
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


# --- AIME (harder held-out benchmark; base MATH-500 was saturated at ~95% among-
# answered). Answers are integers 0-999 -> robust scoring via _math_equiv's numeric
# fast-path. Column names vary across mirrors, so detect generically and try several
# 2025 mirrors so one unavailable repo can't zero the run (schemas confirmed 2026-09-12).
_AIME_2024 = [("Maxwell-Jia/AIME_2024", None, "train")]
_AIME_2025 = [("math-ai/aime25", None, "test"),
              ("MathArena/aime_2025", None, "train"),
              ("yentinglin/aime_2025", None, "train"),
              ("opencompass/AIME2025", None, "test")]


def _first_present(r, keys):
    for k in keys:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return None


def _load_aime_group(cands, year, start_idx):
    import datasets
    last = None
    for name, cfg, split in cands:
        try:
            ds = datasets.load_dataset(name, cfg, split=split) if cfg else datasets.load_dataset(name, split=split)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        rows = []
        for i, r in enumerate(ds):
            problem = _first_present(r, ["problem", "Problem", "question", "Question"]) or ""
            a = _first_present(r, ["answer", "Answer"])   # integer 0-999; NOT "solution" (worked text)
            if not problem or a is None:
                continue
            try:
                gt = str(int(str(a).strip()))      # AIME answers are integers 0-999
            except Exception:  # noqa: BLE001
                gt = str(a).strip()
            rows.append({"idx": start_idx + i, "problem": problem, "gt": gt,
                         "level": year, "type": f"aime{year}"})
        print(f"[eval] AIME {year}: loaded {len(rows)} from {name}", flush=True)
        return rows
    print(f"[eval] AIME {year}: ALL sources failed ({type(last).__name__}: {last})", flush=True)
    return []


def _load_aime():
    rows = _load_aime_group(_AIME_2024, 2024, 0)
    rows += _load_aime_group(_AIME_2025, 2025, 10000)   # disjoint idx so 2024/2025 never collide
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


# --- MathArena 2026 (UNCONTAMINATED held-out: competitions released AFTER the model's
# training cutoff -> genuinely unseen, unlike AIME 2024/2025 which a 2026 model has
# ingested). MathArena schema: problem (LaTeX str), answer (int64 or str), problem_idx.
# AIME 2026 (30) + HMMT Feb 2026 (33, harder) = 63 problems, integer/short answers.
_MATHARENA_2026 = [
    ("MathArena/aime_2026", "aime_2026"),
    ("MathArena/hmmt_feb_2026", "hmmt_feb_2026"),
]


def _load_matharena_one(name, tag, start_idx):
    import datasets
    ds = None
    for split in ("train", "test"):
        try:
            ds = datasets.load_dataset(name, split=split)
            break
        except Exception:  # noqa: BLE001
            continue
    if ds is None:
        print(f"[eval] MathArena {tag}: FAILED to load {name}", flush=True)
        return []
    rows = []
    for i, r in enumerate(ds):
        problem = r.get("problem") or r.get("question") or ""
        a = r.get("answer")
        if not problem or a is None:
            continue
        try:
            gt = str(int(str(a).strip()))
        except Exception:  # noqa: BLE001
            gt = str(a).strip()          # HMMT answers can be non-integer -> _math_equiv handles it
        rows.append({"idx": start_idx + i, "problem": problem, "gt": gt,
                     "level": tag, "type": tag})
    print(f"[eval] MathArena {tag}: loaded {len(rows)} from {name}", flush=True)
    return rows


def _load_matharena2026():
    rows = []
    for k, (name, tag) in enumerate(_MATHARENA_2026):
        rows += _load_matharena_one(name, tag, k * 100000)   # disjoint idx per competition
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


def _load_dataset():
    if EVAL_DATASET in ("matharena2026", "aime2026", "hmmt2026"):
        return _load_matharena2026()
    if EVAL_DATASET == "aime":
        return _load_aime()
    return _load_math500()


async def _main_async():
    import aiohttp
    from collections import Counter, defaultdict

    print(f"[eval] {EVAL_DATASET} agentic eval | model={SERVED_MODEL} url={BASE_URL} "
          f"turns<={MAX_TURNS} temp={TEMPERATURE} conc={CONCURRENCY}", flush=True)
    tok = _load_tokenizer()
    parse = _load_parser(tok)
    rows = _load_dataset()
    print(f"[eval] dataset={EVAL_DATASET}  loaded {len(rows)} problems", flush=True)

    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[_run_one(session, tok, parse, sem, r) for r in rows])
    dt = time.time() - t0

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    acc = n_correct / n if n else 0.0
    by_lvl_tot, by_lvl_ok = Counter(), Counter()
    for r in results:
        by_lvl_tot[r["level"]] += 1
        by_lvl_ok[r["level"]] += int(r["correct"])
    mean_tool = sum(r["n_tool"] for r in results) / n if n else 0
    tool_err = sum(r["n_tool_err"] for r in results)
    no_ans = sum(1 for r in results if r["pred"] is None)
    used_tool = sum(1 for r in results if r["n_tool"] > 0)
    answered = n - no_ans
    acc_ans = n_correct / answered if answered else 0.0   # controls for answer-rate: math ability among problems the model actually answered
    n_trunc = sum(1 for r in results if r.get("truncated"))  # STILL truncated after MAX_CONT continuations
    tot_cont = sum(r.get("n_cont", 0) for r in results)

    print(f"\n==================== {EVAL_DATASET} (agentic) RESULT ====================", flush=True)
    print(f"model={SERVED_MODEL}  n={n}  ACCURACY={acc:.4f} ({n_correct}/{n})  wall={dt:.0f}s", flush=True)
    print(f"answered={answered}/{n}  ACCURACY_AMONG_ANSWERED={acc_ans:.4f} ({n_correct}/{answered})", flush=True)
    print("by level: " + "  ".join(
        f"L{lvl}={by_lvl_ok[lvl]}/{by_lvl_tot[lvl]}({by_lvl_ok[lvl]/by_lvl_tot[lvl]:.2f})"
        for lvl in sorted(by_lvl_tot)), flush=True)
    print(f"agentic: used_tool={used_tool}/{n}  mean_tool_calls={mean_tool:.2f}  "
          f"tool_errors={tool_err}  no_boxed_answer={no_ans}  "
          f"turn_truncated={n_trunc}  total_continuations={tot_cont}", flush=True)
    print("---- 3 sample trajectories (tail) ----", flush=True)
    for r in results[:3]:
        print(f"  [L{r['level']} {'OK' if r['correct'] else 'XX'}] gt={r['gt']!r} pred={r['pred']!r} "
              f"tools={r['n_tool']} turns={r['turns']}  tail={r['final_tail']!r}", flush=True)
    print("===================================================================", flush=True)

    if OUT:
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump({"model": SERVED_MODEL, "n": n, "accuracy": acc,
                       "answered": answered, "accuracy_among_answered": acc_ans,
                       "by_level": {str(k): [by_lvl_ok[k], by_lvl_tot[k]] for k in by_lvl_tot},
                       "mean_tool_calls": mean_tool, "used_tool": used_tool,
                       "no_answer": no_ans, "turn_truncated": n_trunc,
                       "total_continuations": tot_cont, "wall_s": dt, "results": results}, fh, indent=2)
        print(f"[eval] wrote {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main_async())
