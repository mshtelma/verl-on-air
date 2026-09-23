#!/usr/bin/env python3
"""Agentic MATH-500 eval — the held-out benchmark for the RL run.

Goal (user): show a REAL improvement on a well-known held-out benchmark, proving
the GRPO-trained model learned something meaningful. This scores a model on
MATH-500 (HuggingFaceH4/MATH-500 — 500 problems from the MATH *test* split, so it
is disjoint from our MATH *train* L3-5 data) using the SAME agentic setup we train
with: multi-turn, the model may call the `calculator` tool, and we grade the final
\\boxed{} answer by mathematical equivalence. Run it on the BASE model (baseline)
and on the RL checkpoint (after) with identical settings; the delta is the result.

Shared with training, exactly (not vLLM's server-side tool parsing):
  * the training SYSTEM_PROMPT (prep_data.py) and the calculator schema TRAINING renders --
    read from verl's @function_tool registry, not copied -- in the model's own chat template;
  * verl's own tool parser for TOOL_FORMAT through its public extract_tool_calls (no fallback:
    without verl the eval refuses to start), and the reasoning before a tool call stays in the
    assistant message, as the model's own tokens stay in training's context;
  * tool calls run usecases/math/tool.evaluate (the same safe AST calculator);
  * the final answer is extracted from ALL of the model's turns and graded by grading.py
    (verl's prime_math) -- the same extractor and grader as the training rule score (`acc`).
    The judge that trains the model is a surrogate objective; this deterministic grade is the
    independent target.

Its own policy, recorded as a versioned `eval_policy` in every artifact: the chat is re-rendered
each turn (training continues raw tokens); each request is capped at EVAL_MAX_TOKENS with
EVAL_MAX_CONT continuations and stops at </tool_call> (training has one episode-wide budget); and
EVAL_MAX_TURNS (8 in both eval jobs) is larger than training's MAX_TURNS (4) -- the base model
needed the room to reach a box. Compare only artifacts with equal policies.

Serving: talks to an OpenAI-compatible vLLM endpoint (EVAL_BASE_URL, default
http://127.0.0.1:8000/v1) via the /completions (raw text) route, so we control the
prompt string exactly. engine/serve/serve_and_eval.sh brings the server up first.

Follows engine/serve/eval_contract.py: the served model must be listed before any problem
runs; an inference failure is recorded as infrastructure (never as a wrong answer); the run
is valid only with the expected problem count (EVAL_EXPECT_N) and no infrastructure errors
beyond EVAL_MAX_INFRA_ERRORS; artifacts are atomic, never overwritten, and carry the served
model's identity; each finished problem is written under <EVAL_OUT>.parts/.

Knobs (env): EVAL_BASE_URL, EVAL_MODEL (served name), EVAL_MAX_TURNS (4),
EVAL_MAX_TOKENS (1024/turn), EVAL_TEMPERATURE (0), EVAL_CONCURRENCY (32),
EVAL_LIMIT (0=all 500; >0 = smoke), EVAL_OUT (json results path), MODEL_PATH
(tokenizer), MATH500_ID (HuggingFaceH4/MATH-500) + MATH500_REVISION.

Every eval set is read at a pinned commit (engine/lib/data_manifest.py), recorded in the
artifact's dataset header. AIME 2025 lists mirrors: they are tried only with
ALLOW_FALLBACK_SOURCE=1, and accepted only if their 30 answers are the pinned set's. A set that
cannot be loaded as pinned stops the eval (exit 2) before any problem runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# reuse the EXACT training-time scorer + tool, from sibling script dirs
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import grading  # noqa: E402  (the one answer extractor + grader, shared with reward.py)
from tool import evaluate as calc_evaluate  # noqa: E402
from prep_data import SYSTEM_PROMPT  # noqa: E402  (the training prompt, not a copy)

sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "engine", "serve"))
import eval_contract as ec  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "engine", "lib"))
import data_manifest as dm  # noqa: E402

_DATASET_META: dict = {}   # filled by the loaders: what exactly was evaluated


MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SERVED_MODEL = os.environ.get("EVAL_MODEL", "eval")
MATH500_ID = os.environ.get("MATH500_ID", "HuggingFaceH4/MATH-500")
# The pin applies to the default set; pointing MATH500_ID elsewhere requires pinning that one too.
MATH500_REVISION = os.environ.get("MATH500_REVISION", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
                                  if MATH500_ID == "HuggingFaceH4/MATH-500" else "")
EVAL_DATASET = os.environ.get("EVAL_DATASET", "math500").lower()  # math500 | aime
MAX_TURNS = int(os.environ.get("EVAL_MAX_TURNS", "8"))
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
TOOL_FORMAT = os.environ.get("TOOL_FORMAT", "qwen3_coder")      # the training job's parser
EVAL_POLICY_VERSION = 2   # bump whenever a change alters what the eval measures
STOP = ["<|im_end|>", "</tool_call>"]
TOOL_NAMES = ["calculator"]
TOOLS: list[dict] = []   # set by _main_async from _tool_schemas() before any problem

def _tool_schemas() -> list[dict]:
    """The schemas training renders: verl's @function_tool registry (filled when tool.py was
    imported), dumped exactly as ToolAgentLoop dumps them. Raises if verl is not importable."""
    from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY
    missing = [n for n in TOOL_NAMES if n not in FUNCTION_TOOL_REGISTRY]
    if missing:
        raise RuntimeError(f"tools {missing} are not in verl's registry (tool.py imported without verl?)")
    return [FUNCTION_TOOL_REGISTRY[n].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for n in TOOL_NAMES]


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _load_parser(tok, schemas: list[dict]):
    """verl's own parser for TOOL_FORMAT, called as training calls it: extract_tool_calls(ids, tools).
    -> async parse(text) -> (content before the calls, [(name, args)], calls it could not parse)."""
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema
    parser = ToolParser.get_tool_parser(TOOL_FORMAT, tok)
    tools = [OpenAIFunctionToolSchema.model_validate(t) for t in schemas]

    async def parse(text: str):
        content, calls = await parser.extract_tool_calls(tok.encode(text, add_special_tokens=False), tools)
        out = [(c.name, json.loads(c.arguments) if c.arguments else {}) for c in calls]
        return content, out, max(0, text.count("<tool_call>") - len(out))
    return parse


def _render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)


async def _complete(session, prompt: str) -> str:
    payload = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        # stop at the end of the assistant turn so we can inspect/parse it, and at a
        # tool-call close so the model hands control back to run the tool.
        "stop": STOP,
        "include_stop_str_in_output": True,
    }
    data = await ec.post_json(session, f"{BASE_URL}/completions", payload)  # raises ec.InfraError
    try:
        ch = data["choices"][0]
        return ch["text"], ch.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        raise ec.InfraError("infra_inference", f"malformed completion: {str(data)[:200]}") from None


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


async def _run_one(session, tok, parse, sem, ex, parts) -> dict:
    async with sem:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
        ]
        n_tool, n_tool_err, n_parse_err, turns, final_text = 0, 0, 0, 0, ""
        assistant_texts: list[str] = []
        n_cont, truncated = 0, False
        status, infra_detail = "scored", ""
        try:
            for turn in range(MAX_TURNS):
                turns = turn + 1
                text, c, tr = await _assistant_turn(session, _render(tok, messages))
                n_cont += c
                truncated = tr
                final_text = text
                assistant_texts.append(text)
                content, calls, bad = await parse(text)
                n_parse_err += bad          # a malformed call is the model's: like training, not run
                if not calls:
                    break
                messages.append({
                    # the reasoning before the call stays: training keeps the model's own tokens
                    "role": "assistant", "content": content.strip(),
                    "tool_calls": [
                        # arguments as a DICT: the Qwen3.5 chat template iterates
                        # .arguments.items(), so a JSON string raises "Can only get
                        # item pairs from a mapping" (seen on the first harness smoke).
                        {"id": f"c{turn}_{i}", "type": "function",
                         "function": {"name": n, "arguments": a}}
                        for i, (n, a) in enumerate(calls)
                    ],
                })
                for i, (name, args) in enumerate(calls):
                    if name == "calculator":
                        res = calc_evaluate(str(args.get("expression", "")))
                        if res.startswith("Error:"):
                            n_tool_err += 1   # a bad expression is the model's doing
                    else:
                        res = f"Error: unknown tool {name!r}"
                        n_tool_err += 1
                    n_tool += 1
                    messages.append({"role": "tool", "content": res,
                                     "tool_call_id": f"c{turn}_{i}", "name": name})
        except ec.InfraError as e:
            status, infra_detail = e.kind, e.detail
        except Exception as e:  # noqa: BLE001 - never let the harness score its own bug
            status, infra_detail = "infra_harness", f"{type(e).__name__}: {e}"
        # Every turn the model wrote -- the text the training rule score reads -- so the last
        # \boxed{} of the episode is its answer, whichever turn it is in.
        pred = grading.extract_answer("\n".join(assistant_texts)) if status == "scored" else None
        correct = status == "scored" and await asyncio.get_running_loop().run_in_executor(
            None, grading.grade, pred, ex["gt"])   # sympy can take seconds: off the event loop
        rec = {
            "idx": ex["idx"], "level": ex["level"], "type": ex["type"],
            "gt": ex["gt"], "status": status, "infra_detail": infra_detail[:400],
            "pred": pred, "correct": bool(correct),
            "n_tool": n_tool, "n_tool_err": n_tool_err, "n_parse_err": n_parse_err, "turns": turns,
            "n_cont": n_cont, "truncated": bool(truncated),
            "final_tail": final_text[-300:],
        }
        parts.write(ex["idx"], rec)
        return rec


def _load_math500():
    src = dm.Source(MATH500_ID, MATH500_REVISION)
    ds, record = dm.load_verified([(src, lambda s: s.load(split="test"))], lambda _ds: None)
    _DATASET_META.update(**record, split="test", fingerprint=getattr(ds, "_fingerprint", None),
                         n_rows=len(ds), limit=LIMIT)
    rows = []
    for i, r in enumerate(ds):
        problem = r.get("problem") or r.get("question") or ""
        ans = r.get("answer")
        gt = ans if (ans and "\\boxed" not in str(ans)) else grading.last_boxed(r.get("solution") or "")
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
# answered). Answers are integers 0-999, graded like every other answer (grading.grade). Column names vary across mirrors, so detect generically. The first entry of each
# list is the pinned source; the rest are mirrors, used only with ALLOW_FALLBACK_SOURCE=1 and
# only if their 30 answers are the pinned set's (_AIME_ANSWERS; checked 2026-09-23: the three
# 2025 entries agree. opencompass/AIME2025 needs a per-exam config and never loaded, so it is gone).
_AIME_2024 = [(dm.Source("Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792"), "train")]
_AIME_2025 = [(dm.Source("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b"), "test"),
              (dm.Source("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302"), "train"),
              (dm.Source("yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f"), "train")]
_AIME_ANSWERS = {   # order-independent digest of the 30 answers (data_manifest.content_digest)
    2024: "765d8646ba59127b6fae8c07fdcb9c165f015402719e22b1687a49c4f95c9944",
    2025: "0b3ae5d06d7ed01a120b257cb405dd70f1b0bbac700fd65709c966d90e976496",
}


def _first_present(r, keys):
    for k in keys:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return None


def _aime_rows(ds, year, start_idx):
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
    return rows


def _load_aime_group(cands, year, start_idx):
    def verify(rows):
        got = dm.content_digest(rows, ["gt"])
        ok = len(rows) == 30 and got == _AIME_ANSWERS[year]
        return None if ok else f"{len(rows)} problems, answers digest {got[:12]} (want 30, {_AIME_ANSWERS[year][:12]})"

    rows, record = dm.load_verified(
        [(src, lambda s, split=split: _aime_rows(s.load(split=split), year, start_idx)) for src, split in cands],
        verify)
    _DATASET_META.setdefault("sources", []).append({**record, "year": year})
    print(f"[eval] AIME {year}: loaded {len(rows)} from {record['hf_id']}@{record['revision'][:12]}", flush=True)
    return rows


def _load_aime():
    rows = _load_aime_group(_AIME_2024, 2024, 0)
    rows += _load_aime_group(_AIME_2025, 2025, 10000)   # disjoint idx so 2024/2025 never collide
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


# --- MathArena 2026 (UNCONTAMINATED held-out: competitions released AFTER the model's
# training cutoff -> genuinely unseen, unlike AIME 2024/2025 which a 2026 model has
# ingested). MathArena schema: problem (LaTeX str), answer (int64 or str), problem_idx.
# AIME 2026 (30) + HMMT Feb 2026 (33, harder) = 63 problems, integer/short answers. Each repo
# has one split, "train".
_MATHARENA_2026 = [
    (dm.Source("MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3"), "aime_2026"),
    (dm.Source("MathArena/hmmt_feb_2026", "02fba4f74d8e68e73e66a02d540fd979c05c274c"), "hmmt_feb_2026"),
]


def _load_matharena_one(src, tag, start_idx):
    ds, record = dm.load_verified([(src, lambda s: s.load(split="train"))], lambda _ds: None)
    _DATASET_META.setdefault("sources", []).append({**record, "split": "train", "tag": tag})
    rows = []
    for i, r in enumerate(ds):
        problem = r.get("problem") or r.get("question") or ""
        a = r.get("answer")
        if not problem or a is None:
            continue
        try:
            gt = str(int(str(a).strip()))
        except Exception:  # noqa: BLE001
            gt = str(a).strip()          # HMMT answers can be non-integer -> grading.grade handles it
        rows.append({"idx": start_idx + i, "problem": problem, "gt": gt,
                     "level": tag, "type": tag})
    print(f"[eval] MathArena {tag}: loaded {len(rows)} from {src.label}", flush=True)
    return rows


def _load_matharena2026():
    rows = []
    for k, (src, tag) in enumerate(_MATHARENA_2026):
        rows += _load_matharena_one(src, tag, k * 100000)   # disjoint idx per competition
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


def _load_dataset():
    if EVAL_DATASET in ("matharena2026", "aime2026", "hmmt2026"):
        return _load_matharena2026()
    if EVAL_DATASET == "aime":
        return _load_aime()
    return _load_math500()


def _policy() -> dict:
    """The eval policy recorded in every artifact -- part of what a number means."""
    import hashlib
    return {
        "version": EVAL_POLICY_VERSION, "dataset": EVAL_DATASET,
        "loop": "chat re-rendered each turn (ToolAgentLoop continues raw tokens)",
        "prompt": "prep_data.SYSTEM_PROMPT + the problem", "tool_format": TOOL_FORMAT,
        "parser": "verl ToolParser.extract_tool_calls", "tools": TOOL_NAMES,
        "tool_schema_sha256": hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "max_turns": MAX_TURNS, "max_tokens_per_request": MAX_TOKENS, "max_continuations": MAX_CONT,
        "stop": STOP, "tool_calls_per_turn": 1, "temperature": TEMPERATURE,
        "answer": "grading.extract_answer over all assistant turns (last \\boxed{}, else an explicit "
                  "####; never a bare number), graded by grading.grade (verl prime_math.grade_answer: "
                  "exact after normalization + sympy)"}


async def _main_async() -> int:
    import aiohttp
    from collections import Counter

    print(f"[eval] {EVAL_DATASET} agentic eval | model={SERVED_MODEL} url={BASE_URL} "
          f"turns<={MAX_TURNS} temp={TEMPERATURE} conc={CONCURRENCY}", flush=True)
    ec.refuse_overwrite(OUT)
    started = time.time()
    tok = _load_tokenizer()
    global TOOLS
    try:
        TOOLS = _tool_schemas()
        parse = _load_parser(tok, TOOLS)
    except Exception as e:  # noqa: BLE001 - the training tools/parser, or no eval at all
        ec.fatal_not_ready("verl's tool schemas and parser", e)
    try:
        rows = _load_dataset()
    except dm.SourceError as e:
        print(f"[eval] FATAL: the {EVAL_DATASET} set could not be loaded as pinned: {e}", flush=True)
        return 2
    n_expected = ec._env_int("EVAL_EXPECT_N", LIMIT if LIMIT > 0 else None)
    print(f"[eval] dataset={EVAL_DATASET}  loaded {len(rows)} problems (expected {n_expected})", flush=True)

    timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            await ec.check_served_model(session, BASE_URL, SERVED_MODEL)
        except ec.InfraError as e:
            ec.fatal_not_ready("the served model", e)
        t0 = time.time()
        sem = asyncio.Semaphore(CONCURRENCY)
        parts = ec.PartsWriter(OUT)
        results = await asyncio.gather(*[_run_one(session, tok, parse, sem, r, parts) for r in rows])
    dt = time.time() - t0

    v = ec.verdict(results, n_loaded=len(rows), n_expected=n_expected)
    scored = [r for r in results if r["status"] == "scored"]
    n = len(scored)
    n_correct = sum(r["correct"] for r in scored)
    acc = n_correct / n if n else 0.0
    by_lvl_tot, by_lvl_ok = Counter(), Counter()
    for r in scored:
        by_lvl_tot[r["level"]] += 1
        by_lvl_ok[r["level"]] += int(r["correct"])
    mean_tool = sum(r["n_tool"] for r in scored) / n if n else 0
    tool_err = sum(r["n_tool_err"] for r in scored)
    no_ans = sum(1 for r in scored if r["pred"] is None)
    used_tool = sum(1 for r in scored if r["n_tool"] > 0)
    answered = n - no_ans
    acc_ans = n_correct / answered if answered else 0.0   # controls for answer-rate: math ability among problems the model actually answered
    n_trunc = sum(1 for r in scored if r.get("truncated"))  # STILL truncated after MAX_CONT continuations
    tot_cont = sum(r.get("n_cont", 0) for r in scored)

    print(f"\n==================== {EVAL_DATASET} (agentic) RESULT ====================", flush=True)
    print(f"model={SERVED_MODEL}  scored={n}/{len(rows)}  valid={v['valid']}  "
          f"ACCURACY={acc:.4f} ({n_correct}/{n})  wall={dt:.0f}s", flush=True)
    print(f"answered={answered}/{n}  ACCURACY_AMONG_ANSWERED={acc_ans:.4f} ({n_correct}/{answered})  "
          f"infra={v['infra_errors']}", flush=True)
    print("by level: " + "  ".join(
        f"L{lvl}={by_lvl_ok[lvl]}/{by_lvl_tot[lvl]}({by_lvl_ok[lvl]/by_lvl_tot[lvl]:.2f})"
        for lvl in sorted(by_lvl_tot, key=str)), flush=True)
    print(f"agentic: used_tool={used_tool}/{n}  mean_tool_calls={mean_tool:.2f}  "
          f"tool_errors={tool_err}  no_boxed_answer={no_ans}  "
          f"turn_truncated={n_trunc}  total_continuations={tot_cont}", flush=True)
    print("===================================================================", flush=True)

    if OUT:
        ec.write_json_atomic(OUT, {
            **ec.header(dataset=dict(_DATASET_META) or {"name": EVAL_DATASET},
                        question_ids=[r["idx"] for r in rows], policy=_policy(), started_at=started),
            **v,
            "model": SERVED_MODEL, "n": n, "accuracy": acc,
            "answered": answered, "accuracy_among_answered": acc_ans,
            "by_level": {str(k): [by_lvl_ok[k], by_lvl_tot[k]] for k in by_lvl_tot},
            "mean_tool_calls": mean_tool, "used_tool": used_tool,
            "no_answer": no_ans, "turn_truncated": n_trunc,
            "total_continuations": tot_cont, "wall_s": dt, "results": results})
        print(f"[eval] wrote {OUT}", flush=True)
    return ec.report_and_exit_code(v)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main_async()))
