#!/usr/bin/env python3
"""GATE: does TOOL_FORMAT's parser read the tool calls this model's own chat template writes?

A training run logged 120x `Failed to decode tool call: Expecting value: line 2
column 1 (char 1)` and never executed the calculator: TOOL_FORMAT=hermes ran
json.loads() on what the model actually emitted (Qwen XML). This probe answers it
cheaply (tokenizer only -- NO model weights, NO GPU compute, seconds on 1xA10):

  (1) How does the model's OWN chat template serialize an assistant tool call --
      hermes JSON ({"name": ..., "arguments": {...}}) or Qwen XML
      (<function=name><parameter=key>value</parameter></function>)?
  (2) Does verl's parser for PROBE_TOOL_FORMAT (default qwen3_coder -- set it to the
      TOOL_FORMAT you will train with) extract exactly that call -- name=calculator,
      arguments={"expression": "18 - 3 - 4"} -- through verl's PUBLIC parser API
      (ToolParser.get_tool_parser(fmt, tok) + `await extract_tool_calls(ids, tools)`),
      the call the rollout itself makes?

PASS only when (2) holds for the TEMPLATE-RENDERED sample. A canonical hand-written XML
sample and the other parser (hermes <-> qwen3_coder) are parsed too, but only reported:
they explain a failure, they never supply a pass. No render, a parse error, a different
call, or any other surprise is a FAIL / INCONCLUSIVE with a non-zero exit, and the last
log line is a machine-readable `PROBE_VERDICT {...}` (probe_verdict.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_verdict  # noqa: E402

# verl parsers report decode failures via logging -- surface them.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
TOOL_FORMAT = os.environ.get("PROBE_TOOL_FORMAT", os.environ.get("TOOL_FORMAT", "qwen3_coder"))

# The calculator tool schema, matching usecases/math/tool.py (name + single
# string param `expression`); this is what the rollout injects via the template.
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
EXPECTED = ("calculator", {"expression": "18 - 3 - 4"})

SYS = "You are a careful math solver. Use the calculator tool for arithmetic."
USER = "Janet has 18 eggs, eats 3, bakes 4. How many are left?"
ASSISTANT_TOOLCALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "18 - 3 - 4"}}}
    ],
}

# A canonical Qwen XML tool call (template-independent): reported only, to tell a parser
# problem from a template problem when the gate fails.
XML_SAMPLE = (
    "<tool_call>\n<function=calculator>\n<parameter=expression>\n18 - 3 - 4\n"
    "</parameter>\n</function>\n</tool_call>"
)


def classify(text: str | None) -> str:
    if not text:
        return "UNKNOWN"
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    inner = m.group(1) if m else text
    if "<function=" in inner:
        return "QWEN_XML"
    if re.search(r'\{\s*"name"', inner):
        return "HERMES_JSON"
    return "OTHER"


def _norm(calls) -> list[tuple[str, dict]]:
    """[(name, arguments dict)] from verl FunctionCalls (arguments is a JSON string)."""
    out = []
    for c in calls:
        try:
            args = json.loads(c.arguments) if c.arguments else {}
        except (TypeError, ValueError):
            args = {"<unparseable>": c.arguments}
        out.append((c.name, args if isinstance(args, dict) else {"<non-object>": args}))
    return out


def decide(tool_format: str, detected: str, template_calls: list | None,
           template_error: str | None = None) -> tuple[bool, str, list[str]]:
    """The gate. template_calls = what `tool_format`'s parser extracted from the TEMPLATE-rendered
    sample (None = there was no rendered sample)."""
    if template_calls is None:
        return False, "INCONCLUSIVE", ["the chat template did not render an assistant tool call"]
    if template_error:
        return False, "FAIL", [f"{tool_format} raised on the template-rendered call: {template_error}"]
    if template_calls == [EXPECTED]:
        return True, "PASS", []
    if not template_calls:
        why = f"{tool_format} extracted NO tool call from the template-rendered call (format {detected})"
    else:
        why = f"{tool_format} extracted {template_calls}, not exactly {[EXPECTED]}"
    return False, "FAIL", [why]


async def _parse(parser, tok, text: str, tools):
    _content, calls = await parser.extract_tool_calls(tok.encode(text, add_special_tokens=False), tools)
    return _norm(calls)


def render_toolcall(tok) -> tuple[str | None, str | None]:
    """-> (the assistant turn's serialization, arguments mode) or (None, None)."""
    def render(msgs, **kw):
        return tok.apply_chat_template(msgs, tools=[CALC_TOOL], tokenize=False, **kw)

    base = [{"role": "system", "content": SYS}, {"role": "user", "content": USER}]
    try:
        prompt = render(base, add_generation_prompt=True)
        print("\n===== [1] GENERATION PROMPT =====\n" + prompt, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] render prompt FAILED: {type(e).__name__}: {e}", flush=True)
        return None, None
    for arg_mode in ("dict", "json"):  # some templates want arguments as a dict, others a JSON string
        try:
            msg = json.loads(json.dumps(ASSISTANT_TOOLCALL))
            if arg_mode == "json":
                fn = msg["tool_calls"][0]["function"]
                fn["arguments"] = json.dumps(fn["arguments"])
            full = render(base + [msg], add_generation_prompt=False)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] render tool_call (arguments as {arg_mode}) FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        if not full.startswith(prompt):
            print("[probe] the rendered conversation does not extend the generation prompt", flush=True)
            return None, None
        return full[len(prompt):], arg_mode
    return None, None


async def run(tok) -> int:
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    resp_text, arg_mode = render_toolcall(tok)
    print(f"\n===== [2] ASSISTANT TOOL CALL (arguments as {arg_mode}) =====\n{resp_text!r}", flush=True)
    detected = classify(resp_text)
    print(f"[probe] DETECTED TEMPLATE FORMAT: {detected}", flush=True)

    tools = [OpenAIFunctionToolSchema.model_validate(CALC_TOOL)]
    other = "hermes" if TOOL_FORMAT != "hermes" else "qwen3_coder"
    parser = ToolParser.get_tool_parser(TOOL_FORMAT, tok)
    template_calls, template_error, info = None, None, {}
    if resp_text is not None:
        try:
            template_calls = await _parse(parser, tok, resp_text, tools)
        except Exception as e:  # noqa: BLE001
            template_calls, template_error = [], f"{type(e).__name__}: {e}"
    for label, fmt, sample in (("canonical_xml", TOOL_FORMAT, XML_SAMPLE),
                               (f"template_via_{other}", other, resp_text)):
        if sample is None:
            continue
        try:
            info[label] = await _parse(ToolParser.get_tool_parser(fmt, tok), tok, sample, tools)
        except Exception as e:  # noqa: BLE001
            info[label] = f"raised {type(e).__name__}: {e}"
    print(f"\n===== [3] verl PARSERS =====\n  {TOOL_FORMAT} on the template call -> {template_calls}"
          f"{' (' + template_error + ')' if template_error else ''}", flush=True)
    for k, v in info.items():
        print(f"  {k} -> {v}  (informational)", flush=True)

    ok, status, reasons = decide(TOOL_FORMAT, detected, template_calls, template_error)
    print(f"\n===== VERDICT: {status} =====" + "".join(f"\n  - {r}" for r in reasons), flush=True)
    return probe_verdict.emit("probe_tool_format", ok, status=status, reasons=reasons, tool_format=TOOL_FORMAT,
                              detected_format=detected, template_calls=template_calls,
                              informational=info, model=MODEL_PATH)


def main() -> int:
    try:
        from transformers import AutoTokenizer
        print(f"[probe] loading tokenizer: {MODEL_PATH}", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return asyncio.run(run(tok))
    except Exception as e:  # noqa: BLE001 - a probe that could not run has not passed
        return probe_verdict.emit("probe_tool_format", False, status="ERROR", tool_format=TOOL_FORMAT,
                                  reasons=[f"{type(e).__name__}: {e}"], model=MODEL_PATH)


if __name__ == "__main__":
    sys.exit(main())
