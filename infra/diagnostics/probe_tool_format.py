#!/usr/bin/env python3
"""De-risk a tool-call FORMAT MISMATCH before spending another multi-node run.

A training run logged 120x `Failed to decode tool call: Expecting value: line 2
column 1 (char 1)` and never executed the calculator: TOOL_FORMAT=hermes ran
json.loads() on what the model actually emitted. This probe answers, definitively
and cheaply (tokenizer only — NO model weights, NO GPU compute, seconds on 1xA10):

  (1) Does Qwen3.5-35B-A3B's OWN chat template serialize a tool call as hermes
      JSON  ({"name": ..., "arguments": {...}})  or as Qwen XML
      (<function=name><parameter=key>value</parameter></function>) ?
  (2) Does verl v0.9.0's `qwen3_coder` parser (Qwen3XMLToolParser) round-trip that
      call to name=calculator, arguments={"expression": ...} ?
  (3) Does `hermes` reproduce run3's EXACT error on the same text (proving it was
      the wrong parser, not a model/data problem) ?

PASS bar for setting TOOL_FORMAT=qwen3_coder with confidence:
  DETECTED FORMAT = QWEN_XML, qwen3_coder parses calculator/expression, hermes errors.
"""

import json
import logging
import os
import re

# verl parsers report decode failures via logging.error — surface them.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

MODEL_PATH = os.environ.get("MODEL_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")

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

SYS = "You are a careful math solver. Use the calculator tool for arithmetic."
USER = "Janet has 18 eggs, eats 3, bakes 4. How many are left?"
ASSISTANT_TOOLCALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "18 - 3 - 4"}}}
    ],
}

# A canonical Qwen XML tool call (deterministic, template-independent). If the
# template render is quirky, this still proves parser behaviour + reproduces the bug.
XML_SAMPLE = (
    "<tool_call>\n<function=calculator>\n<parameter=expression>\n18 - 3 - 4\n"
    "</parameter>\n</function>\n</tool_call>"
)


def classify(text: str | None) -> str:
    if not text:
        return "UNKNOWN (no render)"
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    inner = m.group(1) if m else text
    if "<function=" in inner:
        return "QWEN_XML (<function=...>)"
    if re.search(r'\{\s*"name"', inner):
        return "HERMES_JSON"
    return f"OTHER (inner={inner[:120]!r})"


def hermes_on(text: str):
    """Reproduce verl HermesToolParser's decode step: json.loads the <tool_call> body.
    (verl uses the `regex` module; stdlib `re` gives an identical result for this pattern.)"""
    calls, err = [], None
    for m in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m)["name"])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return calls, err


def main() -> None:
    from transformers import AutoTokenizer

    print(f"[probe] loading tokenizer: {MODEL_PATH}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def render(msgs, **kw):
        return tok.apply_chat_template(msgs, tools=[CALC_TOOL], tokenize=False, **kw)

    print("\n===== [1] GENERATION PROMPT (what the model is told about the tool) =====", flush=True)
    prompt = None
    try:
        prompt = render(
            [{"role": "system", "content": SYS}, {"role": "user", "content": USER}],
            add_generation_prompt=True,
        )
        print(prompt, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] render prompt FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n===== [2] HOW THIS TEMPLATE SERIALIZES AN ASSISTANT TOOL CALL =====", flush=True)
    resp_text = None
    for arg_mode in ("dict", "json"):  # some templates want arguments as a dict, others a JSON string
        try:
            msg = json.loads(json.dumps(ASSISTANT_TOOLCALL))  # deep copy
            if arg_mode == "json":
                fn = msg["tool_calls"][0]["function"]
                fn["arguments"] = json.dumps(fn["arguments"])
            full = render(
                [{"role": "system", "content": SYS}, {"role": "user", "content": USER}, msg],
                add_generation_prompt=False,
            )
            resp_text = full[len(prompt):] if (prompt and full.startswith(prompt)) else full
            print(f"[probe] (arguments passed as {arg_mode}) assistant-turn serialization:", flush=True)
            print(repr(resp_text), flush=True)
            print("---- readable ----", flush=True)
            print(resp_text, flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[probe] render tool_call (arguments as {arg_mode}) FAILED: {type(e).__name__}: {e}", flush=True)

    detected = classify(resp_text)
    print(f"\n[probe] DETECTED TEMPLATE FORMAT: {detected}", flush=True)

    print("\n===== [3] verl v0.9.0 PARSER ROUND-TRIP (qwen3_coder vs hermes) =====", flush=True)
    qwen_ok = False
    hermes_failed = False
    try:
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema

        try:
            schemas = [OpenAIFunctionToolSchema.model_validate(CALC_TOOL)]
        except Exception as e:  # noqa: BLE001
            print(f"[probe] OpenAIFunctionToolSchema.model_validate failed ({e}); parsing with tools=None", flush=True)
            schemas = None

        qp = ToolParser.get_tool_parser("qwen3_coder", tok)
        for name, sample in [("template-rendered", resp_text), ("canonical-xml", XML_SAMPLE)]:
            if not sample:
                print(f"\n-- {name}: (skipped, no text) --", flush=True)
                continue
            print(f"\n-- sample: {name} --", flush=True)
            try:
                fcs = qp._get_function_calls(sample)
                parsed = [qp._parse_xml_function_call(s, schemas) for s in fcs]
                parsed = [(p.name, p.arguments) for p in parsed if p]
                print(f"   qwen3_coder -> {parsed}", flush=True)
                if any(n == "calculator" and "expression" in a for n, a in parsed):
                    qwen_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"   qwen3_coder RAISED: {type(e).__name__}: {e}", flush=True)
            calls, err = hermes_on(sample)
            print(f"   hermes      -> calls={calls} err={err!r}", flush=True)
            if err and not calls:
                hermes_failed = True
    except Exception as e:  # noqa: BLE001
        print(f"[probe] verl parser section FAILED to import/run: {type(e).__name__}: {e}", flush=True)

    print("\n===== VERDICT =====", flush=True)
    xml = detected.startswith("QWEN_XML")
    if xml and qwen_ok and hermes_failed:
        print("PASS: Qwen3.5 emits XML tool calls; qwen3_coder parses them; hermes fails "
              "(reproduces run3). -> TOOL_FORMAT=qwen3_coder is CORRECT.", flush=True)
    elif detected.startswith("HERMES_JSON"):
        print("UNEXPECTED: template renders HERMES_JSON — the run3 bug is NOT a hermes/xml "
              "mismatch; re-investigate before changing TOOL_FORMAT.", flush=True)
    else:
        print(f"INCONCLUSIVE: detected={detected} qwen3_coder_ok={qwen_ok} hermes_failed={hermes_failed}. "
              "Inspect sections [2]/[3] above.", flush=True)
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
