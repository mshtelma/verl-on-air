"""usecases/math/eval.py under the eval contract (REVIEW.md R07): an inference outage is an
INVALID run, not a low accuracy; the problem set must be the expected one."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from support import FakeOpenAIServer, FakeTokenizer, clear_verl_tool_registry, env, load_usecase, pinned_verl

MODELS = {"object": "list", "data": [{"id": "eval", "object": "model"}]}
PROBLEMS = [{"idx": i, "problem": f"What is {i} + {i}?", "gt": str(2 * i), "level": 3, "type": "Algebra"}
            for i in range(1, 4)]


def completion(text: str) -> dict:
    return {"choices": [{"text": text, "finish_reason": "stop"}]}


def solver(path, payload, n):
    """Calls the calculator once, then boxes the calculator's result."""
    if path.endswith("/models"):
        return 200, MODELS, 0
    prompt = payload["prompt"]
    if '"role": "tool"' not in prompt:
        i = int(prompt.split("What is ")[1].split(" +")[0])
        call = (f"<tool_call>\n<function=calculator>\n<parameter=expression>\n{i}+{i}\n</parameter>\n"
                "</function>\n</tool_call>")
        return 200, completion(call), 0
    result = json.loads(prompt)[-1]["content"]
    return 200, completion(f"The sum is \\boxed{{{result}}}"), 0


def run_eval(tmp: Path, url: str, rows=PROBLEMS, tok: FakeTokenizer | None = None, **overrides: str) -> int:
    """The real eval against a fake vLLM; verl's parser, tool registry and prime_math are the pinned
    verl's own (tests/support.pinned_verl)."""
    e = {"EVAL_BASE_URL": url, "EVAL_MODEL": "eval", "EVAL_OUT": str(tmp / "out.json"),
         "EVAL_HTTP_RETRIES": "1", "EVAL_MAX_TURNS": "3", **overrides}
    tok = tok or FakeTokenizer()
    with pinned_verl():
        clear_verl_tool_registry()
        E = load_usecase("math", "eval", **e)
        E._load_tokenizer = lambda: tok
        E._load_dataset = lambda: list(rows)
        with env(**e):
            return asyncio.run(E._main_async())


def test_healthy_run_with_tool_calls_is_valid(tmp_path: Path):
    with FakeOpenAIServer(solver) as srv:
        rc = run_eval(tmp_path, srv.url, EVAL_EXPECT_N="3")
    a = json.loads((tmp_path / "out.json").read_text())
    assert rc == 0 and a["valid"] and a["accuracy"] == 1.0 and a["used_tool"] == 3


def test_inference_outage_is_invalid_not_wrong(tmp_path: Path):
    def down(path, payload, n):
        return (200, MODELS, 0) if path.endswith("/models") else (502, {"error": "bad gateway"}, 0)
    with FakeOpenAIServer(down) as srv:
        rc = run_eval(tmp_path, srv.url, EVAL_EXPECT_N="3")
    a = json.loads((tmp_path / "out.json").read_text())
    assert rc == 1 and not a["valid"] and a["n_scored"] == 0
    assert all(r["status"] == "infra_inference" and r["pred"] is None for r in a["results"])


def test_a_missing_problem_invalidates_the_run(tmp_path: Path):
    with FakeOpenAIServer(solver) as srv:
        rc = run_eval(tmp_path, srv.url, rows=PROBLEMS[:2], EVAL_EXPECT_N="500")
    assert rc == 1
    assert "expected 500" in json.loads((tmp_path / "out.json").read_text())["invalid_reasons"][0]


def test_unlisted_served_model_is_not_ready(tmp_path: Path):
    with FakeOpenAIServer(solver) as srv:
        with pytest.raises(SystemExit) as ex:
            run_eval(tmp_path, srv.url, EVAL_MODEL="missing")
    assert ex.value.code == 2


def test_the_model_sees_the_training_prompt_tools_and_its_own_reasoning(tmp_path: Path):
    def thinks_then_calls(path, payload, n):
        if path.endswith("/models"):
            return 200, MODELS, 0
        if '"role": "tool"' not in payload["prompt"]:
            return 200, completion("Let me add them first. <tool_call>\n<function=calculator>\n"
                                   "<parameter=expression>\n1+1\n</parameter>\n</function>\n</tool_call>"), 0
        return 200, completion("So \\boxed{2}"), 0

    tok = FakeTokenizer()
    with FakeOpenAIServer(thinks_then_calls) as srv:
        rc = run_eval(tmp_path, srv.url, rows=PROBLEMS[:1], tok=tok, EVAL_EXPECT_N="1")
    assert rc == 0
    with pinned_verl():
        clear_verl_tool_registry()
        prep = load_usecase("math", "prep_data")
        load_usecase("math", "tool")        # registers the calculator, as training's function_tool_path does
        from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY
        training_tools = [FUNCTION_TOOL_REGISTRY["calculator"].tool_schema.model_dump(
            exclude_unset=True, exclude_none=True)]
    first, second = tok.renders[0], tok.renders[1]
    assert first["messages"][0] == {"role": "system", "content": prep.SYSTEM_PROMPT}
    assert first["tools"] == training_tools                       # not a hand-written copy
    call = second["messages"][2]
    assert call["role"] == "assistant" and call["content"] == "Let me add them first."   # reasoning kept
    assert call["tool_calls"][0]["function"] == {"name": "calculator", "arguments": {"expression": "1+1"}}
    a = json.loads((tmp_path / "out.json").read_text())
    assert a["eval_policy"]["version"] == 2 and a["eval_policy"]["parser"] == "verl ToolParser.extract_tool_calls"


def test_without_the_training_parser_the_eval_does_not_start(tmp_path: Path):
    with FakeOpenAIServer(solver) as srv:
        with pytest.raises(SystemExit) as ex:
            run_eval(tmp_path, srv.url, TOOL_FORMAT="no-such-parser")
        assert ex.value.code == 2 and not [p for p, _ in srv.requests if p.endswith("/completions")]
