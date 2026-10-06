"""Exercise the actual generation hook with deterministic server/parser seams."""
import asyncio
import contextlib
import json
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest

import protocol
from support import REPO, load_module


@pytest.fixture
def controller(monkeypatch):
    class State(Enum):
        GENERATING = "generating"
        PROCESSING_TOOLS = "processing_tools"
        TERMINATED = "terminated"

    class Parent:
        def _build_assistant_message(self, content, data):
            return {"role": "assistant", "content": content}

    modules = {
        "role_span_agent_loop": {"RoleSpanToolAgentLoop": Parent},
        "verl.experimental.agent_loop.tool_agent_loop": {"AgentState": State, "SPEC_DECODE_EXTRA_KEYS": ()},
        "verl.tools.schemas": {"ToolResponse": SimpleNamespace},
        "verl.utils.profiler": {"simple_timer": lambda *args: contextlib.nullcontext()},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(__import__("sys").modules, name, module)
    return load_module(REPO / "usecases" / "officeqa" / "agent_loop.py")


def setup(controller, *, response_limit=10, turn_limit=1, merged_tokens=3):
    loop = object.__new__(controller.OfficeQAToolAgentLoop)
    loop.response_length = response_limit
    loop.generation_tokens = 2
    loop.max_assistant_turns = turn_limit
    loop.max_user_turns = turn_limit
    loop.max_parallel_calls = 1
    loop.tools = {}
    rec = protocol.new_record("episode", "q")
    data = SimpleNamespace(extra_fields={"officeqa_record": rec}, assistant_turns=0, user_turns=0,
                           response_mask=[], response_logprobs=[], prompt_ids=[100], metrics={},
                           image_data=None, video_data=None, audio_data=None, mm_processor_output=None,
                           mm_processor_kwargs={}, request_id="episode", messages=[])
    calls = [SimpleNamespace(name="submit_report", arguments=json.dumps({"answer": "DATA NOT AVAILABLE", "path": []}))]

    async def parse(ids, tools):
        return "", calls

    loop.tool_parser = SimpleNamespace(stop_token_ids=[9], extract_tool_calls=parse)
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(token_ids=[1, 2], num_preempted=0, extra_fields={}, log_probs=None, routed_experts=None)

    loop.server_manager = SimpleNamespace(generate=generate)

    async def merge(*args, **kwargs):
        return SimpleNamespace(token_ids=[100] + [1] * merged_tokens), [1] * merged_tokens, None

    loop.ct_merge_assistant_token = merge
    return loop, data, seen, calls


@pytest.mark.parametrize("response_limit", [3, 10])
def test_complete_last_turn_submission_is_accepted_even_at_exact_response_budget(controller, response_limit):
    loop, data, seen, _ = setup(controller, response_limit=response_limit)
    state = asyncio.run(loop._handle_generating_state(data, {"temperature": 1}))
    assert state == controller.AgentState.TERMINATED
    assert data.extra_fields["officeqa_record"]["terminal_seen"]
    assert data.extra_fields["officeqa_record"]["termination"] == "submit_report"
    assert seen[0]["sampling_params"]["max_tokens"] == 2
    assert seen[0]["sampling_params"]["stop_token_ids"] == [9]


def test_submission_whose_tokens_will_be_sliced_off_is_not_accepted(controller):
    loop, data, _, _ = setup(controller, response_limit=2, merged_tokens=3)
    asyncio.run(loop._handle_generating_state(data, {}))
    rec = data.extra_fields["officeqa_record"]
    assert not rec["terminal_seen"] and rec["termination"] == "response_limit"


def test_evidence_is_marked_delivered_before_generation_on_committed_context(controller):
    loop, data, _, _ = setup(controller, turn_limit=2)
    data.assistant_turns = 1
    data.metrics["num_preempted"] = 0
    rec = data.extra_fields["officeqa_record"]
    protocol.observation(rec, name="read_document", arguments={}, text="100", request=0, max_chars=100)
    asyncio.run(loop._handle_generating_state(data, {}))
    assert rec["observations"][0]["delivered_to_request"] == 1
    assert rec["observations"][0]["delivered_text"] == "100"


def test_last_turn_retrieval_is_not_executed_when_no_later_context_exists(controller):
    loop, data, _, calls = setup(controller)
    calls[0].name = "read_document"
    state = asyncio.run(loop._handle_generating_state(data, {}))
    assert state == controller.AgentState.TERMINATED
    assert data.extra_fields["officeqa_record"]["observations"] == []
