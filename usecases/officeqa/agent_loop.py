"""OfficeQA terminal submissions and evidence delivery on verl's continuous-token loop.

The generation hook follows verl 8718ca30. Tool execution and context merging stay
upstream; runtime evidence is recorded only after clipping and committed delivery.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "train"))
from role_span_agent_loop import RoleSpanToolAgentLoop  # noqa: E402
from verl.experimental.agent_loop.tool_agent_loop import AgentState, SPEC_DECODE_EXTRA_KEYS  # noqa: E402
from verl.tools.schemas import ToolResponse  # noqa: E402
from verl.utils.profiler import simple_timer  # noqa: E402

import protocol  # noqa: E402


class OfficeQAToolAgentLoop(RoleSpanToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_parallel_calls = 1
        self.generation_tokens = int(os.environ.get("OQ_MAX_GENERATION_TOKENS", "1024"))

    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        record = protocol.finalize(output.extra_fields["officeqa_record"])
        output.extra_fields["officeqa_record"] = json.dumps(record, ensure_ascii=False, allow_nan=False)
        return output

    async def _handle_pending_state(self, agent_data, sampling_params):
        question = next((str(m.get("content") or "") for m in agent_data.messages if m["role"] == "user"), "")
        agent_data.extra_fields["officeqa_record"] = protocol.new_record(agent_data.request_id, question)
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        record = agent_data.extra_fields["officeqa_record"]
        remaining = self.response_length - len(agent_data.response_mask)
        if remaining <= 0:
            record["termination"] = "response_limit"
            return AgentState.TERMINATED
        # Pending tool results reached a committed context only if upstream returned
        # GENERATING. A rejected context merge never enters this hook again.
        protocol.delivered(record, request=agent_data.assistant_turns)
        params = {**sampling_params, "max_tokens": min(remaining, self.generation_tokens)}
        if self.tool_parser.stop_token_ids:
            params["stop_token_ids"] = sorted(set((params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id, prompt_ids=agent_data.prompt_ids, sampling_params=params,
                image_data=agent_data.image_data, video_data=agent_data.video_data,
                mm_processor_output=agent_data.mm_processor_output, audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        first = agent_data.assistant_turns == 0
        if first:
            agent_data.extra_fields.update(output.extra_fields)
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted or 0
            if output.extra_fields.get("max_global_steps") is not None:
                agent_data.extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields.get(key, 0)) + int(output.extra_fields[key])
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        merged, mask, logprobs = await self.ct_merge_assistant_token(
            agent_data.prompt_ids, output.token_ids, agent_data.response_mask,
            agent_data.response_logprobs if (agent_data.response_logprobs or output.log_probs) else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        agent_data.prompt_ids, agent_data.response_mask = merged.token_ids, mask
        if logprobs is not None:
            agent_data.response_logprobs = logprobs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        active_tools = getattr(agent_data, "_active_tools", self.tools)
        content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            output.token_ids, [t.tool_schema for t in active_tools.values()])
        agent_data.messages.append(self._build_assistant_message(content, agent_data))
        calls = agent_data.tool_calls[:self.max_parallel_calls]
        # Accept a complete terminal call BEFORE turn/response termination. It must
        # fit in the actual returned trajectory; tokens later sliced off cannot win.
        if calls and calls[0].name == protocol.SUBMIT_TOOL and len(mask) <= self.response_length:
            protocol.submit(record, calls[0].arguments)
            record["termination"] = "submit_report"
            return AgentState.TERMINATED
        if len(mask) >= self.response_length:
            record["termination"] = "response_limit"
        elif ((self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
              or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)):
            record["termination"] = "turn_limit"
        elif not calls:
            record["termination"] = "no_tool_call"
        else:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        response, reward, extra = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        try:
            arguments = protocol.path_report.loads_strict(tool_call.arguments)
            if not isinstance(arguments, dict):
                arguments = {}
        except (TypeError, ValueError):
            arguments = {}
        text = protocol.observation(
            agent_data.extra_fields["officeqa_record"], name=tool_call.name, arguments=arguments,
            text=response.text or "", request=agent_data.assistant_turns - 1,
            max_chars=self.max_tool_response_length,
        )
        return ToolResponse(text=text), reward, extra
