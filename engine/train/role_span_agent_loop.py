#!/usr/bin/env python3
"""verl's ToolAgentLoop, plus a record of which characters of the decoded response the model wrote.

Registered under verl's own name `tool_agent` by engine/train/agent_loops.yaml, which both launchers
pass as actor_rollout_ref.rollout.agent.agent_loop_config_path whenever MULTI_TURN=True -- so every
sample that already routes to agent_name="tool_agent" gets it with no data change. The spans reach
the reward as extra_info["role_spans"]; see engine/lib/role_spans.py for why the reward needs them.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import role_spans  # noqa: E402

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop  # noqa: E402


class RoleSpanToolAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs):
        output = await super().run(sampling_params, **kwargs)
        # a few tokenizer.decode calls over the episode: off the event loop
        output.extra_fields["role_spans"] = await asyncio.get_running_loop().run_in_executor(
            None, role_spans.compute, self.tokenizer, output.response_ids, output.response_mask)
        return output
