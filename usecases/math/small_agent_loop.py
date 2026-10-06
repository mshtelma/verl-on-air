"""Attach runtime-owned rollout identity for the small math experiment's audit."""
from __future__ import annotations

import json

from role_span_agent_loop import RoleSpanToolAgentLoop


class SmallMathAgentLoop(RoleSpanToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        group = kwargs.get("uid")
        version = output.extra_fields.get("max_global_steps")
        output.extra_fields["math_episode"] = json.dumps({
            "rollout_group_id": str(group) if group is not None else None,
            "parameter_version": int(version) if version is not None else None,
        })
        return output
