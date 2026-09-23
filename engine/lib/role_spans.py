#!/usr/bin/env python3
"""Which characters of a decoded episode the MODEL wrote, and which came from tools or the template.

verl's reward manager hands compute_score one string -- the whole decoded response: assistant turns,
tool responses and chat-template text -- with no record of who wrote what. The DataProto it scores
carries no response_mask (verl v0.9.0, agent_loop._compute_score builds it from prompts / responses /
attention_mask / input_ids / position_ids only). A reward that scans that string cannot tell an
`<answer>` the model committed from one inside a tool response, nor a genuine tool response from
"<tool_response>" text the model typed itself.

The agent loop does know: ToolAgentLoop's response_mask is 1 for every token the model generated and
0 for tool-response / template tokens, aligned 1:1 with response_ids. The RoleSpanToolAgentLoop
(engine/train/role_span_agent_loop.py) turns it into COMPACT character spans over the exact string
the reward receives -- decode(response_ids, skip_special_tokens=True) -- and passes them via
extra_fields, which both the naive and rate_limited managers merge into extra_info["role_spans"].

    spans = compute(tokenizer, response_ids, response_mask)   # agent-loop side
    assistant, tool = split(solution_str, spans)              # reward side; ProvenanceError if unusable

Spans are plain [[role, start, end], ...] (role "assistant" or "tool"): they also become a column of
the training batch, so they must stay small -- integers, never text.
"""
from __future__ import annotations

from typing import Any, Sequence

ASSISTANT, TOOL = "assistant", "tool"


class ProvenanceError(ValueError):
    """The role spans are missing or do not describe this exact string."""


def compute(tokenizer: Any, response_ids: Sequence[int], response_mask: Sequence[int]) -> list[list] | None:
    """Char spans over tokenizer.decode(response_ids, skip_special_tokens=True). None when the
    pieces do not reassemble into that decode (then the reward refuses to score, rather than guess)."""
    ids = [int(t) for t in response_ids]
    mask = [int(m) for m in response_mask]
    if len(mask) != len(ids):
        return None
    runs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or mask[i] != mask[start]:
            runs.append((ASSISTANT if mask[start] else TOOL, start, i))
            start = i
    texts = [tokenizer.decode(ids[a:b], skip_special_tokens=True) for _, a, b in runs]
    if "".join(texts) != tokenizer.decode(ids, skip_special_tokens=True):
        return None
    spans: list[list] = []
    pos = 0
    for (role, _, _), text in zip(runs, texts):
        if text:
            if spans and spans[-1][0] == role:
                spans[-1][2] += len(text)
            else:
                spans.append([role, pos, pos + len(text)])
        pos += len(text)
    return spans


def split(text: str, spans: Any) -> tuple[list[str], list[str]]:
    """(assistant segments, tool segments) of `text`, or ProvenanceError. The spans must be sorted,
    contiguous and cover exactly [0, len(text)) -- anything else means they were computed for a
    different string (a misconfigured agent loop, or another sample's spans)."""
    if spans is None:
        raise ProvenanceError("no role spans: the agent loop did not record which text the model wrote "
                              "(is actor_rollout_ref.rollout.agent.agent_loop_config_path set?)")
    try:
        items = [(str(r), int(a), int(b)) for r, a, b in spans]
    except (TypeError, ValueError):
        raise ProvenanceError(f"malformed role spans: {str(spans)[:200]}") from None
    assistant, tool, pos = [], [], 0
    for role, a, b in items:
        if role not in (ASSISTANT, TOOL) or a != pos or b <= a:
            raise ProvenanceError(f"role spans are not contiguous from 0 at {role, a, b} (expected start {pos})")
        (assistant if role == ASSISTANT else tool).append(text[a:b])
        pos = b
    if pos != len(text):
        raise ProvenanceError(f"role spans cover {pos} of {len(text)} characters: they describe a different string")
    return assistant, tool
