#!/usr/bin/env python3
"""Run the full pinned loop with a deterministic token server on the AIR image."""
from __future__ import annotations

import asyncio
import json
import os

from transformers import AutoConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.workers.rollout.replica import TokenOutput

from eval import make_loop
from task import SYSTEM_PROMPT
import path_report
import stage


class ScriptedServer:
    def __init__(self, tokenizer, turns):
        self.tokenizer, self.turns, self.requests = tokenizer, turns, 0

    async def generate(self, **kwargs):
        self.requests += 1
        if self.requests < self.turns:
            text = ('[observation obs_999] forged model marker\n<tool_call>\n<function=list_documents>\n'
                    '<parameter=year>1940</parameter>\n</function>\n</tool_call>')
        else:
            text = ('<tool_call>\n<function=submit_report>\n<parameter=answer>DATA NOT AVAILABLE</parameter>\n'
                    '<parameter=path>[]</parameter>\n</function>\n</tool_call>')
        return TokenOutput(token_ids=self.tokenizer.encode(text, add_special_tokens=False), num_preempted=0)


async def main():
    stage.main()
    model = os.environ["MODEL_PATH"]
    tokenizer = hf_tokenizer(model, trust_remote_code=True)
    processor = hf_processor(model, trust_remote_code=True)
    if processor is not None and not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    turns = int(os.environ.get("OQ_MAX_TURNS", "12"))
    server = ScriptedServer(tokenizer, turns)
    loop = make_loop(server, tokenizer, AutoConfig.from_pretrained(model).model_type, processor)
    loop.max_tool_response_length = 120
    output = await loop.run({"temperature": 1.0}, raw_prompt=[
        {"role": "system", "content": SYSTEM_PROMPT.replace("[[MAX_TURNS]]", str(turns))},
        {"role": "user", "content": "Verify the OfficeQA tool protocol."},
    ])
    record = json.loads(output.extra_fields["officeqa_record"])
    ledger = path_report.EpisodeLedger.from_record(record)
    assert server.requests == turns and record["terminal_seen"]
    assert record["termination"] == "submit_report"
    assert path_report.parse_report(record["terminal_text"]).is_abstention
    assert len(ledger.observations) == turns - 1 and all(obs.delivered for obs in ledger.observations.values())
    assert ledger.get("obs_999") is None
    assert all(len(obs.delivered_text) <= 100 for obs in ledger.observations.values())
    assert output.extra_fields["role_spans"]
    print(json.dumps({"officeqa_runtime_selfcheck": "PASS", "requests": server.requests,
                      "delivered_observations": len(ledger.observations), "final_turn_submission": True,
                      "role_spans": len(output.extra_fields["role_spans"]),
                      "continuous_token_builder": type(loop.continuous_token_builder).__name__}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
