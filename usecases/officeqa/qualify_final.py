#!/usr/bin/env python3
"""Prepare the final split and qualify its image, 128-step protocol and hosted judge."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path


async def main():
    import torch
    import prep_data
    import runtime_selfcheck
    import judge_selfcheck

    versions = {name: importlib.metadata.version(name) for name in ("torch", "verl", "vllm", "transformers")}
    for name, expected in {"torch": "2.13.0", "verl": "0.10.0.dev", "vllm": "0.30.0", "transformers": "5.16.1"}.items():
        if not versions[name].startswith(expected):
            raise RuntimeError(f"tested image mismatch: {name}={versions[name]}, expected {expected}")
    print(json.dumps({"officeqa_versions": versions, "cuda": torch.version.cuda}), flush=True)
    prep_data.main()
    await runtime_selfcheck.main()
    await judge_selfcheck.main()
    receipt = {"status": "PASS", "versions": versions, "cuda": torch.version.cuda,
               "max_turns": int(os.environ["OQ_MAX_TURNS"]),
               "episode_token_budget": int(os.environ["EPISODE_TOKEN_BUDGET"]),
               "judge_model": os.environ["JUDGE_MODEL"], "judge_provider": os.environ["JUDGE_PROVIDER"],
               "data_dir": os.environ["OQ_DATA_DIR"], "run_id": os.environ["RUN_ID"],
               "git_sha": os.environ["GIT_SHA"]}
    path = Path(os.environ["OQ_CALIBRATION_OUT"]).with_name("qualification.json")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"officeqa_final_qualification": receipt}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
