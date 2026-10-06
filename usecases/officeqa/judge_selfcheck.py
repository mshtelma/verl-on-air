#!/usr/bin/env python3
"""Live reward calibration: correct answers with supported/unsupported paths."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import protocol
import reward


def cases():
    question = "What were U.S. national defense expenditures in fiscal year 1940, in millions of dollars?"
    lookup = {"id": "a", "observation": "obs_1", "claim": "FY1940 national defense expenditures were 2,602 million dollars."}
    for name, text, expected in [
        ("supported_lookup", "Treasury Bulletin 1941-06\nExpenditures (millions of dollars)\nNational defense | FY1940 | 2,602", 1),
        ("wrong_period", "Treasury Bulletin 1942-06\nExpenditures (millions of dollars)\nNational defense | FY1941 | 2,602", 0),
        ("wrong_category", "Treasury Bulletin 1941-06\nExpenditures (millions of dollars)\nAgriculture | FY1940 | 2,602\nNational defense | FY1940 | 6,404", 0),
    ]:
        rec = protocol.new_record(f"calibration-{name}-{uuid4().hex}", question)
        protocol.observation(rec, name="read_document", arguments={"file_name": "treasury_bulletin_1941_06.txt"},
                             text=text, request=0, max_chars=4000)
        protocol.delivered(rec, request=1)
        protocol.submit(rec, {"answer": "2,602", "path": [lookup]})
        yield name, rec, "2602", expected
    rec = protocol.new_record(f"calibration-ungrounded-compute-{uuid4().hex}", question)
    protocol.observation(rec, name="compute", arguments={"code": "print(2602)"}, text="Output:\n2602", request=0, max_chars=4000)
    protocol.delivered(rec, request=1)
    protocol.submit(rec, {"answer": "2602", "path": [{**lookup, "claim": "Computed FY1940 national defense expenditures: 2602."}]})
    yield "ungrounded_compute", rec, "2602", 0
    rec = protocol.new_record(f"calibration-supported-arithmetic-{uuid4().hex}",
                              "How much did national defense expenditures increase from FY1940 to FY1941, in millions of dollars?")
    protocol.observation(rec, name="read_document", arguments={"file_name": "treasury_bulletin_1942_06.txt"},
                         text="Treasury Bulletin 1942-06\nNational defense expenditures (millions of dollars) | FY1940 2,602 | FY1941 6,404",
                         request=0, max_chars=4000)
    protocol.delivered(rec, request=1)
    protocol.observation(rec, name="compute", arguments={"code": "print(6404-2602)"}, text="Output:\n3802", request=1, max_chars=4000)
    protocol.delivered(rec, request=2)
    protocol.submit(rec, {"answer": "3,802", "path": [
        {"id": "a", "observation": "obs_1", "claim": "National defense was 2,602 million in FY1940 and 6,404 million in FY1941."},
        {"id": "b", "observation": "obs_2", "claim": "Increase is 6404 minus 2602 = 3802 million dollars.", "depends_on": ["a"]},
    ]})
    yield "supported_arithmetic", rec, "3802", 1


async def main():
    checks = []
    semaphore = asyncio.Semaphore(4)

    async def check(name, record, gold, expected, repeat):
        async with semaphore:
            result = await reward.compute_score(ground_truth=gold, extra_info={"officeqa_record": record, "question": record["question"]})
        passed = result["score"] == expected and result["judge_called"] == 1 and not result["infrastructure_error"]
        row = {"case": name, "repeat": repeat, "expected": expected, "metrics": result, "pass": bool(passed)}
        print(json.dumps({"officeqa_calibration": row}), flush=True)
        checks.append(row)

    try:
        await asyncio.gather(*(check(*case, repeat) for repeat in range(2) for case in cases()))
    finally:
        await reward.close_sessions()
    artifact = {"status": "PASS" if all(row["pass"] for row in checks) else "FAIL", "checks": checks,
                "judge_model": os.environ.get("JUDGE_MODEL_PATH"), "run_id": os.environ.get("RUN_ID")}
    if os.environ.get("OQ_CALIBRATION_OUT"):
        path = Path(os.environ["OQ_CALIBRATION_OUT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n")
    if artifact["status"] != "PASS":
        raise RuntimeError("OfficeQA support judge failed calibration")
    print(json.dumps({"officeqa_calibration": "PASS", "checks": len(checks)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
