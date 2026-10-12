#!/usr/bin/env python3
"""Freeze the staged OfficeQA corpus and experiment split as verl parquet."""
from __future__ import annotations

import csv
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

from build_corpus import ARCHIVE_NAME, LOCK_PATH, archive_identity, check_identity, file_identity
from task import SYSTEM_PROMPT, Question, official_split
import sandbox

HERE = Path(__file__).resolve().parent
TOOLS = json.loads((HERE / "tools.json").read_text())


def verl_rows(questions: list[Question], *, max_turns: int, difficulties: dict[str, str]) -> list[dict]:
    """Gold and source-file hints never enter the policy's prompt."""
    system = SYSTEM_PROMPT.replace("[[MAX_TURNS]]", str(max_turns))
    return [
        {
            "data_source": "officeqa-path-report",
            "agent_name": "tool_agent",
            "prompt": [{"role": "system", "content": system}, {"role": "user", "content": q.question}],
            "reward_model": {"style": "rule", "ground_truth": q.answer},
            "extra_info": {
                "uid": q.uid,
                "question": q.question,
                "question_requirements": q.requirements,
                "difficulty": difficulties[q.uid],
                "tool_selection": [tool["function"]["name"] for tool in TOOLS],
            },
        }
        for q in questions
    ]


def pilot_questions(train: list[Question], difficulties: dict[str, str], per_difficulty: int = 8) -> list[Question]:
    ordered = sorted(train, key=lambda q: hashlib.sha256(q.uid.encode()).hexdigest())
    selected = []
    for difficulty in ("easy", "hard"):
        candidates = [q for q in ordered if difficulties[q.uid] == difficulty]
        selected.extend(candidates[:per_difficulty])
    return selected


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def verify_snapshot(source: Path, lock: dict) -> dict:
    """Reject changed questions, document text or retrieval chunks before staging."""
    check_identity("question CSV", file_identity(source / "officeqa_full.csv"), lock["question_csv"])
    corpus = archive_identity(source / ARCHIVE_NAME)
    check_identity("document corpus", corpus, lock["corpus"])
    chunks = file_identity(source / "chunks.jsonl")
    with (source / "chunks.jsonl").open(encoding="utf-8") as stream:
        chunks["count"] = sum(1 for line in stream if line.strip())
    check_identity("retrieval chunks", chunks, lock["chunks"])
    return {"status": "MATCH", "source": lock["source"], "lock_sha256": dm.sha256_file(LOCK_PATH),
            "corpus": corpus, "chunks": chunks}


def main() -> None:
    source = Path(os.environ["OQ_SOURCE_DIR"])
    output = Path(os.environ["OQ_DATA_DIR"])
    turns = int(os.environ.get("OQ_MAX_TURNS", "12"))
    if (output / "DATA_MANIFEST.json").exists():
        raise ValueError(f"snapshot already exists: {output}; choose a new OQ_DATA_DIR")
    required = [source / "officeqa_full.csv", source / ARCHIVE_NAME, source / "chunks.jsonl"]
    for path in required:
        if not path.is_file():
            available = sorted(p.name for p in source.iterdir()) if source.is_dir() else []
            raise FileNotFoundError(f"missing {path}; available OfficeQA files: {available}")
    snapshot = verify_snapshot(source, json.loads(LOCK_PATH.read_text()))
    print("[officeqa-prep] measured data snapshot MATCH", flush=True)
    sandbox_result = "PENDING"
    if os.environ.get("OQ_REQUIRE_SANDBOX", "1") == "1":
        if not sandbox._sandbox_ok():
            raise RuntimeError("OfficeQA compute needs working bubblewrap/user namespaces on AIR")
        result = sandbox.run_code("print(sum([132,129,143,159,154,153,177,200,219,287,376,473]))")
        if "2602" not in result or result.startswith("Error:"):
            raise RuntimeError(f"compute arithmetic probe failed: {result}")
        print("[officeqa-prep] sandbox arithmetic PASS", flush=True)
        sandbox_result = "PASS"
    else:
        for label, flags in [("user_net", ["--unshare-user", "--unshare-net"]),
                             ("user_net_pid", ["--unshare-user", "--unshare-net", "--unshare-pid"])]:
            probe = subprocess.run(["bwrap", *flags, "--ro-bind", "/", "/", "true"],
                                   capture_output=True, text=True, timeout=15)
            print(json.dumps({"sandbox_probe": label, "rc": probe.returncode, "error": probe.stderr.strip()}), flush=True)
        libc = ctypes.CDLL(None, use_errno=True)
        abi = libc.syscall(444, 0, 0, 1)
        print(json.dumps({"landlock_abi": abi, "errno": ctypes.get_errno(),
                          "seccomp_library": ctypes.util.find_library("seccomp")}), flush=True)
    with required[0].open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    difficulties = {row["uid"]: row["difficulty"] for row in csv_rows}
    if len(difficulties) != len(csv_rows):
        raise ValueError("duplicate question uid")
    counts = Counter(difficulties.values())
    if counts != {"easy": 113, "hard": 133}:
        raise ValueError(f"expected the official 246-question benchmark, got {dict(counts)}")
    train, heldout = official_split(csv_rows, hard_train=90)
    pilot = pilot_questions(train, difficulties)
    if set(q.uid for q in train) & set(q.uid for q in heldout):
        raise ValueError("training/held-out question overlap")
    if {q.question for q in train} & {q.question for q in heldout}:
        raise ValueError("training/held-out exact question text overlap")
    output.mkdir(parents=True, exist_ok=True)
    sources = [{"path": str(path), "bytes": path.stat().st_size, "sha256": dm.sha256_file(path)} for path in required]
    print(f"[officeqa-prep] sources: {json.dumps(sources)}", flush=True)
    copy_atomic(required[1], output / required[1].name)
    outputs = []
    all_rows = []
    for filename, questions in [("train.parquet", train), ("test.parquet", heldout), ("pilot_train.parquet", pilot)]:
        rows = verl_rows(questions, max_turns=turns, difficulties=difficulties)
        dm.write_parquet(output / filename, rows)
        outputs.append(dm.output_record(output / filename, len(rows)))
        all_rows.extend(rows)
    documents, chunks = snapshot["corpus"]["documents"], snapshot["chunks"]["count"]
    chunk_receipt = {"name": "chunks.jsonl", "bytes": sources[2]["bytes"], "sha256": sources[2]["sha256"]}
    copy_atomic(required[2], output / "chunks.jsonl")
    from transformers import AutoTokenizer
    tokenizer_path = os.environ.get("OQ_TOKENIZER_PATH", "/Volumes/main/mshtelma/verl/models/Qwen3.5-35B-A3B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_lengths = [len(tokenizer.apply_chat_template(row["prompt"], tools=TOOLS, add_generation_prompt=True,
                                                      tokenize=True, return_dict=False)) for row in all_rows]
    maximum = max(prompt_lengths)
    manifest = dm.write_manifest(
        output / "DATA_MANIFEST.json", tool="officeqa/prep_data.py", sources=sources, outputs=outputs,
        miles_source=json.loads((HERE / "SOURCE.json").read_text()), max_turns=turns,
        split={"train": 203, "heldout": 43, "hard_train": 90, "pilot_train": len(pilot),
               "train_ids": [q.uid for q in train], "heldout_ids": [q.uid for q in heldout]},
        corpus={"name": required[1].name, "sha256": sources[1]["sha256"], "bytes": sources[1]["bytes"]},
        chunks=chunk_receipt, document_count=documents, chunk_count=chunks, data_snapshot=snapshot,
        maximum_rendered_prompt_tokens=maximum, tokenizer=tokenizer_path, sandbox_probe=sandbox_result,
    )
    print(json.dumps({"status": "PASS", "data_dir": str(output), "rows": {r["name"]: r["rows"] for r in outputs},
                      "max_prompt_tokens": maximum, "chunks": chunks, "manifest": str(output / "DATA_MANIFEST.json"),
                      "run_id": manifest["run_id"], "data_snapshot": snapshot["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
