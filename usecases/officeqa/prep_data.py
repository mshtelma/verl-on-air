#!/usr/bin/env python3
"""Freeze the staged OfficeQA corpus and Miles' official split as verl parquet."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

from chunker import chunk_file
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


def build_chunks(archive_path: Path, destination: Path) -> tuple[int, int]:
    """Use the original table-aware chunker; never skip a document silently."""
    n_documents = n_chunks = 0
    with tempfile.TemporaryDirectory(prefix="officeqa_chunks_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive, destination.open("w") as output:
            names = sorted(name for name in archive.namelist() if name.endswith(".txt"))
            basenames = [Path(name).name for name in names]
            if len(set(basenames)) != len(basenames):
                raise ValueError("corpus has duplicate document basenames")
            for name in names:
                local = root / Path(name).name
                local.write_bytes(archive.read(name))
                chunks = chunk_file(local)
                n_documents += 1
                for chunk in chunks:
                    if chunk.content.strip():
                        output.write(json.dumps({
                            "content": chunk.content, "source": chunk.file_name,
                            "bulletin_date": chunk.bulletin_date, "page_info": chunk.page_info,
                            "chunk_type": chunk.chunk_type,
                        }) + "\n")
                        n_chunks += 1
                local.unlink()
                if n_documents % 100 == 0:
                    print(f"[officeqa-prep] {n_documents} documents, {n_chunks} chunks", flush=True)
    if not n_documents or not n_chunks:
        raise ValueError("empty OfficeQA corpus/index")
    return n_documents, n_chunks


def main() -> None:
    source = Path(os.environ.get("OQ_SOURCE_DIR", "/Volumes/main/mshtelma/verl/data/officeqa"))
    output = Path(os.environ["OQ_DATA_DIR"])
    turns = int(os.environ.get("OQ_MAX_TURNS", "12"))
    if (output / "DATA_MANIFEST.json").exists():
        raise ValueError(f"snapshot already exists: {output}; choose a new OQ_DATA_DIR")
    required = [source / "officeqa_full.csv", source / "treasury_bulletins_clean.zip"]
    for path in required:
        if not path.is_file():
            available = sorted(p.name for p in source.iterdir()) if source.is_dir() else []
            raise FileNotFoundError(f"missing {path}; available OfficeQA files: {available}")
    if os.environ.get("OQ_REQUIRE_SANDBOX", "1") == "1":
        if not sandbox._sandbox_ok():
            raise RuntimeError("OfficeQA compute needs working bubblewrap/user namespaces on AIR")
        result = sandbox.run_code("print(sum([132,129,143,159,154,153,177,200,219,287,376,473]))")
        if "2602" not in result or result.startswith("Error:"):
            raise RuntimeError(f"compute arithmetic probe failed: {result}")
        print("[officeqa-prep] sandbox arithmetic PASS", flush=True)
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
    with tempfile.TemporaryDirectory(prefix="officeqa_index_") as temporary:
        local_chunks = Path(temporary) / "chunks.jsonl"
        if (source / "chunks.jsonl").is_file():
            shutil.copyfile(source / "chunks.jsonl", local_chunks)
            sources.append({"path": str(source / "chunks.jsonl"), "sha256": dm.sha256_file(local_chunks)})
            documents = None
            with local_chunks.open() as stream:
                chunks = sum(1 for line in stream if line.strip())
        else:
            documents, chunks = build_chunks(required[1], local_chunks)
        chunk_receipt = {"name": "chunks.jsonl", "bytes": local_chunks.stat().st_size, "sha256": dm.sha256_file(local_chunks)}
        copy_atomic(local_chunks, output / "chunks.jsonl")
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
        chunks=chunk_receipt, document_count=documents, chunk_count=chunks,
        maximum_rendered_prompt_tokens=maximum, tokenizer=tokenizer_path, sandbox_probe="PASS",
    )
    print(json.dumps({"status": "PASS", "data_dir": str(output), "rows": {r["name"]: r["rows"] for r in outputs},
                      "max_prompt_tokens": maximum, "chunks": chunks, "manifest": str(output / "DATA_MANIFEST.json"),
                      "run_id": manifest["run_id"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
