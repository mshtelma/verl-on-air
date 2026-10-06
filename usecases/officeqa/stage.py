#!/usr/bin/env python3
"""Verify and stage the immutable corpus, then check actual compute/retrieval readiness."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "lib"))
import data_manifest as dm  # noqa: E402

import corpus
import sandbox


def main() -> None:
    if int(os.environ.get("POD_RANK", "0")) >= int(os.environ.get("TRAINING_NODES", "1")):
        return
    if os.environ.get("OQ_REQUIRE_BM25", "0") == "1" and corpus.bm25s is None:
        raise RuntimeError("OfficeQA requires BM25 retrieval; this image has no bm25s")
    source = Path(os.environ["OQ_DATA_DIR"])
    target = Path(os.environ["OQ_LOCAL_DIR"])
    manifest = json.loads((source / "DATA_MANIFEST.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    for receipt in manifest["outputs"]:
        if dm.sha256_file(source / receipt["name"]) != receipt["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {receipt['name']}")
    for receipt in [manifest["corpus"], manifest["chunks"]]:
        name = receipt["name"]
        if name not in {"treasury_bulletins_clean.zip", "chunks.jsonl"}:
            raise ValueError(f"unexpected corpus member {name!r}")
        destination = target / name
        if not destination.exists() or dm.sha256_file(destination) != receipt["sha256"]:
            temporary = destination.with_name(f".{name}.partial")
            shutil.copyfile(source / name, temporary)
            if dm.sha256_file(temporary) != receipt["sha256"]:
                raise ValueError(f"corpus checksum mismatch: {name}")
            temporary.replace(destination)
    result = sandbox.run_code("print(100 + 200)")
    if result.strip() != "Output:\n300":
        raise RuntimeError(f"OfficeQA compute preflight failed: {result}")
    loaded = corpus.load_corpus(target)
    if not loaded.documents or not loaded.chunks:
        raise RuntimeError("empty corpus/index")
    search = corpus.search_documents(loaded, "national defense expenditures", 2)
    if search.startswith(("Error:", "No ")):
        raise RuntimeError(f"retrieval preflight failed: {search}")
    report = {"officeqa_stage": "PASS", "documents": len(loaded.documents),
              "chunks": len(loaded.chunks), "retrieval_backend": "bm25s" if loaded.retriever else "keyword_overlap",
              "compute": "PASS", "source": str(source), "local": str(target),
              "corpus_sha256": manifest["corpus"]["sha256"], "chunks_sha256": manifest["chunks"]["sha256"],
              "run_id": os.environ.get("RUN_ID"), "pod_rank": os.environ.get("POD_RANK", "0")}
    if os.environ.get("OQ_STAGE_OUT"):
        path = Path(os.environ["OQ_STAGE_OUT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
