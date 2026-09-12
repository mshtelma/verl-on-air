#!/usr/bin/env python3
"""Chunk the OfficeQA Treasury corpus into chunks.jsonl for LOCAL BM25 retrieval.

Reuses the table-aware chunker (scripts/officeqa/chunker.py, lifted from
databricks-deep-research). Each output line is one chunk:
    {"content", "source", "bulletin_date", "page_info", "chunk_type"}
`content` already has the source file + bulletin date prepended by the chunker,
so a retrieved chunk is self-identifying. This runs offline (CPU) once; the tool
(scripts/tools/officeqa_tools.py) builds the BM25 index from chunks.jsonl once
per worker.

  OFFICEQA_CORPUS_DIR=/path/to/txts python3 scripts/officeqa/build_chunks.py --out chunks.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from chunker import chunk_file  # sibling module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default=os.environ.get("OFFICEQA_CORPUS_DIR", "/tmp/officeqa_corpus"))
    ap.add_argument("--out", default=os.environ.get("OFFICEQA_CHUNKS", "chunks.jsonl"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt")))
    n_chunks = 0
    with open(args.out, "w") as fh:
        for f in files:
            try:
                chunks = chunk_file(Path(f))
            except Exception as e:  # noqa: BLE001 - skip a bad doc, keep the rest
                print(f"[chunks] WARN {os.path.basename(f)}: {type(e).__name__}: {e}", flush=True)
                continue
            for c in chunks:
                if not c.content.strip():
                    continue
                fh.write(json.dumps({
                    "content": c.content,
                    "source": c.file_name or os.path.basename(f),
                    "bulletin_date": c.bulletin_date,
                    "page_info": c.page_info,
                    "chunk_type": c.chunk_type,
                }) + "\n")
                n_chunks += 1
    print(f"wrote {n_chunks} chunks from {len(files)} docs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
