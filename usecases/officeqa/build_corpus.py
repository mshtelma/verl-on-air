#!/usr/bin/env python3
"""Rebuild OfficeQA's measured document archive and BM25 chunks.

Table cells stay strings: commas, empty cells and decimal precision survive
conversion. The archive uses fixed timestamps; corpus identity depends on the
sorted document names and text hashes, independently of ZIP metadata. The
measured BM25 index uses the creators' separately published transformed text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

from html_table_parser import parse_html_tables
from chunker import chunk_file

HERE = Path(__file__).resolve().parent
LOCK_PATH = HERE / "DATA.lock.json"
PARSED_DIR = "treasury_bulletins_parsed/jsons"
TRANSFORMED_DIR = "treasury_bulletins_parsed/transformed"
ARCHIVE_NAME = "treasury_bulletins_clean.zip"


def render_document(document: dict) -> bytes:
    """Keep the creators' element order and text layout, with lossless tables."""
    elements = (document.get("document") or {}).get("elements")
    if not isinstance(elements, list):
        raise ValueError("parsed document has no elements list")
    lines = []
    for element in elements:
        content = element.get("content") if isinstance(element, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        if "<table" in content.lower():
            for table in parse_html_tables(content):
                lines.extend([table, ""])
        else:
            lines.extend(re.sub(r"\r\n?", "\n", content.strip()).split("\n"))
            lines.append("")
    if not lines:
        raise ValueError("parsed document has no usable text")
    return "".join(line + "\n" for line in lines).encode("utf-8")


def archive_identity(archive_path: Path) -> dict:
    """Hash UTF-8 lines of '<basename>\\0<text SHA256>\\n', sorted by basename."""
    digest = hashlib.sha256()
    text_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted((Path(name).name, name) for name in archive.namelist() if name.endswith(".txt"))
        if not members or len({base for base, _ in members}) != len(members):
            raise ValueError("corpus is empty or has duplicate document basenames")
        for basename, member in members:
            content = archive.read(member)
            text_bytes += len(content)
            digest.update(f"{basename}\0{hashlib.sha256(content).hexdigest()}\n".encode("utf-8"))
    return {"documents": len(members), "text_bytes": text_bytes, "content_sha256": digest.hexdigest()}


def build_archive(parsed_dir: Path, destination: Path) -> dict:
    files = sorted(parsed_dir.glob("treasury_bulletin_*.json"))
    if not files:
        raise FileNotFoundError(f"no parsed Treasury Bulletins in {parsed_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="officeqa_corpus_", dir=destination.parent) as temporary:
        local = Path(temporary) / ARCHIVE_NAME
        with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for count, source in enumerate(files, 1):
                content = render_document(json.loads(source.read_text(encoding="utf-8")))
                member = zipfile.ZipInfo(source.with_suffix(".txt").name, date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                member.external_attr = 0o644 << 16
                archive.writestr(member, content, compresslevel=6)
                if count % 100 == 0:
                    print(f"[officeqa-corpus] rendered {count}/{len(files)} documents", flush=True)
        identity = archive_identity(local)
        local.replace(destination)
    return identity


def check_identity(label: str, actual: dict, expected: dict) -> None:
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"{label} differs from the measured snapshot ({', '.join(mismatches)}); "
                         "rebuild with make officeqa-corpus using the pinned sources")


def file_identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def build_chunks(text_dir: Path, destination: Path) -> dict:
    """Preserve the measured index's rendering, order and JSON serialization."""
    files = sorted(text_dir.glob("treasury_bulletin_*.txt"))
    if not files:
        raise FileNotFoundError(f"no transformed Treasury Bulletins in {text_dir}")
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for documents, source in enumerate(files, 1):
            for chunk in chunk_file(source):
                if chunk.content.strip():
                    output.write(json.dumps({
                        "content": chunk.content, "source": chunk.file_name,
                        "bulletin_date": chunk.bulletin_date, "page_info": chunk.page_info,
                        "chunk_type": chunk.chunk_type,
                    }) + "\n")
                    count += 1
            if documents % 100 == 0:
                print(f"[officeqa-corpus] indexed {documents} documents, {count} chunks", flush=True)
    if not count:
        raise ValueError("empty OfficeQA retrieval index")
    return {**file_identity(destination), "count": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Hugging Face dataset directory")
    parser.add_argument("--download", action="store_true", help="download pinned JSONs, transformed text and CSV using Hugging Face authentication")
    args = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text())
    source = lock["source"]
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(source["repository"], repo_type="dataset", revision=source["revision"],
                          allow_patterns=["officeqa_full.csv", f"{PARSED_DIR}/*.json", f"{TRANSFORMED_DIR}/*.txt"],
                          local_dir=args.source_dir)
    check_identity("question CSV", file_identity(args.source_dir / "officeqa_full.csv"), lock["question_csv"])
    identity = build_archive(args.source_dir / PARSED_DIR, args.source_dir / ARCHIVE_NAME)
    check_identity("rendered corpus", identity, lock["corpus"])
    with tempfile.TemporaryDirectory(prefix="officeqa_index_", dir=args.source_dir) as temporary:
        chunks_path = Path(temporary) / "chunks.jsonl"
        chunks = build_chunks(args.source_dir / TRANSFORMED_DIR, chunks_path)
        check_identity("retrieval chunks", chunks, lock["chunks"])
        chunks_path.replace(args.source_dir / "chunks.jsonl")
    print(json.dumps({"status": "MATCH", "archive": str(args.source_dir / ARCHIVE_NAME),
                      "source": source, "corpus": identity, "chunks": chunks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
