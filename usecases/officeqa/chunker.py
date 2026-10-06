"""Table-aware Markdown chunking for Treasury Bulletin text files.

The OfficeQA `transform_parsed_files.py` produces files named
`treasury_bulletin_{YEAR}_{MONTH}.txt` with:
- Plain text paragraphs (separated by blank lines)
- Markdown tables with `|` delimiters (header + `|---|---|` separator + rows)
- Multi-level headers flattened with " > " separator
- Escaped pipes in cells (`\\|`)
- Occasional HTML fallback blocks in ```html``` fences
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Chunking parameters."""

    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 200
    table_max_chars: int = 4000
    section_max_chars: int = 8000  # heading + table + footnotes grouped as one chunk


@dataclass
class Chunk:
    """A single chunk of text or table content."""

    chunk_id: str = ""
    file_name: str = ""
    bulletin_date: str = ""  # "YYYY-MM" from filename
    page_info: str = ""  # nearest heading before this chunk
    content: str = ""
    chunk_type: str = "text"  # "text" | "table" | "html_fallback"
    char_count: int = 0


def _is_table_line(line: str) -> bool:
    """A line is a table line if it starts with '|' and contains 2+ pipes."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return stripped.count("|") >= 2


def _extract_date(filename: str) -> str:
    """Extract YYYY-MM from filename like treasury_bulletin_2024_03.txt."""
    match = re.search(r"(\d{4})_(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    logger.warning("CHUNKER_NO_DATE filename=%s", filename)
    return ""


def _split_into_blocks(
    text: str,
) -> list[tuple[str, str, str]]:
    """Split text into (block_type, block_text, heading_context) tuples.

    Table detection: a line is a table line if it starts with '|' after
    stripping whitespace AND contains at least 2 pipe characters.
    A table block is a maximal run of consecutive table lines (blank lines
    between table rows tolerated up to 1).

    HTML fallback: ```html ... ``` fenced blocks.

    Heading tracking: lines starting with '#' update the current heading context.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str, str]] = []
    current_heading = ""
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track headings
        if line.strip().startswith("#"):
            current_heading = line.strip().lstrip("#").strip()
            i += 1
            continue

        # HTML fallback block
        if line.strip().startswith("```html"):
            html_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                html_lines.append(lines[i])
                i += 1
            if i < len(lines):
                html_lines.append(lines[i])
                i += 1
            blocks.append(("html_fallback", "\n".join(html_lines), current_heading))
            continue

        # Table block
        if _is_table_line(line):
            table_lines = [line]
            i += 1
            while i < len(lines):
                if _is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                elif lines[i].strip() == "" and i + 1 < len(lines) and _is_table_line(
                    lines[i + 1]
                ):
                    # Tolerate 1 blank line within a table
                    table_lines.append(lines[i])
                    i += 1
                else:
                    break
            blocks.append(("table", "\n".join(table_lines), current_heading))
            continue

        # Text line — accumulate until next non-text block
        text_lines = []
        while i < len(lines):
            if lines[i].strip().startswith("#"):
                break
            if _is_table_line(lines[i]):
                break
            if lines[i].strip().startswith("```html"):
                break
            text_lines.append(lines[i])
            i += 1

        text_block = "\n".join(text_lines).strip()
        if text_block:
            blocks.append(("text", text_block, current_heading))

    return blocks


def _chunk_table(
    table_text: str, heading: str, config: ChunkConfig
) -> list[Chunk]:
    """Keep tables whole if possible. Split large tables by rows, preserving header."""
    if len(table_text) <= config.table_max_chars:
        return [
            Chunk(
                page_info=heading,
                content=table_text,
                chunk_type="table",
                char_count=len(table_text),
            )
        ]

    # Split: extract header (line 0) + separator (line 1), split rows
    lines = table_text.split("\n")
    header_lines: list[str] = []
    data_lines: list[str] = []

    # Find header + separator
    for j, line in enumerate(lines):
        if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
            header_lines = lines[: j + 1]
            data_lines = lines[j + 1 :]
            break
    else:
        # No separator found — treat first line as header
        header_lines = lines[:1]
        data_lines = lines[1:]

    header_text = "\n".join(header_lines)
    header_len = len(header_text) + 1  # +1 for newline

    chunks: list[Chunk] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in data_lines:
        row_len = len(row) + 1
        if current_len + row_len > config.table_max_chars and current_rows:
            chunk_content = header_text + "\n" + "\n".join(current_rows)
            chunks.append(
                Chunk(
                    page_info=heading,
                    content=chunk_content,
                    chunk_type="table",
                    char_count=len(chunk_content),
                )
            )
            current_rows = []
            current_len = header_len

        current_rows.append(row)
        current_len += row_len

    # Flush remaining rows
    if current_rows:
        chunk_content = header_text + "\n" + "\n".join(current_rows)
        chunks.append(
            Chunk(
                page_info=heading,
                content=chunk_content,
                chunk_type="table",
                char_count=len(chunk_content),
            )
        )

    if not chunks:
        # Edge case: single row larger than table_max_chars — emit as-is
        logger.warning(
            "CHUNKER_OVERSIZED_TABLE chars=%d heading=%s", len(table_text), heading
        )
        chunks.append(
            Chunk(
                page_info=heading,
                content=table_text,
                chunk_type="table",
                char_count=len(table_text),
            )
        )

    return chunks


def _chunk_text(
    text: str, heading: str, config: ChunkConfig
) -> list[Chunk]:
    """Split at paragraph boundaries with overlap."""
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len + 2 > config.chunk_max_chars and current_parts:
            chunk_content = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    page_info=heading,
                    content=chunk_content,
                    chunk_type="text",
                    char_count=len(chunk_content),
                )
            )
            # Overlap: carry the last part if it fits in overlap budget
            if config.chunk_overlap_chars > 0 and current_parts:
                last = current_parts[-1]
                if len(last) <= config.chunk_overlap_chars:
                    current_parts = [last]
                    current_len = len(last)
                else:
                    current_parts = []
                    current_len = 0
            else:
                current_parts = []
                current_len = 0

        current_parts.append(para)
        current_len += para_len + 2  # +2 for "\n\n" separator

    # Flush remaining
    if current_parts:
        chunk_content = "\n\n".join(current_parts)
        chunks.append(
            Chunk(
                page_info=heading,
                content=chunk_content,
                chunk_type="text",
                char_count=len(chunk_content),
            )
        )

    return chunks


def _merge_into_sections(
    blocks: list[tuple[str, str, str]],
    config: ChunkConfig,
) -> list[tuple[str, str, str]]:
    """Merge consecutive text→table→text blocks into section blocks.

    A *section* is a heading/description text block, followed by a table, and
    optionally followed by footnote/source text.  When the combined size fits
    within ``config.section_max_chars`` the three are merged into a single
    ``("section", combined_text, heading)`` block.  If the combined size
    exceeds the limit the blocks are returned individually (fallback to
    per-block chunking).
    """
    merged: list[tuple[str, str, str]] = []
    i = 0
    while i < len(blocks):
        btype, btext, heading = blocks[i]

        if btype == "text" and i + 1 < len(blocks) and blocks[i + 1][0] == "table":
            # Candidate: text_before + table (+ optional text_after)
            text_before = btext
            table_block = blocks[i + 1]
            text_after = ""
            consumed = 2  # text_before + table

            # Look ahead for a trailing text block (footnotes/source line)
            if (
                i + 2 < len(blocks)
                and blocks[i + 2][0] == "text"
                # Don't consume if the next block is very large (probably a
                # separate narrative section, not a footnote)
                and len(blocks[i + 2][1]) < 1000
            ):
                text_after = blocks[i + 2][1]
                consumed = 3

            combined = "\n\n".join(
                part for part in [text_before, table_block[1], text_after] if part
            )
            if len(combined) <= config.section_max_chars:
                merged.append(("section", combined, heading or table_block[2]))
                i += consumed
                continue

        # Default — pass through unchanged
        merged.append((btype, btext, heading))
        i += 1

    return merged


def _prepend_context(content: str, file_name: str, bulletin_date: str, heading: str) -> str:
    """Prepend document-level context to chunk content for better embeddings.

    Per OfficeQA paper (arxiv 2603.08655, Section D.4) contextual embeddings
    that include document name, date, and section title yield +21% accuracy.
    """
    parts: list[str] = []
    if file_name:
        parts.append(f"Document: {file_name}")
    if bulletin_date:
        parts.append(f"Bulletin date: {bulletin_date}")
    if heading:
        parts.append(f"Section: {heading}")
    if parts:
        return "\n".join(parts) + "\n\n" + content
    return content


def chunk_file(file_path: Path, config: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a single Treasury Bulletin text file.

    Parameters
    ----------
    file_path:
        Path to the .txt file.
    config:
        Chunking parameters. Uses defaults if None.

    Returns
    -------
    list[Chunk]:
        Chunks with IDs, metadata, and content.
    """
    if config is None:
        config = ChunkConfig()

    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        logger.warning("CHUNKER_EMPTY_FILE path=%s", file_path)
        return []

    blocks = _split_into_blocks(text)

    # Merge heading + table + footnote blocks into sections when they fit
    blocks = _merge_into_sections(blocks, config)

    bulletin_date = _extract_date(file_path.name)
    chunks: list[Chunk] = []

    for block_type, block_text, heading_context in blocks:
        if block_type == "section":
            # Already merged — emit as a single chunk with contextual prefix
            content = _prepend_context(
                block_text, file_path.name, bulletin_date, heading_context
            )
            chunks.append(
                Chunk(
                    page_info=heading_context,
                    content=content,
                    chunk_type="section",
                    char_count=len(content),
                )
            )
        elif block_type == "table":
            table_chunks = _chunk_table(block_text, heading_context, config)
            for c in table_chunks:
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(table_chunks)
        elif block_type == "html_fallback":
            html_chunks = _chunk_text(block_text, heading_context, config)
            for c in html_chunks:
                c.chunk_type = "html_fallback"
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(html_chunks)
        else:
            text_chunks = _chunk_text(block_text, heading_context, config)
            for c in text_chunks:
                c.content = _prepend_context(
                    c.content, file_path.name, bulletin_date, heading_context
                )
                c.char_count = len(c.content)
            chunks.extend(text_chunks)

    # Assign IDs and metadata
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{file_path.stem}_c{i:04d}"
        chunk.file_name = file_path.name
        chunk.bulletin_date = bulletin_date

    logger.info(
        "CHUNKER_DONE file=%s chunks=%d total_chars=%d",
        file_path.name,
        len(chunks),
        sum(c.char_count for c in chunks),
    )
    return chunks
