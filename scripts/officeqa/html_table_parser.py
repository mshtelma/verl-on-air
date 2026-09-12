"""HTML table → Markdown conversion with exact cell text preservation.

Replaces ``pd.read_html()`` which corrupts data (NaN injection, comma stripping,
float coercion, Unnamed: headers).  Uses stdlib ``html.parser`` only — no pandas
or external dependencies for table parsing.

Public API:
    ``parse_html_tables(html) -> list[str]``  — returns Markdown table strings.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

# ---------------------------------------------------------------------------
# OCR corrections for header text
# ---------------------------------------------------------------------------

_OCR_CORRECTIONS: dict[str, str] = {
    "Receipte": "Receipts",
    "Expendituree": "Expenditures",
    "Expendltures": "Expenditures",
    "Unexpanded": "Unexpended",
}


def _apply_ocr_corrections(text: str) -> str:
    for wrong, right in _OCR_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def _escape_md_cell(text: str) -> str:
    """Escape pipe characters and normalize whitespace for Markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------


class _HTMLTableParser(HTMLParser):
    """Parse ``<table>`` blocks into 2D grids of exact cell text.

    Handles ``rowspan`` / ``colspan`` by maintaining an ``_occupied`` dict
    that tracks which ``(row, col)`` positions are filled.  Cell text is
    preserved exactly as it appears in the HTML source — no type coercion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[list[list[str]], int]] = []

        # Per-table state
        self._occupied: dict[tuple[int, int], str] = {}
        self._header_rows: int = 0
        self._in_table: bool = False
        self._table_depth: int = 0
        self._row_idx: int = -1
        self._col_idx: int = 0

        # Per-cell state
        self._in_cell: bool = False
        self._cell_text: str = ""
        self._cell_is_header: bool = False
        self._cell_rowspan: int = 1
        self._cell_colspan: int = 1

    # -- HTMLParser overrides ------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                # Only parse outermost table
                self._occupied = {}
                self._header_rows = 0
                self._row_idx = -1
                self._in_table = True

        elif tag == "tr" and self._in_table:
            self._row_idx += 1
            self._col_idx = 0

        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_text = ""
            self._cell_is_header = tag == "th"
            self._cell_rowspan = int(attrs_dict.get("rowspan") or 1)
            self._cell_colspan = int(attrs_dict.get("colspan") or 1)
            # Advance past positions already occupied by rowspan/colspan
            while (self._row_idx, self._col_idx) in self._occupied:
                self._col_idx += 1

        elif tag == "br" and self._in_cell:
            self._cell_text += " "

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            text = self._cell_text.strip()

            # Fill occupied positions for rowspan/colspan
            for r in range(self._cell_rowspan):
                for c in range(self._cell_colspan):
                    pos = (self._row_idx + r, self._col_idx + c)
                    # Origin cell gets the text; spanned positions get ""
                    self._occupied[pos] = text if (r == 0 and c == 0) else ""

            if self._cell_is_header:
                self._header_rows = max(self._header_rows, self._row_idx + 1)

            self._col_idx += self._cell_colspan

        elif tag == "table":
            if self._table_depth == 1 and self._in_table:
                self._in_table = False
                grid = self._build_grid()
                if grid:
                    self.tables.append((grid, self._header_rows))
            self._table_depth = max(0, self._table_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text += data

    def handle_entityref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text += unescape(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._in_cell:
            self._cell_text += unescape(f"&#{name};")

    # -- Grid construction ---------------------------------------------------

    def _build_grid(self) -> list[list[str]]:
        if not self._occupied:
            return []
        max_row = max(r for r, _ in self._occupied)
        max_col = max(c for _, c in self._occupied)
        return [
            [self._occupied.get((r, c), "") for c in range(max_col + 1)]
            for r in range(max_row + 1)
        ]


# ---------------------------------------------------------------------------
# Grid → Markdown conversion
# ---------------------------------------------------------------------------


def _grid_to_markdown(grid: list[list[str]], header_rows: int) -> str:
    """Convert a 2D grid of cell text into a Markdown table string."""
    if not grid:
        return ""

    num_cols = max(len(row) for row in grid)

    # Pad rows to uniform column count
    for row in grid:
        while len(row) < num_cols:
            row.append("")

    # --- Build headers ---
    if header_rows == 0:
        # No <th> found — generate positional headers
        headers = ["Category"] + [f"col_{i + 1}" for i in range(1, num_cols)]
        data_rows = grid
    elif header_rows == 1:
        headers = list(grid[0])
        data_rows = grid[1:]
    else:
        # Multi-row headers: flatten by column
        headers = []
        for col_idx in range(num_cols):
            parts: list[str] = []
            for row_idx in range(header_rows):
                val = grid[row_idx][col_idx] if col_idx < len(grid[row_idx]) else ""
                # Deduplicate consecutive identical values (e.g., Year > Year → Year)
                if val and (not parts or parts[-1] != val):
                    parts.append(val)
            headers.append(" > ".join(parts) if parts else "")
        data_rows = grid[header_rows:]

    # --- Clean headers ---
    headers = [_apply_ocr_corrections(h) for h in headers]

    # First-column heuristic: if empty, label as "Category"
    if headers and not headers[0]:
        headers[0] = "Category"

    # Ensure unique, non-empty header names
    seen: dict[str, int] = {}
    for i, h in enumerate(headers):
        name = h if h else f"col_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        headers[i] = name

    # --- Build Markdown ---
    lines: list[str] = []
    lines.append("| " + " | ".join(_escape_md_cell(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in data_rows:
        cells = [_escape_md_cell(row[i] if i < len(row) else "") for i in range(len(headers))]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_html_tables(html: str) -> list[str]:
    """Parse HTML containing ``<table>`` elements into Markdown table strings.

    Preserves exact cell text — no NaN injection, no comma stripping, no float
    coercion.  Handles ``rowspan`` and ``colspan``.

    Parameters
    ----------
    html:
        HTML string potentially containing one or more ``<table>`` blocks.

    Returns
    -------
    list[str]:
        One Markdown table string per ``<table>`` found.  If parsing fails,
        returns the HTML wrapped in a code fence as fallback.
    """
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
    except Exception:
        return ["```html\n" + html.strip() + "\n```"]

    # Flush any in-progress table — handles truncated HTML missing </table>.
    # Many upstream JSON elements have HTML that ends mid-row without closing.
    if parser._in_table and parser._occupied:
        grid = parser._build_grid()
        if grid:
            parser.tables.append((grid, parser._header_rows))
        parser._in_table = False

    if not parser.tables:
        return ["```html\n" + html.strip() + "\n```"]

    return [_grid_to_markdown(grid, hrows) for grid, hrows in parser.tables]
