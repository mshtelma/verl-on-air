#!/usr/bin/env python3
"""OfficeQA corpus tools for the grounded-reasoning tool-agent rollout.

LEAN + LOCAL + stateless ``@function_tool`` callables over the Treasury Bulletin
corpus. Retrieval is **local BM25** (``bm25s``) over table-aware chunks -- no
vector DB, no embedding model, no Databricks service. Reuses the deep-research
repo's ``FileSearchTool`` ranking idea + its ``chunker.py``, but builds the BM25
index ONCE per worker (the original re-indexed on every query). Graceful
keyword-overlap fallback if ``bm25s`` is not installed.

Data (staged on a UC Volume; the job copies/unzips to local NVMe):
  * OFFICEQA_CHUNKS      chunks.jsonl -- one JSON/line: {content, source,
                         bulletin_date, page_info, chunk_type}. ``content`` is
                         self-identifying ("Document:<file>|Bulletin date:YYYY-MM|...").
  * OFFICEQA_CORPUS_DIR  the 697 ``treasury_bulletin_YYYY_MM.txt`` docs (read_document).

Tools (one file -> several @function_tool defs; all offered to every tool_agent
sample via ``rollout.multi_turn.function_tool_path``):
  * search_documents(query, top_k) -- BM25 top-k passages (source + date + snippet).
  * read_document(file_name, start_line, num_lines) -- read a slice of one doc.
  * list_documents(year) -- list available bulletins.
  * calculator(expression) -- safe arithmetic over retrieved figures.

Contract identical to calc_tool.py: Google-style docstring + type hints -> the
OpenAI tool schema is inferred; a ``str`` return -> a text ToolResponse. Each
tool's LOGIC lives in a plain ``_impl`` function (verl-free, directly callable)
so the eval harness (scripts/eval_officeqa_agentic.py) can call the exact same
code the rollout uses without depending on what verl's decorator returns.
"""

from __future__ import annotations

import glob
import json
import os

# --- verl decorator, no-op shim so this file is importable/testable standalone.
try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover

    def function_tool(name=None, *, schema=None):
        def deco(fn):
            return fn

        return deco(name) if callable(name) else deco


# Reuse the safe calculator core from the DECORATOR-FREE module. Importing
# calc_tool here would run its @function_tool("calculator") and then collide with
# our own calculator registration (verl refuses a duplicate tool name in one
# process) -- so we import the pure evaluator from safe_eval instead.
try:
    from tools.safe_eval import evaluate as _calc_evaluate
except Exception:  # noqa: BLE001
    try:
        from safe_eval import evaluate as _calc_evaluate
    except Exception:  # noqa: BLE001
        _calc_evaluate = None

# Optional local BM25 (pip install bm25s numpy). Keyword fallback if absent.
try:
    import bm25s

    _HAS_BM25 = True
except Exception:  # noqa: BLE001
    _HAS_BM25 = False


CORPUS_DIR = os.environ.get("OFFICEQA_CORPUS_DIR", "/local_disk0/officeqa_corpus")
CHUNKS_PATH = os.environ.get("OFFICEQA_CHUNKS", "/local_disk0/officeqa/chunks.jsonl")
_TOP_K = int(os.environ.get("OFFICEQA_SEARCH_TOP_K", "6"))
_MAX_CHARS = int(os.environ.get("OFFICEQA_TOOL_MAX_CHARS", "3500"))     # cap each tool response
_READ_MAX_LINES = int(os.environ.get("OFFICEQA_READ_MAX_LINES", "400"))
_SNIPPET_CHARS = int(os.environ.get("OFFICEQA_SNIPPET_CHARS", "500"))

_CHUNKS = None   # list[dict], lazy per-process cache
_BM25 = None     # bm25s retriever (built once), or False sentinel = "use fallback"


def _clip(s: str) -> str:
    return s if len(s) <= _MAX_CHARS else s[:_MAX_CHARS] + "\n...[truncated; refine the query or read a slice]"


def _load_chunks():
    global _CHUNKS
    if _CHUNKS is not None:
        return _CHUNKS
    chunks = []
    if os.path.isfile(CHUNKS_PATH):
        with open(CHUNKS_PATH, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    _CHUNKS = chunks
    return _CHUNKS


def _get_bm25():
    """Build the BM25 index ONCE per worker and cache it (retriever, or False)."""
    global _BM25
    if _BM25 is not None:
        return _BM25
    chunks = _load_chunks()
    if not chunks or not _HAS_BM25:
        _BM25 = False
        return _BM25
    corpus = [c.get("content", "") for c in chunks]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    _BM25 = retriever
    return _BM25


def _highlight(content: str, query: str) -> str:
    cl = content.lower()
    pos = -1
    for t in (t for t in query.lower().split() if len(t) >= 3):
        p = cl.find(t)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return content[:_SNIPPET_CHARS] + ("..." if len(content) > _SNIPPET_CHARS else "")
    s = max(0, pos - _SNIPPET_CHARS // 3)
    e = min(len(content), pos + (2 * _SNIPPET_CHARS) // 3)
    return ("..." if s > 0 else "") + content[s:e] + ("..." if e < len(content) else "")


def _norm_name(file_name: str) -> str:
    name = os.path.basename(str(file_name).strip())   # basename only (no traversal)
    if name and not name.endswith(".txt"):
        name += ".txt"
    return name


def _coerce_int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Plain implementations -- verl-free, directly callable (imported by the eval
# harness). The @function_tool wrappers below just carry the schema docstring.
# ---------------------------------------------------------------------------
def _search_documents(query: str, top_k: int = _TOP_K) -> str:
    chunks = _load_chunks()
    if not chunks:
        return f"Error: chunk index not available at {CHUNKS_PATH}"
    q = str(query).strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters"
    k = _coerce_int(top_k, _TOP_K)
    k = max(1, min(k, 20))

    ranked = []   # list[(score, chunk)]
    retriever = _get_bm25()
    if retriever:
        idxs, scores = retriever.retrieve(
            bm25s.tokenize([q], show_progress=False), k=min(k, len(chunks)), show_progress=False
        )
        for idx, sc in zip(idxs[0], scores[0]):
            if sc > 0:
                ranked.append((float(sc), chunks[int(idx)]))
    else:
        qt = {t for t in q.lower().split() if len(t) >= 3}
        if qt:
            for c in chunks:
                txt = c.get("content", "").lower()
                m = sum(1 for t in qt if t in txt)
                if m:
                    ranked.append((m / len(qt), c))
            ranked.sort(key=lambda x: x[0], reverse=True)
            ranked = ranked[:k]

    if not ranked:
        return f"No passages found for {query!r}. Try different terms or a specific year."
    out = []
    for i, (sc, c) in enumerate(ranked):
        src = c.get("source", "?")
        date = c.get("bulletin_date", "")
        out.append(f"[{i}] {src}{f' ({date})' if date else ''}  score={sc:.2f}\n    {_highlight(c.get('content', ''), q)}")
    return _clip("\n\n".join(out))


def _read_document(file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    fname = _norm_name(file_name)
    path = os.path.join(CORPUS_DIR, fname)
    if not os.path.isfile(path):
        return f"Error: no document named {fname!r}. Use list_documents(year) or search_documents first."
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.read().splitlines()
    except Exception as e:  # noqa: BLE001
        return f"Error reading {fname}: {e}"
    start = max(0, _coerce_int(start_line, 0))
    n = max(1, min(_coerce_int(num_lines, 100), _READ_MAX_LINES))
    chunk = lines[start:start + n]
    if not chunk:
        return f"{fname}: start_line {start} is past end ({len(lines)} lines)."
    body = "\n".join(f"{start + i}: {ln}" for i, ln in enumerate(chunk))
    return _clip(f"{fname} lines {start}-{start + len(chunk) - 1} of {len(lines)}:\n{body}")


def _list_documents(year: str = "") -> str:
    if not os.path.isdir(CORPUS_DIR):
        return f"Error: corpus dir not available at {CORPUS_DIR}"
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    y = str(year).strip()
    if y:
        names = [n for n in names if f"_{y}_" in n]
    if not names:
        return f"No documents found{f' for year {y}' if y else ''}."
    return _clip(f"{len(names)} document(s){f' for {y}' if y else ''}:\n" + ", ".join(names[:250]))


def _calculator(expression: str) -> str:
    if _calc_evaluate is None:
        return "Error: calculator core unavailable"
    return _calc_evaluate(str(expression))


# ---------------------------------------------------------------------------
# verl @function_tool wrappers: the docstring + type hints drive the inferred
# OpenAI schema; each delegates to its plain impl above.
# ---------------------------------------------------------------------------
@function_tool("search_documents")
def search_documents(query: str, top_k: int = _TOP_K) -> str:
    """Search the Treasury Bulletin corpus and return the most relevant passages, each with its source document and bulletin date.

    This is the primary way to FIND figures: passages (financial tables / text)
    are ranked by relevance to your query. Each returned passage already names
    its source document and YYYY-MM date; use read_document to read more of a
    promising document. Prefer specific queries combining a category and a year,
    e.g. "national defense expenditures 1940".

    Args:
        query: What to look for, e.g. "national defense expenditures 1940".
        top_k: How many passages to return (default 6; max 20).
    """
    return _search_documents(query, top_k)


@function_tool("read_document")
def read_document(file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    """Read a slice of one Treasury Bulletin document by file name.

    Use after search_documents / list_documents to read the relevant section in
    full (financial tables are Markdown). Lines are numbered so you can page on
    by increasing start_line.

    Args:
        file_name: Document file name, e.g. "treasury_bulletin_1940_06.txt".
        start_line: 0-based line to start from (default 0).
        num_lines: How many lines to return (default 100; capped at 400).
    """
    return _read_document(file_name, start_line, num_lines)


@function_tool("list_documents")
def list_documents(year: str = "") -> str:
    """List available Treasury Bulletin documents, optionally filtered by year.

    Documents are named ``treasury_bulletin_YYYY_MM.txt`` (monthly, 1939 onward).

    Args:
        year: Optional 4-digit year to filter by, e.g. "1940". Empty lists all (capped).
    """
    return _list_documents(year)


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact numeric result.

    Use this for ANY non-trivial arithmetic over figures you retrieved (sums,
    differences, ratios, percentages) so you never miscalculate. Supported:
    + - * / // % ** and parentheses; sqrt, floor, ceil, log, log2, log10, exp,
    abs, round, min, max, gcd, factorial, fabs; and pi, e, tau. Pass a single
    arithmetic expression (no variables/assignments).

    Args:
        expression: One arithmetic expression, e.g. "2602 + 44463" or
            "round(100 * 2602 / 44463, 2)".
    """
    return _calculator(expression)


# ---------------------------------------------------------------------------
# Local sanity check (BM25 validated in-job where bm25s is installed):
#   OFFICEQA_CORPUS_DIR=/tmp/officeqa_corpus OFFICEQA_CHUNKS=/tmp/officeqa_chunks.jsonl \
#     PYTHONPATH=scripts python3 scripts/tools/officeqa_tools.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[officeqa_tools] bm25s={_HAS_BM25}  CORPUS_DIR={CORPUS_DIR}  CHUNKS_PATH={CHUNKS_PATH}")
    print("calc 2602 + 44463 ->", calculator("2602 + 44463"))
    chunks = _load_chunks()
    print(f"chunks loaded: {len(chunks)}")
    if os.path.isdir(CORPUS_DIR):
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
        if names:
            yr = names[0].split("_")[2]
            print("list_documents(year) ->", list_documents(yr)[:160])
            print(f"read_document({names[0]!r}, 0, 4) ->", read_document(names[0], 0, 4)[:200])
    if chunks:
        print("search_documents('national defense expenditures 1940') ->\n",
              search_documents("national defense expenditures 1940", 3)[:600])
