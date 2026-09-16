#!/usr/bin/env python3
"""OfficeQA corpus tools for the grounded-reasoning tool-agent rollout.

LEAN + LOCAL + self-contained reimplementation of the deep-research OfficeQA
agent's winning toolset (which used a live Databricks SQL warehouse + Vector
Search). We run the SAME semantics over the CLEAN corpus on local NVMe -- no live
warehouse/VS endpoint (that coupling would wreck RL rollout throughput). Tools:

  * search_documents(query, top_k)         -- find the right file: local BM25 over
                                              clean table-aware chunks (hybrid/dense
                                              is a fast-follow). == treasury_search.
  * grep_documents(pattern, file_name,     -- substring (default, case-insensitive)
       year, regex)                           or regex search OVER the files; the
                                              "grep-like SQL search" that beat vector
                                              search. == treasury_grep. Pins an exact
                                              row label / value / category.
  * read_document(file_name, start, n)     -- read ALL / any slice of a file in order
                                              (full-file access). == treasury_file_read.
  * list_documents(year)                   -- enumerate available bulletins.
  * compute(code)                          -- sandboxed Python w/ numpy/pandas (the
                                              "pandas lift"): parse a retrieved table
                                              and compute the answer. == compute.

CORPUS: the CLEAN .txt regenerated from the JSON sources via parse_html_tables
(NOT the pd.read_html-garbled variant). read/grep operate on the .txt files;
search operates on chunks.jsonl built from the same clean .txt.

Contract identical to calc_tool.py: Google-style docstring + type hints -> the
OpenAI tool schema is inferred; a ``str`` return -> a text ToolResponse. Each
tool's LOGIC lives in a plain ``_impl`` function so the eval harness can call the
exact same code the rollout uses.
"""

from __future__ import annotations

import glob
import json
import os
import re

# --- verl decorator, no-op shim so this file is importable/testable standalone.
try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover

    def function_tool(name=None, *, schema=None):
        def deco(fn):
            return fn

        return deco(name) if callable(name) else deco


# Sandboxed Python executor for the compute tool (decorator-free -> no registry
# collision). Dual import path so it resolves whether scripts/ or scripts/tools/
# is on sys.path.
try:
    from tools.py_compute import run_code as _run_code
except Exception:  # noqa: BLE001
    try:
        from py_compute import run_code as _run_code
    except Exception:  # noqa: BLE001
        _run_code = None

# Optional local BM25 (pip install bm25s numpy). Keyword fallback if absent.
try:
    import bm25s

    _HAS_BM25 = True
except Exception:  # noqa: BLE001
    _HAS_BM25 = False


CORPUS_DIR = os.environ.get("OFFICEQA_CORPUS_DIR", "/local_disk0/officeqa_corpus")
CHUNKS_PATH = os.environ.get("OFFICEQA_CHUNKS", "/local_disk0/officeqa/chunks.jsonl")
_TOP_K = int(os.environ.get("OFFICEQA_SEARCH_TOP_K", "6"))
_MAX_CHARS = int(os.environ.get("OFFICEQA_TOOL_MAX_CHARS", "4000"))     # cap each tool response
_READ_MAX_LINES = int(os.environ.get("OFFICEQA_READ_MAX_LINES", "400"))
_SNIPPET_CHARS = int(os.environ.get("OFFICEQA_SNIPPET_CHARS", "500"))
_GREP_MAX_MATCHES = int(os.environ.get("OFFICEQA_GREP_MAX_MATCHES", "40"))

_CHUNKS = None      # list[dict], lazy per-process cache
_BM25 = None        # bm25s retriever (built once), or False sentinel = "use fallback"
_DOC_LINES: dict[str, list[str]] = {}   # lazy per-file line cache (for grep/read)


def _clip(s: str) -> str:
    return s if len(s) <= _MAX_CHARS else s[:_MAX_CHARS] + "\n...[truncated; refine the query / read a slice]"


def _coerce_int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:  # noqa: BLE001
        return default


def _norm_name(file_name: str) -> str:
    name = os.path.basename(str(file_name).strip())   # basename only (no traversal)
    if name and not name.endswith(".txt"):
        name += ".txt"
    return name


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


def _load_doc_lines(fname: str) -> list[str] | None:
    """Read a corpus .txt file's lines, cached per process. None if missing."""
    if fname in _DOC_LINES:
        return _DOC_LINES[fname]
    path = os.path.join(CORPUS_DIR, fname)
    if not os.path.isfile(path):
        _DOC_LINES[fname] = None  # type: ignore[assignment]
        return None
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.read().splitlines()
    except Exception:  # noqa: BLE001
        lines = []
    _DOC_LINES[fname] = lines
    return lines


def _corpus_files(year: str = "") -> list[str]:
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    y = str(year).strip()
    if y:
        names = [n for n in names if f"_{y}_" in n]
    return names


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


# ---------------------------------------------------------------------------
# Plain implementations (verl-free, directly callable by the eval harness).
# ---------------------------------------------------------------------------
def _search_documents(query: str, top_k: int = _TOP_K) -> str:
    chunks = _load_chunks()
    if not chunks:
        return f"Error: chunk index not available at {CHUNKS_PATH}"
    q = str(query).strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters"
    k = max(1, min(_coerce_int(top_k, _TOP_K), 20))

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


def _grep_documents(pattern: str, file_name: str = "", year: str = "",
                    regex: bool = False, max_matches: int = _GREP_MAX_MATCHES) -> str:
    pat = str(pattern)
    if len(pat.strip()) < 2:
        return "Error: pattern must be at least 2 characters"
    use_regex = bool(regex) and str(regex).lower() not in ("false", "0", "no", "")
    if use_regex:
        try:
            rx = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            return f"Error: invalid regex {pat!r}: {e}"
        match = lambda ln: rx.search(ln) is not None  # noqa: E731
    else:
        needle = pat.lower()
        match = lambda ln: needle in ln.lower()       # noqa: E731

    if file_name:
        files = [_norm_name(file_name)]
    else:
        files = _corpus_files(year)
    if not files:
        return f"No documents to search{f' for year {year}' if year else ''}."

    cap = max(1, min(_coerce_int(max_matches, _GREP_MAX_MATCHES), 100))
    hits = []
    scanned = 0
    for fname in files:
        lines = _load_doc_lines(fname)
        if not lines:
            continue
        scanned += 1
        for i, ln in enumerate(lines):
            if match(ln):
                hits.append(f"{fname}:{i}: {ln.strip()}")
                if len(hits) >= cap:
                    break
        if len(hits) >= cap:
            break
    if not hits:
        where = file_name or (f"year {year}" if year else "the corpus")
        return f"No matches for {pattern!r} in {where}."
    head = (f"{len(hits)} match(es) for {pattern!r}"
            f"{' (regex)' if use_regex else ''} in "
            f"{file_name or (f'{scanned} files for {year}' if year else f'{scanned} files')}:")
    return _clip(head + "\n" + "\n".join(hits))


def _read_document(file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    fname = _norm_name(file_name)
    lines = _load_doc_lines(fname)
    if lines is None:
        return f"Error: no document named {fname!r}. Use list_documents(year) or search_documents first."
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
    names = _corpus_files(year)
    y = str(year).strip()
    if not names:
        return f"No documents found{f' for year {y}' if y else ''}."
    return _clip(f"{len(names)} document(s){f' for {y}' if y else ''}:\n" + ", ".join(names[:250]))


def _compute(code: str) -> str:
    if _run_code is None:
        return "Error: compute sandbox unavailable"
    return _run_code(code)


# ---------------------------------------------------------------------------
# verl @function_tool wrappers: docstring + type hints drive the OpenAI schema.
# ---------------------------------------------------------------------------
@function_tool("search_documents")
def search_documents(query: str, top_k: int = _TOP_K) -> str:
    """Search the Treasury Bulletin corpus and return the most relevant passages, each with its source document and bulletin date.

    Use this FIRST to locate the document/table that holds a figure. Each result
    names its source file and YYYY-MM bulletin date. NOTE: Treasury Bulletins
    report data for PRIOR periods, so a calendar-1940 figure typically appears in a
    1940 or 1941 bulletin. Combine a category and a year, e.g. "national defense
    expenditures 1940". Then use grep_documents / read_document on the file.

    Args:
        query: What to look for, e.g. "national defense expenditures 1940".
        top_k: How many passages to return (default 6; max 20).
    """
    return _search_documents(query, top_k)


@function_tool("grep_documents")
def grep_documents(pattern: str, file_name: str = "", year: str = "", regex: bool = False) -> str:
    """Grep for an exact text pattern over the Treasury Bulletin files (like grep).

    The most reliable way to PIN an exact row label, category, or value. Returns
    matching lines as "file:line_number: text". Substring match is default
    (case-insensitive); set regex=true for a Python regular expression. Scope to
    one file (file_name) or one year (year) to cut noise; omit both to search all.

    Args:
        pattern: Text or regex to find, e.g. "National defense" or "2,602".
        file_name: Optional file to search within, e.g. "treasury_bulletin_1941_01.txt".
        year: Optional 4-digit year filter, e.g. "1941" (ignored if file_name is set).
        regex: If true, treat pattern as a regular expression (default false = substring).
    """
    return _grep_documents(pattern, file_name, year, regex)


@function_tool("read_document")
def read_document(file_name: str, start_line: int = 0, num_lines: int = 100) -> str:
    """Read a slice of one Treasury Bulletin document by file name (financial tables are Markdown).

    Use after search_documents / grep_documents to read the relevant table in full.
    Lines are numbered so you can page on by increasing start_line (e.g. jump to a
    line grep_documents reported).

    Args:
        file_name: Document file name, e.g. "treasury_bulletin_1941_01.txt".
        start_line: 0-based line to start from (default 0).
        num_lines: How many lines to return (default 100; capped at 400).
    """
    return _read_document(file_name, start_line, num_lines)


@function_tool("list_documents")
def list_documents(year: str = "") -> str:
    """List available Treasury Bulletin documents, optionally filtered by year.

    Documents are named ``treasury_bulletin_YYYY_MM.txt`` (monthly, 1939 onward).

    Args:
        year: Optional 4-digit year to filter by, e.g. "1941". Empty lists all (capped).
    """
    return _list_documents(year)


@function_tool("compute")
def compute(code: str) -> str:
    """Execute Python code for calculations and table analysis; returns stdout and the last expression's value.

    Use this for ALL non-trivial arithmetic over figures you retrieved (summing the
    monthly cells of a table, differences, ratios, percentage change) so you never
    miscalculate. numpy and pandas are available; e.g. build a list of the values
    you read and print(sum(values)), or load a table into pandas. Each call runs in
    a FRESH sandbox (no variables persist) -- pass self-contained code. Blocked:
    file/network/OS access, import of anything outside math/statistics/numpy/pandas/etc.

    Args:
        code: Self-contained Python, e.g.
            "vals=[132,129,143,159,154,153,177,200,219,287,376,473]\\nprint(sum(vals))".
    """
    return _compute(code)


# ---------------------------------------------------------------------------
# Local sanity check:
#   OFFICEQA_CORPUS_DIR=/tmp/officeqa_corpus_clean OFFICEQA_CHUNKS=/tmp/chunks_clean.jsonl \
#     PYTHONPATH=scripts python3 scripts/tools/officeqa_tools.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[officeqa_tools] bm25s={_HAS_BM25} compute={_run_code is not None} "
          f"CORPUS_DIR={CORPUS_DIR} CHUNKS_PATH={CHUNKS_PATH}")
    print("compute ->", _compute("vals=[132,129,143,159,154,153,177,200,219,287,376,473]\nprint(sum(vals))"))
    print(f"chunks: {len(_load_chunks())}")
    files = _corpus_files()
    if files:
        yr = files[0].split("_")[2]
        print("list ->", _list_documents(yr)[:120])
        print("grep 'National defense' in 1941 ->\n ",
              _grep_documents("National defense", year="1941")[:500].replace("\n", "\n  "))
        print("read ->", _read_document(files[0], 0, 3)[:160])
    if _load_chunks():
        print("search ->", _search_documents("national defense expenditures 1940", 2)[:300])
