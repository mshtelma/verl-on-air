#!/usr/bin/env python3
"""Agentic search/RAG tools over a Databricks Vector Search index (NQ + HotpotQA, Search-R1 style).

The model orchestrates a small retrieval toolset to answer open-domain / multi-hop questions and
commits its final answer in ``<answer> ... </answer>`` (scored by usecases/agentic-search/reward.py):

  * vector_search(query, top_k)   -- SEMANTIC (ANN) retrieval: best for conceptual / paraphrased
                                     queries.
  * keyword_search(query, top_k)  -- HYBRID (keyword + semantic) retrieval: best when the query has
                                     exact terms that MUST match -- proper nouns, titles, numbers,
                                     technical terms.
  * read_article(title)           -- pull the FULL passage(s) of one article by exact title, to read
                                     it end-to-end (multi-hop: find an entity, then read its page).
                                     Reads one full document by exact title.

Backend: a Databricks Vector Search Delta-Sync index built by usecases/agentic-search/build_corpus.py +
create_vs_index.py over a curated Wikipedia subset. One index serves both vector_search
(query_type=ANN) and keyword_search (query_type=HYBRID). The index is a MANAGED external service, so
-- unlike an in-process BM25 index -- nothing is loaded per rollout worker and the corpus scale never
lands on the GPU node; the tool just issues authenticated REST queries.

We query the index over its REST API via the databricks-sdk WorkspaceClient's generic
``api_client.do("POST", "/api/2.0/vector-search/indexes/{index}/query", body=...)``. This deliberately
AVOIDS the databricks-vectorsearch client: that client's install downgraded protobuf below the pinned
image's vLLM gencode (6.33.x) and stopped vLLM from starting. databricks-sdk is PREINSTALLED
in the image (0.139.0), protobuf-safe, and resolves ambient workspace auth inside a df1 job with no
env vars (verified by probe_vs_access.py: host+token resolve, ANN+HYBRID queries return rows). So there is NO runtime
pip install and NO protobuf perturbation.

Contract identical to usecases/math/tool.py: Google-style docstring + type hints -> the
OpenAI tool schema is inferred; a ``str`` return -> a text ToolResponse. The LOGIC lives in plain
``_impl`` functions so the eval harness calls the exact code the rollout uses. Importable (and the
schema inferable) on CPU without verl or the sdk (both are optional-shimmed).

Env:
    QA_VS_INDEX           full index name catalog.schema.index (required at runtime)
    QA_VS_ENDPOINT        endpoint name (kept for logging/compat; the query REST API keys off the index)
    QA_VS_TEXT_COL        text column name (default "text")
    QA_VS_TITLE_COL       title column name (default "title")
    QA_VS_ID_COL          primary-key column name (default "id")
    QA_SEARCH_TOP_K       default results per search (default 5; max 20)
    QA_TOOL_MAX_CHARS     cap each tool response (default 4000)
    QA_SNIPPET_CHARS      per-passage snippet length for search results (default 600)
    DATABRICKS_HOST / DATABRICKS_TOKEN  optional explicit auth; ambient auth is used when unset
"""
from __future__ import annotations

import json
import os
import re

# --- verl decorator: no-op shim so this file imports/tests standalone. -------------------------
try:
    from verl.tools.function_tool import function_tool
except Exception:  # pragma: no cover

    def function_tool(name=None, *, schema=None):
        def deco(fn):
            return fn

        return deco(name) if callable(name) else deco


# --- databricks-sdk: optional import so CPU import / schema inference works without it. ----------
try:
    from databricks.sdk import WorkspaceClient

    _HAS_SDK = True
except Exception:  # noqa: BLE001
    _HAS_SDK = False


QA_VS_ENDPOINT = os.environ.get("QA_VS_ENDPOINT", "")
QA_VS_INDEX = os.environ.get("QA_VS_INDEX", "")
_TEXT_COL = os.environ.get("QA_VS_TEXT_COL", "text")
_TITLE_COL = os.environ.get("QA_VS_TITLE_COL", "title")
_ID_COL = os.environ.get("QA_VS_ID_COL", "id")
_TOP_K = int(os.environ.get("QA_SEARCH_TOP_K", "5"))
_MAX_CHARS = int(os.environ.get("QA_TOOL_MAX_CHARS", "4000"))
_SNIPPET_CHARS = int(os.environ.get("QA_SNIPPET_CHARS", "600"))

_CLIENT = None          # cached WorkspaceClient (per process)
_QCACHE: dict = {}      # (mode, query, k) -> rendered result string; cuts duplicate VS QPS


def _clip(s: str) -> str:
    return s if len(s) <= _MAX_CHARS else s[:_MAX_CHARS] + "\n...[truncated; refine the query or read_article a title]"


def _coerce_int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:  # noqa: BLE001
        return default


def _get_client():
    """Build + cache the WorkspaceClient once per worker. Returns the client or an error string."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not _HAS_SDK:
        return "Error: databricks-sdk not installed in this environment."
    if not QA_VS_INDEX:
        return "Error: QA_VS_INDEX not configured."
    try:
        host = os.environ.get("DATABRICKS_HOST") or None
        token = os.environ.get("DATABRICKS_TOKEN") or None
        if host and token:
            _CLIENT = WorkspaceClient(host=host, token=token)
        else:
            _CLIENT = WorkspaceClient()  # ambient workspace auth (works inside a df1 job)
    except Exception as e:  # noqa: BLE001
        return f"Error: could not init Databricks client: {e}"
    return _CLIENT


def _vs_query(query_text: str, k: int, query_type: str, filters: dict | None = None) -> dict | str:
    """POST the Vector Search REST query. Returns the response dict or an error string."""
    client = _get_client()
    if isinstance(client, str):
        return client
    body = {
        "num_results": k,
        "columns": [_ID_COL, _TITLE_COL, _TEXT_COL],
        "query_text": query_text,
        "query_type": query_type,
    }
    if filters:
        body["filters_json"] = json.dumps(filters)
    try:
        return client.api_client.do("POST", f"/api/2.0/vector-search/indexes/{QA_VS_INDEX}/query", body=body)
    except Exception as e:  # noqa: BLE001
        return f"Error: Vector Search query failed: {e}"


def _rows(res) -> list[list]:
    """Pull the data_array out of a VS query response (dict shape)."""
    try:
        return res.get("result", {}).get("data_array", []) or []
    except AttributeError:
        return []


def _col_order(res, fallback: list[str]) -> list[str]:
    """Column names in the order they appear in each data_array row (score is appended last)."""
    try:
        cols = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
        return cols or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _search(query: str, top_k: int, query_type: str) -> str:
    q = str(query).strip()
    if len(q) < 2:
        return "Error: query must be at least 2 characters"
    k = max(1, min(_coerce_int(top_k, _TOP_K), 20))
    key = (query_type, q, k)
    if key in _QCACHE:
        return _QCACHE[key]

    res = _vs_query(q, k, query_type)
    if isinstance(res, str):
        return res  # error string
    cols = [_ID_COL, _TITLE_COL, _TEXT_COL]
    order = _col_order(res, cols + ["score"])
    ti = order.index(_TITLE_COL) if _TITLE_COL in order else 1
    xi = order.index(_TEXT_COL) if _TEXT_COL in order else 2
    rows = _rows(res)
    if not rows:
        out = f"No passages found for {query!r}. Try different terms, or the other search tool."
        _QCACHE[key] = out
        return out

    parts = []
    for i, row in enumerate(rows):
        title = row[ti] if ti < len(row) else "?"
        text = row[xi] if xi < len(row) else ""
        score = row[-1]
        snippet = text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS] + "..."
        score_s = f"  score={score:.3f}" if isinstance(score, (int, float)) else ""
        parts.append(f"[{i}] {title}{score_s}\n    {snippet}")
    out = _clip("\n\n".join(parts))
    _QCACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Plain implementations (client-free-safe, directly callable by the eval harness).
# ---------------------------------------------------------------------------
def _vector_search(query: str, top_k: int = _TOP_K) -> str:
    return _search(query, top_k, "ANN")


def _keyword_search(query: str, top_k: int = _TOP_K) -> str:
    return _search(query, top_k, "HYBRID")


def _read_article(title: str) -> str:
    t = str(title).strip()
    if len(t) < 2:
        return "Error: title must be at least 2 characters"
    # Exact-title fetch: server-side filter on the title column; query_text=title keeps it a valid
    # query. Even if the server-side filter is a no-op for this index version, the client-side
    # exact-title match below still guarantees we only return the requested article.
    res = _vs_query(t, 20, "ANN", filters={_TITLE_COL: t})
    if isinstance(res, str):
        return res

    order = _col_order(res, [_ID_COL, _TITLE_COL, _TEXT_COL, "score"])
    ti = order.index(_TITLE_COL) if _TITLE_COL in order else 1
    xi = order.index(_TEXT_COL) if _TEXT_COL in order else 2
    rows = _rows(res)
    # Keep only exact (case-insensitive) title matches, in returned order.
    passages = [row[xi] for row in rows if xi < len(row) and str(row[ti]).strip().lower() == t.lower()]
    if not passages:
        return (f"No article titled {title!r} found. Use vector_search / keyword_search to find the "
                f"exact title first (titles are shown in each result).")
    return _clip(f"Article: {title}\n\n" + "\n".join(passages))


# ---------------------------------------------------------------------------
# verl @function_tool wrappers: docstring + type hints drive the OpenAI schema.
# ---------------------------------------------------------------------------
@function_tool("vector_search")
def vector_search(query: str, top_k: int = _TOP_K) -> str:
    """Semantic search over the Wikipedia corpus; returns the most relevant passages, each with its article title.

    Use this for conceptual or paraphrased questions where the wording may differ from the source
    ("who composed the score for ..."). Each result shows its article title -- pass that title to
    read_article to read the full page. For a multi-hop question, search for the first entity, read
    it to find the second entity, then search/read again.

    Args:
        query: What to look for, in natural language.
        top_k: How many passages to return (default 5; max 20).
    """
    return _vector_search(query, top_k)


@function_tool("keyword_search")
def keyword_search(query: str, top_k: int = _TOP_K) -> str:
    """Hybrid keyword+semantic search over the Wikipedia corpus; best when exact terms must match.

    Prefer this over vector_search when the query contains a proper noun, an exact title, a number,
    or a rare/technical term that must appear verbatim (e.g. a specific person, film, or place name).
    Returns passages with their article titles; pass a title to read_article for the full page.

    Args:
        query: What to look for; include the exact names/terms that must match.
        top_k: How many passages to return (default 5; max 20).
    """
    return _keyword_search(query, top_k)


@function_tool("read_article")
def read_article(title: str) -> str:
    """Read the full passage(s) of one Wikipedia article by its EXACT title.

    Use after a search returns a promising article title, to read that page end-to-end (e.g. to find
    a linked entity for the next hop of a multi-hop question). The title must match exactly as shown
    in a search result.

    Args:
        title: The exact article title, e.g. "Christopher Nolan".
    """
    return _read_article(title)


# ---------------------------------------------------------------------------
# Live smoke (requires a reachable index; ambient auth inside a df1 job, or set
# DATABRICKS_HOST/DATABRICKS_TOKEN locally):
#   QA_VS_INDEX=main.mshtelma.wiki_qa_corpus_index python3 usecases/agentic-search/tool.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[qa_search_tools] has_sdk={_HAS_SDK} index={QA_VS_INDEX!r}")
    print("vector_search ->", _vector_search("who directed the film Inception", 3)[:400])
    print("keyword_search ->", _keyword_search("Inception 2010 film", 3)[:400])
    print("read_article  ->", _read_article("Inception")[:300])
