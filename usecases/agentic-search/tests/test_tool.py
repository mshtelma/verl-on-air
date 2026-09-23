"""usecases/agentic-search/tool.py: backend failures vs model-visible text (R07, R09).

The Vector Search call is stubbed at `_vs_query`; everything above it is the real tool code.
"""
from __future__ import annotations

import pytest

from support import load_usecase

EMPTY = {"manifest": {"columns": [{"name": "id"}, {"name": "title"}, {"name": "text"}, {"name": "score"}]},
         "result": {"data_array": []}}


def rows(*r):
    return {**EMPTY, "result": {"data_array": [list(x) for x in r]}}


@pytest.fixture
def tool():
    t = load_usecase("agentic-search", "tool", QA_VS_INDEX="main.x.idx")
    t._QCACHE.clear()
    return t


def test_no_results_does_not_echo_the_models_query(tool):
    tool._vs_query = lambda *a, **k: EMPTY
    out = tool._vector_search("Miquette Giraudy spouse", 5)
    assert "No passages found" in out and "Miquette" not in out


def test_missing_article_does_not_echo_the_models_title(tool):
    tool._vs_query = lambda *a, **k: rows(("1", "Other", "text", 0.1))
    out = tool._read_article("Miquette Giraudy")
    assert "No article" in out and "Miquette" not in out


def test_article_is_named_by_its_corpus_title(tool):
    tool._vs_query = lambda *a, **k: rows(("1", "Gong (band)", "Formed in 1967.", 0.9))
    out = tool._read_article("gong (BAND)")
    assert out.startswith("Article: Gong (band)\n") and "Formed in 1967." in out


def test_backend_failure_raises_for_the_eval_and_is_text_for_the_model(tool):
    def boom(*a, **k):
        raise tool.ToolInfraError("Vector Search query failed: 403")
    tool._vs_query = boom
    with pytest.raises(tool.ToolInfraError):
        tool.vector_search_impl("capital of France", 5)
    assert tool._vector_search("capital of France", 5) == "Error: Vector Search query failed: 403"
    assert tool.vector_search("capital of France") == "Error: Vector Search query failed: 403"


def test_bad_arguments_are_the_models_error_not_an_infra_error(tool):
    assert tool.vector_search_impl("x", 5).startswith("Error: query must be at least 2 characters")
    assert tool.read_article_impl("").startswith("Error: title must be at least 2 characters")
