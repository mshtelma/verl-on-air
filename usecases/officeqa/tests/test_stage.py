import pytest

import stage


def test_required_bm25_failure_precedes_data_and_compute_access(monkeypatch):
    monkeypatch.setenv("OQ_REQUIRE_BM25", "1")
    monkeypatch.setenv("POD_RANK", "0")
    monkeypatch.setenv("TRAINING_NODES", "2")
    monkeypatch.delenv("OQ_DATA_DIR", raising=False)
    monkeypatch.setattr(stage.corpus, "bm25s", None)
    with pytest.raises(RuntimeError, match="requires BM25 retrieval"):
        stage.main()


def test_judge_node_skips_actor_retrieval_preflight(monkeypatch):
    monkeypatch.setenv("OQ_REQUIRE_BM25", "1")
    monkeypatch.setenv("POD_RANK", "2")
    monkeypatch.setenv("TRAINING_NODES", "2")
    monkeypatch.delenv("OQ_DATA_DIR", raising=False)
    monkeypatch.setattr(stage.corpus, "bm25s", None)
    assert stage.main() is None
