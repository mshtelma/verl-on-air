"""usecases/agentic-search/eval.py under the eval contract (REVIEW.md R07).

Reviewer reproduction: a retrieval-auth failure plus an inference ConnectionError finished
"normally" and wrote em=0, n=1, tool_errors=0 -- an outage indistinguishable from a bad model.
Here the eval runs against a fake vLLM (tests/support.FakeOpenAIServer) with the real HTTP,
retry, parsing, scoring and artifact code; only the tokenizer and the Vector Search calls are stubbed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import datasets
import pytest

from support import FakeOpenAIServer, env, load_usecase

ANSWER = {"choices": [{"text": "I know this. <answer>Paris</answer>", "finish_reason": "stop"}]}
TOOL_CALL = {"choices": [{"text": "<tool_call>\n<function=vector_search>\n<parameter=query>\ncapital of France"
                                  "\n</parameter>\n</function>\n</tool_call>", "finish_reason": "stop"}]}
MODELS = {"object": "list", "data": [{"id": "eval", "object": "model"}]}
PASSAGES = "[0] France  score=0.9\n    Paris is the capital of France."


class FakeTok:
    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages)


def serve(completion):
    """completion(n) -> (status, body): the /completions behaviour for the n-th request."""
    def reply(path, payload, n):
        if path.endswith("/models"):
            return 200, MODELS, 0
        status, body = completion(n)
        return status, body, 0
    return reply


@pytest.fixture
def make_eval(tmp_path: Path):
    val = tmp_path / "val.parquet"
    datasets.Dataset.from_list([
        {"data_source": "musique", "reward_model": {"ground_truth": ["Paris"]},
         "extra_info": {"question": f"What is the capital of France? ({i})", "index": i, "hop_type": "2hop"}}
        for i in range(3)]).to_parquet(str(val))
    ident = tmp_path / "identity.json"
    ident.write_text(json.dumps({"kind": "train_checkpoint", "step": 20, "identity": "abc123"}))

    def make(url: str, *, retrieval=None, **overrides: str):
        e = {"EVAL_BASE_URL": url, "EVAL_MODEL": "eval", "QA_VAL_PARQUET": str(val), "EVAL_LIMIT": "3",
             "EVAL_OUT": str(tmp_path / "out.json"), "EVAL_TRACE_OUT": str(tmp_path / "traces.jsonl"),
             "EVAL_MAX_TURNS": "3", "EVAL_HTTP_RETRIES": "1", "EVAL_CONCURRENCY": "2",
             "EVAL_MODEL_IDENTITY_FILE": str(ident), "QA_VS_INDEX": "main.x.idx", **overrides}
        E = load_usecase("agentic-search", "eval", **e)
        E._load_tokenizer = FakeTok
        E._qst.vector_search_impl = retrieval or (lambda q, k=5: PASSAGES)
        E._env = e
        return E

    make.tmp, make.val = tmp_path, val
    return make


def run_eval(E) -> int:
    with env(**E._env):
        return asyncio.run(E._main_async())


def artifact(make) -> dict:
    return json.loads((make.tmp / "out.json").read_text())


def test_healthy_run_is_valid_and_carries_its_identity(make_eval):
    with FakeOpenAIServer(serve(lambda n: (200, ANSWER))) as srv:
        rc = run_eval(make_eval(srv.url))
    a = artifact(make_eval)
    assert rc == 0 and a["valid"] and a["n_scored"] == 3 and a["em"] == 1.0
    assert a["model_identity"]["identity"] == "abc123"
    assert a["dataset"]["sha256"] == hashlib.sha256(make_eval.val.read_bytes()).hexdigest()
    assert a["question_ids_sha256"] and a["eval_policy"]["tools"] == ["vector_search", "keyword_search", "read_article"]
    assert len(list((make_eval.tmp / "out.json.parts").glob("*.json"))) == 3


def test_retrieval_auth_failure_stops_before_any_question(make_eval):
    with FakeOpenAIServer(serve(lambda n: (200, ANSWER))) as srv:
        E = make_eval(srv.url)

        def denied(q, k=5):
            raise E._qst.ToolInfraError("Vector Search query failed: 403 PERMISSION_DENIED")
        E._qst.vector_search_impl = denied
        with pytest.raises(SystemExit) as ex:
            run_eval(E)
        assert ex.value.code == 2
        assert not [p for p, _ in srv.requests if p.endswith("/completions")]
    assert not (make_eval.tmp / "out.json").exists()


def test_inference_outage_is_an_invalid_run_not_a_zero_score(make_eval):
    with FakeOpenAIServer(serve(lambda n: (503, {"error": "overloaded"}))) as srv:
        rc = run_eval(make_eval(srv.url))
    a = artifact(make_eval)
    assert rc == 1 and not a["valid"] and a["n_scored"] == 0
    assert a["infra_errors"]["infra_inference"] == 3
    assert all(r["status"] == "infra_inference" and not r["correct"] for r in a["results"])


def test_a_transient_failure_is_retried(make_eval):
    with FakeOpenAIServer(serve(lambda n: (503, {}) if n == 2 else (200, ANSWER))) as srv:
        rc = run_eval(make_eval(srv.url, EVAL_CONCURRENCY="1"))
    assert rc == 0 and artifact(make_eval)["valid"]


def test_context_limit_is_classified_and_not_retried(make_eval):
    too_long = (400, {"error": {"message": "This model's maximum context length is 32768 tokens"}})
    with FakeOpenAIServer(serve(lambda n: too_long)) as srv:
        rc = run_eval(make_eval(srv.url, EVAL_LIMIT="1", EVAL_EXPECT_N="1"))
        completions = [p for p, _ in srv.requests if p.endswith("/completions")]
    a = artifact(make_eval)
    assert rc == 1 and a["results"][0]["status"] == "context_limit" and len(completions) == 1


def test_a_short_dataset_invalidates_the_run(make_eval):
    with FakeOpenAIServer(serve(lambda n: (200, ANSWER))) as srv:
        rc = run_eval(make_eval(srv.url, EVAL_LIMIT="5"))
    a = artifact(make_eval)
    assert rc == 1 and "loaded 3 questions, expected 5" in a["invalid_reasons"][0]


def test_retrieval_failure_mid_episode_is_infrastructure(make_eval):
    with FakeOpenAIServer(serve(lambda n: (200, TOOL_CALL))) as srv:
        E = make_eval(srv.url, EVAL_LIMIT="1", EVAL_EXPECT_N="1")
        calls = {"n": 0}

        def flaky(q, k=5):
            calls["n"] += 1
            if calls["n"] > 1:  # the readiness probe succeeds; the episode's search does not
                raise E._qst.ToolInfraError("Vector Search query failed: timeout")
            return PASSAGES
        E._qst.vector_search_impl = flaky
        rc = run_eval(E)
    assert rc == 1 and artifact(make_eval)["results"][0]["status"] == "infra_retrieval"


def test_an_unknown_tool_is_the_models_error_and_is_scored(make_eval):
    bad_call = {"choices": [{"text": "<tool_call>\n<function=web_search>\n<parameter=q>x</parameter>\n"
                                     "</function>\n</tool_call>", "finish_reason": "stop"}]}
    with FakeOpenAIServer(serve(lambda n: (200, bad_call) if n % 2 else (200, ANSWER))) as srv:
        rc = run_eval(make_eval(srv.url, EVAL_CONCURRENCY="1"))
    a = artifact(make_eval)
    assert rc == 0 and a["valid"] and all(r["status"] == "scored" for r in a["results"])
    assert a["tool_errors"] >= 1


def test_the_served_name_must_be_listed(make_eval):
    with FakeOpenAIServer(serve(lambda n: (200, ANSWER))) as srv:
        with pytest.raises(SystemExit) as ex:
            run_eval(make_eval(srv.url, EVAL_MODEL="not-served"))
    assert ex.value.code == 2


def test_an_existing_artifact_is_never_overwritten(make_eval):
    (make_eval.tmp / "out.json").write_text("{}")
    with FakeOpenAIServer(serve(lambda n: (200, ANSWER))) as srv:
        with pytest.raises(SystemExit) as ex:
            run_eval(make_eval(srv.url))
        assert ex.value.code == 2 and not srv.requests
        assert run_eval(make_eval(srv.url, EVAL_OVERWRITE="1")) == 0
