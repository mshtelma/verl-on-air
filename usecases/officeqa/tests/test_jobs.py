import hashlib
import json
from pathlib import Path

import yaml

from support import REPO, load_module

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_and_checkpoint_eval_have_the_same_policy():
    before = yaml.safe_load((ROOT / "air" / "3_baseline_eval.yaml").read_text())
    after = yaml.safe_load((ROOT / "air" / "5_eval.yaml").read_text())
    before.pop("experiment_name")
    after.pop("experiment_name")
    before["env_variables"].pop("EVAL_MODEL_PATH")
    after["env_variables"].pop("EVAL_CKPT_ROOT")
    assert before == after
    train = yaml.safe_load((ROOT / "air" / "4_train.yaml").read_text())
    assert before["env_variables"]["OQ_MAX_TURNS"] == train["env_variables"]["MAX_TURNS"]
    assert int(before["env_variables"]["OQ_PROMPT_TOKENS"]) == train["parameters"]["max_prompt_length"]
    assert int(before["env_variables"]["OQ_MAX_GENERATION_TOKENS"]) == train["parameters"]["max_response_length"]
    assert before["env_variables"]["OQ_TOOL_MAX_CHARS"] == train["env_variables"]["MAX_TOOL_RESPONSE_LEN"]


def test_vendored_source_receipts_match_the_port():
    receipt = json.loads((ROOT / "SOURCE.json").read_text())
    for row in receipt["files"]:
        assert hashlib.sha256((ROOT / row["file"]).read_bytes()).hexdigest() == row["ported_sha256"]


def test_composition_allows_the_officeqa_subclass_but_rejects_unverified_loops(tmp_path):
    cc = load_module(REPO / "scripts" / "compose_check.py")
    assert cc.is_role_span_loop("agent_loop.OfficeQAToolAgentLoop", [str(ROOT)])
    (tmp_path / "bad.py").write_text("class Unverified:\n    pass\n")
    assert not cc.is_role_span_loop("bad.Unverified", [str(tmp_path)])


def test_long_context_training_resolves_packed_fused_engine(tmp_path):
    cc = load_module(REPO / "scripts" / "compose_check.py")
    rendered = cc.render(ROOT / "air" / "4_train.yaml", tmp_path, 29500)
    assert rendered["returncode"] == 0, rendered["stderr"]
    config = cc.compose(cc.ensure_verl_src(), "async", rendered["overrides"])
    model = config["actor_rollout_ref"]["model"]
    actor = config["actor_rollout_ref"]["actor"]["megatron"]
    assert model["use_remove_padding"] is True
    assert model["use_fused_kernels"] is True
    assert actor["use_remove_padding"] is True
    assert actor["tensor_model_parallel_size"] == 2
    assert actor["context_parallel_size"] == 4
    assert config["actor_rollout_ref"]["rollout"]["response_length"] == 96256


def test_fused_engine_requires_packed_sequences():
    preflight = load_module(REPO / "engine" / "lib" / "preflight.py")
    errors, _ = preflight.check_knobs("async", {"MODEL_USE_FUSED_KERNELS": "True"})
    assert any("requires MODEL_USE_REMOVE_PADDING=True" in error for error in errors)
    errors, _ = preflight.check_knobs(
        "async", {"MODEL_USE_FUSED_KERNELS": "true", "MODEL_USE_REMOVE_PADDING": "true"})
    assert errors == []


def test_final_run_uses_all_training_questions_and_frozen_heldout_policy(tmp_path):
    before = yaml.safe_load((ROOT / "air" / "3_baseline_final.yaml").read_text())
    after = yaml.safe_load((ROOT / "air" / "5_eval_final.yaml").read_text())
    before.pop("experiment_name")
    after.pop("experiment_name")
    before["env_variables"].pop("EVAL_MODEL_PATH")
    after["env_variables"].pop("EVAL_CKPT_ROOT")
    assert before == after
    assert before["compute"]["num_accelerators"] == 8
    assert before["env_variables"]["OQ_EVAL_HELDOUT_LIMIT"] == "43"
    assert before["env_variables"]["EVAL_EXPECT_N"] == "236"
    train = yaml.safe_load((ROOT / "air" / "4_train_final.yaml").read_text())
    assert train["compute"]["num_accelerators"] == 16
    assert train["env_variables"]["OQ_TRAIN_SELECTION"] == "official"
    assert train["parameters"]["train_files"].endswith("/train.parquet")
    assert train["parameters"]["total_rollout_steps"] == 400
    assert int(train["env_variables"]["JUDGE_DEADLINE_S"]) > int(train["env_variables"]["JUDGE_TIMEOUT"]) * (int(train["env_variables"]["JUDGE_RETRIES"]) + 1) + 10
    cc = load_module(REPO / "scripts" / "compose_check.py")
    rendered = cc.render(ROOT / "air" / "4_train_final.yaml", tmp_path, 29500)
    assert rendered["returncode"] == 0, rendered["stderr"]
    config = cc.compose(cc.ensure_verl_src(), "async", rendered["overrides"])
    rollout = config["actor_rollout_ref"]["rollout"]
    assert rollout["multi_turn"]["max_assistant_turns"] == 128
    assert rollout["response_length"] == 96256
    assert rollout["max_model_len"] == 98304


def test_episode_budget_rejects_context_overflow_and_single_turn(tmp_path):
    cc = load_module(REPO / "scripts" / "compose_check.py")
    job = ROOT / "air" / "4_train_final.yaml"
    for knobs in [{"MAX_MODEL_LEN": "8192"}, {"EPISODE_TOKEN_BUDGET": "4096"}, {"MULTI_TURN": "False"}]:
        rendered = cc.render(job, tmp_path, 29500, knobs)
        assert rendered["returncode"] != 0 and "FATAL" in rendered["stderr"]


def test_run_manifest_keeps_public_token_limits_without_recording_credentials(monkeypatch):
    manifest = load_module(REPO / "engine" / "lib" / "run_manifest.py")
    monkeypatch.setenv("EPISODE_TOKEN_BUDGET", "98304")
    monkeypatch.setenv("OQ_MAX_GENERATION_TOKENS", "4096")
    monkeypatch.setenv("OQ_TRAIN_SELECTION", "official")
    monkeypatch.setenv("JUDGE_API_KEY", "private-value")
    monkeypatch.setenv("OQ_ACCESS_TOKEN", "private-value")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "private-value")
    knobs = manifest.build([], {})["knobs"]
    assert knobs["EPISODE_TOKEN_BUDGET"] == "98304"
    assert knobs["OQ_MAX_GENERATION_TOKENS"] == "4096"
    assert knobs["OQ_TRAIN_SELECTION"] == "official"
    assert "private-value" not in knobs.values()
