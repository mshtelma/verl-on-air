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
