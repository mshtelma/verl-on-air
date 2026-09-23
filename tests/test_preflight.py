"""engine/lib/preflight.py: typed knobs, mode applicability, model/geometry limits, the plan (R25).

Before: fsdp below 16 GPUs auto-selected the CPU offload that crashes Megatron-FSDP; the async DP
guard ignored CP and floored a non-integral DP; bad TP/PP/EP for the model, zero/negative values
and enum typos surfaced (if at all) after the cluster came up; a knob only the other mode reads
was silently ignored; `MULTI_TURN=true` silently meant False; and a malformed parameters block
fell back to defaults."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from support import ENGINE, REPO, load_module, run

pf = load_module(ENGINE / "lib" / "preflight.py")
cc = load_module(REPO / "scripts" / "compose_check.py")
SEARCH = REPO / "usecases/agentic-search/air/4_train.yaml"
SEARCH_SYNC = REPO / "usecases/agentic-search/air/4_train_sync.yaml"
RUNG2 = REPO / "infra/geo3k/air/rung2_9b_fsdp_8gpu.yaml"


# --- phase 1: knobs -----------------------------------------------------------------------------
@pytest.mark.parametrize("env,msg", [
    ({"TP": "two"}, "TP='two' is not an integer"),
    ({"EP": "0"}, "EP='0': must be >= 1"),
    ({"ROLLOUT_GPU_MEM_UTIL": "0"}, "must be > 0"),
    ({"ROLLOUT_GPU_MEM_UTIL": "1.5"}, "must be <= 1"),
    ({"TOOL_FORMAT": "qwen3-coder"}, "expected one of hermes"),
    ({"MULTI_TURN": "maybe"}, "is not a boolean"),
    ({"ROLLOUT_TEMPERATURE": "1.2"}, "not a knob either launcher reads (a typo?)"),
    ({"STALENESS": "0.5"}, "only the TRAIN_MODE=async launcher reads it"),
])
def test_a_bad_or_inapplicable_knob_is_refused(env, msg):
    mode = "sync" if "STALENESS" in env else "async"
    problems, _ = pf.check_knobs(mode, env)
    assert any(msg in p for p in problems), problems


def test_a_knob_of_the_other_mode_is_refused_both_ways():
    assert pf.check_knobs("async", {"MEGATRON_MODE": "fsdp"})[0]
    assert pf.check_knobs("sync", {"TRIGGER_SYNC_STEP": "2"})[0]
    assert pf.check_knobs("sync", {"ROLLOUT_NNODES": "1"})[0]          # sync co-locates the rollout


@pytest.mark.parametrize("raw,want", [("True", "True"), ("true", "True"), ("1", "True"), ("yes", "True"),
                                      ("False", "False"), ("false", "False"), ("0", "False"), ("off", "False")])
def test_every_boolean_spelling_is_normalised(raw, want):
    assert pf.check_knobs("async", {"MULTI_TURN": raw}) == ([], {"MULTI_TURN": want})


def test_use_case_knobs_are_not_the_engines_business():
    assert pf.check_knobs("async", {"QA_VS_INDEX": "x", "JUDGE_TIMEOUT": "abc", "EVAL_LIMIT": "?"}) == ([], {})


# --- phase 2: geometry against the model ------------------------------------------------------------
BASE = dict(MODEL="/Volumes/x/models/Qwen3.5-35B-A3B", NUM_NODES="2", NODES="2", GPUS_PER_NODE="8",
            TRAINER_NODES="1", TRAINER_GPUS="8", ROLLOUT_GPUS="8", TP="2", PP="1", CP="1", EP="8", ETP="1",
            GEN_TP="8", PPO_MINI="32", ROLLOUT_N="16", MULTI_TURN="True", MAX_TURNS="12",
            TOTAL_ROLLOUT_STEPS="3200", TRIGGER_SYNC_STEP="1", REQUIRE_BATCHES="1", SAVE_FREQ="10")


@pytest.mark.parametrize("change,msg", [
    ({"TP": "4"}, "TP=4 does not divide the model's 2 kv heads"),
    ({"PP": "3"}, "PP=3 does not divide the model's 40 layers"),
    ({"EP": "3", "TRAINER_GPUS": "24"}, "EP=3 does not divide the model's 256 experts"),
    ({"GEN_TP": "3"}, "GEN_TP=3 does not divide the model's 16 attention heads"),
    ({"GEN_TP": "16"}, "GEN_TP=16 does not divide the 8 rollout GPUs"),
    ({"CP": "3"}, "not a whole multiple of TP*PP*CP"),                    # the old guard ignored CP
    ({"EP": "16"}, "EP*ETP*PP = 16*1*1 does not divide the 8 trainer GPUs"),
    ({"PPO_MINI": "6"}, "ppo_mini_batch_size=6 does not split evenly over DP=4"),
    ({"MODEL": "/somewhere/Unknown-7B"}, "limits cannot be checked"),
])
def test_geometry_the_model_or_megatron_cannot_run_is_refused(change, msg):
    problems, _ = pf.plan("async", {**BASE, **change})
    assert any(msg in p for p in problems), problems


def test_a_dense_model_takes_no_expert_parallelism(tmp_path: Path):
    problems, _ = pf.plan("async", {**BASE, "MODEL": "Qwen/Qwen3.5-9B", "EP": "2"})
    assert any("EP=2 on a dense model" in p for p in problems)


def test_limits_come_from_the_models_own_config(tmp_path: Path):
    # a VL checkpoint nests the language model's shape under text_config
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "num_attention_heads": 12, "num_key_value_heads": 3, "num_hidden_layers": 6}}))
    kv = {**BASE, "MODEL": str(tmp_path), "EP": "1", "TP": "3", "GEN_TP": "4", "TRAINER_GPUS": "12", "PPO_MINI": "12"}
    problems, out = pf.plan("async", kv)
    assert problems == [] and out["model"]["limits_from"].endswith("config.json")   # TP=3 fits 12/3 heads
    assert pf.plan("async", {**kv, "MODEL": "Qwen3.5-35B-A3B"})[0]                  # ...not the 35B's 2 KV


def test_sync_refuses_fsdp_with_offload_and_a_batch_that_does_not_divide():
    sync = {**BASE, "TRAINER_GPUS": "32", "ROLLOUT_GPUS": "32", "TP": "1", "MEGATRON_MODE": "fsdp",
            "OFFLOAD": "1", "TRAIN_BATCH_SIZE": "31", "TOTAL_TRAINING_STEPS": "100"}
    problems, _ = pf.plan("sync", sync)
    assert any("Megatron-FSDP crashes with CPU offload" in p for p in problems)
    assert any("train_batch_size*rollout_n = 496 is not divisible by the 32" in p for p in problems)


def test_the_plan_states_the_budget():
    problems, out = pf.plan("async", BASE)
    assert problems == []
    assert out["budget"] == {"prompt_groups": 3200, "trajectories": 51200, "optimizer_updates": 100,
                             "weight_syncs": 100, "prompt_groups_per_sync": 32}
    assert out["checkpoints"]["saved"] == 10 and out["parallelism"]["dp"] == 4


# --- the launchers run it ---------------------------------------------------------------------------
def test_every_shipped_training_job_passes_its_own_preflight(tmp_path: Path):
    for i, job in enumerate(cc.training_jobs()):
        r = cc.render(job, tmp_path, 29800 + 10 * i)
        assert r["returncode"] == 0 and "PREFLIGHT_PLAN " in r["stdout"], (job, r["stderr"][-800:])


@pytest.mark.parametrize("job,env,msg", [
    (SEARCH_SYNC, {"STALENESS": "0.5"}, "STALENESS: only the TRAIN_MODE=async launcher reads it"),
    (SEARCH, {"MEGATRON_MODE": "classic"}, "MEGATRON_MODE: only the TRAIN_MODE=sync launcher reads it"),
    (SEARCH, {"ROLLOUT_TEMPERATURE": "1.2"}, "ROLLOUT_TEMPERATURE: not a knob"),
    (SEARCH, {"TP": "4"}, "TP=4 does not divide the model's 2 kv heads"),
    (RUNG2, {"OFFLOAD": "auto"}, "would need CPU offload, which crashes Megatron-FSDP"),   # the reviewer's case
    (SEARCH_SYNC, {"OFFLOAD": "1"}, "Megatron-FSDP crashes with CPU offload"),
])
def test_the_launcher_stops_before_anything_runs(tmp_path: Path, job, env, msg):
    r = cc.render(job, tmp_path, 29850, env)
    assert r["returncode"] != 0 and msg in r["stderr"], r["stderr"][-600:]
    assert not r["overrides"]


def test_a_lowercase_boolean_now_means_what_it_says(tmp_path: Path):
    r = cc.render(SEARCH, tmp_path, 29860, {"MULTI_TURN": "true"})
    assert r["returncode"] == 0 and "actor_rollout_ref.rollout.multi_turn.enable=True" in r["overrides"]


def test_the_schema_is_what_the_launchers_read():
    """Every knob a launcher reads is typed for that launcher's mode, and every knob typed for a mode is
    read by that mode's launcher (or by run identity / the manifest / the use case)."""
    def reads(path: Path) -> set[str]:
        return set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)(?::-|-|:=|:\?)", path.read_text())) - pf.PLUMBING
    launcher = {"async": reads(ENGINE / "train/run_grpo_fully_async.sh"),
                "sync": reads(ENGINE / "train/run_grpo_megatron.sh")}
    shared = reads(ENGINE / "lib/run_identity.sh") | {"GIT_SHA", "VOA_IMAGE", "REWARD_SOURCE"}
    for mode, names in launcher.items():
        untyped = sorted(n for n in names if n not in pf.KNOBS or mode not in pf.KNOBS[n].modes)
        assert not untyped, f"{mode} launcher reads knobs the schema does not type for it: {untyped}"
    for name, knob in pf.KNOBS.items():
        for mode in knob.modes:
            assert name in launcher[mode] | shared, f"{name} is typed for {mode}, but nothing there reads it"


# --- the parameters block ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,msg", [("model_name: [unclosed\n", "is not valid YAML"),
                                      ("- a\n- b\n", "is not a mapping")])
def test_a_malformed_parameters_block_stops_the_script(tmp_path: Path, text, msg):
    hp = tmp_path / "hp.yaml"
    hp.write_text(text)
    r = run(["bash", "-c", f"set -e; source {ENGINE}/lib/hparams.sh; hp_check; echo SURVIVED"],
            env={"HYPERPARAMETERS_PATH": str(hp), "PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and msg in r.stdout and "SURVIVED" not in r.stdout


def test_make_preflight_reports_the_plan_and_the_gpu_hour_bound():
    r = run(["make", "--no-print-directory", "preflight", f"F={SEARCH.relative_to(REPO)}", "RUN_ID=r-test"])
    assert r.returncode == 0, r.stdout[-800:]
    out = json.loads(r.stdout)
    assert out["gpu_hours_upper_bound"] == 16 * 600 / 60 and out["plan"]["run_id"] == "r-test"
    assert out["plan"]["budget"]["weight_syncs"] == 100 and out["plan"]["roles"]["judge_nodes"] == 0
