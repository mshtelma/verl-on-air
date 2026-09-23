"""engine/train/run_grpo_fully_async.sh exit guard (REVIEW.md R02 + the swallowed-crash path).

The real launcher runs single-node with `python3 -m ...fully_async_main` replaced by
tests/fakes/fake_fully_async_main.py. Before the fix, the first two scenarios exited 0
("Treating as SUCCESS") although the recipe exited 42; the swallowed-crash scenario exited 0 too.
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from support import REPO, StubBin, fake_train_checkpoint, run

FAKE_MAIN = REPO / "tests" / "fakes" / "fake_fully_async_main.py"
FINAL = 10          # 20 prompt groups / (1 trigger * 1 batch * ppo_mini 2) = 10 syncs
RUN_ID = "test-run"


@pytest.fixture
def job(tmp_path: Path, stub_bin: StubBin):
    """A single-node run of the REAL launcher (from a copy of engine/, so logs land in tmp)."""
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "engine", repo / "engine")
    root, rdv, site = tmp_path / "ckpt", tmp_path / "rdv", tmp_path / "site"
    ckpt = root / RUN_ID            # the launcher writes to <output_dir>/<RUN_ID>/
    for d in (ckpt, rdv, site):
        d.mkdir(parents=True)
    data = tmp_path / "train.parquet"
    pq.write_table(pa.table({"prompt": [f"q{i}" for i in range(30)]}), data)
    model = tmp_path / "model"          # preflight checks the geometry against its config.json
    model.mkdir()
    (model / "config.json").write_text('{"num_attention_heads": 16, "num_key_value_heads": 4, "num_hidden_layers": 8}')
    hp = tmp_path / "hparams.yaml"
    hp.write_text(yaml.safe_dump({"output_dir": str(root), "train_files": str(data), "val_files": str(data),
                                  "total_rollout_steps": 2 * FINAL, "ppo_mini_batch_size": 2,
                                  "rollout_n": 2, "total_epochs": 1, "model_name": str(model)}))
    stub_bin.add("python3", "\n".join([
        f'if [[ "${{1:-}}" == "-c" && "${{2:-}}" == *"import os, verl"* ]]; then echo {site}; exit 0; fi',
        'if [[ "${1:-}" == "-m" && "${2:-}" == "verl.experimental.fully_async_policy.fully_async_main" ]]; then',
        f'  exec {sys.executable} {FAKE_MAIN}',
        'fi',
        f'exec {sys.executable} "$@"']))
    stub_bin.add("ray")

    def launch(scenario: str, **env: str):
        e = stub_bin.env(
            NUM_NODES="1", LOCAL_WORLD_SIZE="8", NODE_RANK="0", HYPERPARAMETERS_PATH=str(hp),
            TRIGGER_SYNC_STEP="1", REQUIRE_BATCHES="1", SAVE_FREQ="5", VOA_RDV_DIR=str(rdv), RUN_ID=RUN_ID,
            CERT_SETTLE_S="0", ABORT_POLL_S="1", ABORT_GRACE_S="1",
            FAKE_SCENARIO=scenario, FAKE_CKPT_DIR=str(ckpt), FAKE_FINAL=str(FINAL))
        e.update(env)
        return run(["bash", str(repo / "engine/train/run_grpo_fully_async.sh")], env=e, timeout=90)

    launch.ckpt, launch.rdv = ckpt, rdv
    return launch


def result(job) -> dict:
    return json.loads((job.ckpt / "run_result.json").read_text())


# --- failures that used to be reported as SUCCESS ----------------------------------------------
def test_exception_with_the_finally_marker_keeps_its_exit_code(job):
    r = job("exception_with_finally_marker")
    assert r.returncode == 42, r.stdout[-2000:]
    assert "Treating as SUCCESS" not in r.stdout and not result(job)["certified"]


def test_partial_checkpoint_dir_after_disk_full_keeps_its_exit_code(job):
    r = job("partial_dir_then_disk_full")
    assert r.returncode == 42, r.stdout[-2000:]
    assert "no latest_checkpointed_iteration.txt" in r.stdout


def test_crash_after_an_intermediate_save_fails(job):
    r = job("crash_after_intermediate_save")
    assert r.returncode == 1
    assert f"version {FINAL // 2}" in r.stdout and "stopped early" in r.stdout


def test_swallowed_crash_that_exits_zero_is_a_failure(job):
    # verl's rollouter swallows its own exception and sends the normal stop signal
    r = job("swallowed_crash_exits_zero")
    assert r.returncode == 1, r.stdout[-2000:]
    assert result(job)["raw_rc"] == 0 and not result(job)["certified"]


def test_a_tracker_left_by_a_previous_run_does_not_certify(job):
    fake_train_checkpoint(job.ckpt, FINAL)
    (job.ckpt / "latest_checkpointed_iteration.txt").write_text(str(FINAL))
    r = job("nothing_new", RESUME="auto")      # (RESUME=never refuses to start at all: below)
    assert r.returncode == 1 and "unchanged since before the run" in r.stdout


# --- run identity (W2.1): no implicit resume, no run without an id ------------------------------
def test_existing_checkpoints_refuse_a_fresh_start(job):
    fake_train_checkpoint(job.ckpt, 5)
    r = job("complete_clean")
    assert r.returncode != 0 and "already holds checkpoints of run test-run" in r.stdout
    assert not (job.ckpt / "global_step_10").exists(), "the recipe ran anyway"


def test_a_run_without_a_run_id_is_refused(job):
    r = job("complete_clean", RUN_ID="")
    assert r.returncode != 0 and "RUN_ID is not set" in r.stdout


def test_the_manifest_records_what_ran(job):
    assert job("complete_clean").returncode == 0
    m = json.loads((job.ckpt / "run_manifest.json").read_text())
    assert m["run_id"] == RUN_ID and m["launcher"] == "run_grpo_fully_async.sh"
    assert m["expected_final_version"] == str(FINAL)
    assert "trainer.resume_mode=disable" in m["verl_overrides"]
    assert any(o.startswith("trainer.default_local_dir=") and o.endswith(f"/{RUN_ID}") for o in m["verl_overrides"])


def test_final_checkpoint_without_its_manifest_fails(job):
    r = job("final_without_manifest")
    assert r.returncode == 1 and "ckpt_contents.json" in r.stdout


def test_an_abort_request_vetoes_an_otherwise_complete_run(job):
    r = job("complete_but_aborted")
    assert r.returncode == 1
    assert "abort requested by test-reward-worker" in r.stdout


def test_hard_error_signature_vetoes_overriding_a_nonzero_exit(job):
    r = job("complete_then_oom")
    assert r.returncode == 1 and "hard-failure signature" in r.stdout


# --- genuine completion is still recognized ------------------------------------------------------
def test_teardown_cancellation_after_a_complete_run_is_success(job):
    r = job("complete_then_teardown_cancelled")
    assert r.returncode == 0, r.stdout[-2000:]
    res = result(job)
    assert res["certified"] and res["raw_rc"] == 1 and res["observed_version"] == str(FINAL)
    assert res["checkpoint"]["step"] == FINAL


def test_clean_complete_run_is_success(job):
    r = job("complete_clean")
    assert r.returncode == 0 and "CERTIFIED" in r.stdout


# --- preflight -----------------------------------------------------------------------------------
@pytest.mark.parametrize("env,msg", [
    ({"SAVE_FREQ": "-1"}, "cannot be certified complete"),
    ({"TEST_FREQ": "0"}, "divide by zero"),
    ({"TRIGGER_SYNC_STEP": "3"}, "must be a multiple of samples/sync"),  # 20 groups / 6 per sync
])
def test_preflight_refuses_uncertifiable_plans_before_launch(job, env, msg):
    r = job("complete_clean", **env)
    assert r.returncode != 0 and msg in r.stdout, r.stdout[-2000:]
    assert not (job.ckpt / "global_step_10").exists(), "the recipe ran anyway"


def test_uncertified_smokes_are_allowed_only_explicitly(job):
    r = job("complete_clean", SAVE_FREQ="-1", ALLOW_UNCERTIFIED="1")
    assert r.returncode == 0 and "UNCERTIFIED" in r.stdout


# --- the abort watchdog --------------------------------------------------------------------------
def test_watchdog_stops_a_running_job_when_the_run_is_aborted(job):
    def raise_abort():
        time.sleep(2)
        (job.rdv / "ABORT.json").write_text(json.dumps({"reason": "judge down", "source": "test"}))
    threading.Thread(target=raise_abort, daemon=True).start()
    t0 = time.monotonic()
    r = job("hang")
    assert time.monotonic() - t0 < 30, "the hung job was not stopped"
    assert r.returncode != 0 and "ABORT requested" in r.stdout
    assert "abort requested by test: judge down" in r.stdout
