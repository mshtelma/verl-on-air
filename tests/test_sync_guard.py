"""engine/train/run_grpo_megatron.sh: a sync run is certified from its final checkpoint too, and it
honours the abort channel (R02's principle for the second launcher; R26's cancellation case).

Before: the sync launcher ran main_ppo in the foreground and exited with whatever the tee pipeline
returned. A run that stopped short of its planned step with exit 0 passed; the abort channel (the
search reward raises it on a provenance failure) was never watched; and a TERM waited for training
to end. Here the real launcher runs single-node with `python3 -m verl.trainer.main_ppo` replaced by
tests/fakes/fake_fully_async_main.py (its scenarios are trainer-agnostic).
"""
from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from support import REPO, StubBin, run

FAKE_MAIN = REPO / "tests" / "fakes" / "fake_fully_async_main.py"
FINAL = 6
RUN_ID = "test-sync-run"


@pytest.fixture
def job(tmp_path: Path, stub_bin: StubBin):
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "engine", repo / "engine")
    root, rdv = tmp_path / "ckpt", tmp_path / "rdv"
    ckpt = root / RUN_ID
    for d in (ckpt, rdv):
        d.mkdir(parents=True)
    data = tmp_path / "train.parquet"
    pq.write_table(pa.table({"prompt": [f"q{i}" for i in range(64)]}), data)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"num_attention_heads": 16, "num_key_value_heads": 4, "num_hidden_layers": 8}')
    hp = tmp_path / "hparams.yaml"
    hp.write_text(yaml.safe_dump({"output_dir": str(root), "train_files": str(data), "val_files": str(data),
                                  "total_training_steps": FINAL, "train_batch_size": 4, "ppo_mini_batch_size": 4,
                                  "rollout_n": 2, "total_epochs": 1, "model_name": str(model)}))
    stub_bin.add("python3", "\n".join([
        'if [[ "${1:-}" == "-m" && "${2:-}" == "verl.trainer.main_ppo" ]]; then',
        f'  exec {sys.executable} {FAKE_MAIN}',
        'fi',
        f'exec {sys.executable} "$@"']))
    stub_bin.add("ray")

    def env(scenario: str, **extra: str) -> dict:
        e = stub_bin.env(
            NUM_NODES="1", LOCAL_WORLD_SIZE="8", NODE_RANK="0", HYPERPARAMETERS_PATH=str(hp),
            MEGATRON_MODE="classic", EP="1", SAVE_FREQ="3", VOA_RDV_DIR=str(rdv), RUN_ID=RUN_ID,
            CERT_SETTLE_S="0", ABORT_POLL_S="1", ABORT_GRACE_S="1",
            FAKE_SCENARIO=scenario, FAKE_CKPT_DIR=str(ckpt), FAKE_FINAL=str(FINAL))
        e.update(extra)
        return e

    def launch(scenario: str, **extra: str):
        return run(["bash", str(repo / "engine/train/run_grpo_megatron.sh")], env=env(scenario, **extra),
                   timeout=90, cwd=str(tmp_path))

    launch.ckpt, launch.rdv, launch.env, launch.script = ckpt, rdv, env, repo / "engine/train/run_grpo_megatron.sh"
    launch.cwd = tmp_path
    return launch


def result(job) -> dict:
    return json.loads((job.ckpt / "run_result.json").read_text())


def test_a_complete_run_is_certified(job):
    r = job("complete_clean")
    assert r.returncode == 0 and "CERTIFIED" in r.stdout, r.stdout[-2000:]
    assert result(job)["observed_version"] == str(FINAL)
    assert json.loads((job.ckpt / "run_manifest.json").read_text())["expected_final_version"] == str(FINAL)


def test_a_run_that_stops_short_with_exit_0_fails(job):
    r = job("swallowed_crash_exits_zero")
    assert r.returncode == 1 and "stopped early" in r.stdout, r.stdout[-2000:]


def test_a_non_zero_exit_is_never_overridden(job):
    r = job("complete_then_teardown_cancelled")            # the async recipe's benign teardown case
    assert r.returncode == 1 and "never overridden" in r.stdout
    assert not result(job)["certified"]


def test_an_abort_request_vetoes_the_run(job):
    r = job("complete_but_aborted")
    assert r.returncode == 1 and "abort requested by test-reward-worker" in r.stdout


def test_the_watchdog_stops_a_hung_sync_run(job):
    def raise_abort():
        time.sleep(2)
        (job.rdv / "ABORT.json").write_text(json.dumps({"reason": "provenance failure", "source": "test"}))
    threading.Thread(target=raise_abort, daemon=True).start()
    t0 = time.monotonic()
    r = job("hang")
    assert time.monotonic() - t0 < 30 and r.returncode != 0 and "ABORT requested" in r.stdout


def test_a_run_without_checkpoints_is_reported_as_uncertified(job):
    r = job("nothing_new", SAVE_FREQ="-1")
    assert r.returncode == 0 and "UNCERTIFIED" in r.stdout and not (job.ckpt / "run_result.json").exists()


def test_a_cancelled_launcher_stops_its_trainer_at_once(job):
    p = subprocess.Popen(["bash", str(job.script)], env=job.env("hang"), cwd=str(job.cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not list(job.cwd.glob("logs/*.log")):
        time.sleep(0.2)
    time.sleep(1.5)                                        # the fake is now "training forever"
    t0 = time.monotonic()
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=30)
    assert time.monotonic() - t0 < 10, "the TERM waited for training to finish"
    assert p.returncode == 143 and "TERM received: stopping the training driver" in out, out[-1500:]
    left = subprocess.run(["pgrep", "-f", str(FAKE_MAIN)], capture_output=True, text=True).stdout.strip()
    assert not left, f"the trainer survived its launcher: {left}"
