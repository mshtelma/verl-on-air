"""Distributed start-up and teardown: every node ends in a stated way, soon (R26).

Before: the head's cleanup trap was installed only after its bootstrap succeeded, so a failed one
left Ray running and the workers waiting until the job timeout; workers only watched the head's
port (a head killed without its trap leaves it open); rendezvous waits took files from earlier
attempts; training's judge wait and the judge's own health timeout were one number for two
different deadlines; the judge exited with whatever `wait` said; and multi-node SGLang had no
launch of its own. Here the real scripts run with `ray`, `vllm`, `curl` and `cp` stubbed.
"""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from support import ENGINE, REPO, FakeOpenAIServer, StubBin, chat_completion, fake_hf_model, load_module, run

LIB = ENGINE / "lib"
cc = load_module(REPO / "scripts" / "compose_check.py")
FAST = dict(RAY_JOIN_DELAY_S="0", RAY_JOIN_INTERVAL_S="0", RAY_DRAIN_POLL_S="0.2", RAY_NODES_TIMEOUT_S="0")


def sh(script: str, stub: StubBin, timeout: float = 30, **env: str) -> subprocess.CompletedProcess:
    return run(["bash", "-c", f"set -eu -o pipefail; {script}"], env=stub.env(**FAST, **env), timeout=timeout)


@pytest.fixture
def open_port():
    """A listening TCP port -- a Ray head that is up (or whose daemons outlived its script)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    stop = threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                srv.accept()[0].close()
            except OSError:
                pass
    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield srv.getsockname()[1]
    stop.set()
    srv.close()


# --- Ray: head and workers (engine/lib/ray_cluster.sh) ----------------------------------------------
def test_a_head_whose_bootstrap_fails_still_stops_ray_and_tells_the_workers(tmp_path: Path, stub_bin: StubBin):
    stub_bin.add("ray")
    rdv = tmp_path / "rdv"
    r = sh(f"source {LIB}/ray_cluster.sh; ray_install_cleanup_trap; ray_start_head 2 8 127.0.0.1",
           stub_bin, VOA_RDV_DIR=str(rdv))
    assert r.returncode != 0 and "GPUs registered within" in r.stdout
    assert any(c.startswith("ray stop") for c in stub_bin.calls("ray"))       # the trap ran
    assert (rdv / "ray_head_done").read_text().strip() == "rc=1"


def test_the_head_heartbeats_while_alive_and_not_after(tmp_path: Path, stub_bin: StubBin):
    stub_bin.add("ray")
    rdv = tmp_path / "rdv"
    r = sh(f"source {LIB}/ray_cluster.sh; ray_install_cleanup_trap; sleep 1.5", stub_bin,
           VOA_RDV_DIR=str(rdv), RAY_HEARTBEAT_S="0.3")
    assert r.returncode == 0 and (rdv / "ray_head_done").read_text().strip() == "rc=0"
    beat = (rdv / "ray_head_alive").read_text()
    assert abs(int(beat) - time.time()) < 10
    time.sleep(1)
    assert (rdv / "ray_head_alive").read_text() == beat                          # the heartbeat died too


def test_a_worker_that_cannot_join_gives_up(stub_bin: StubBin):
    stub_bin.add("ray", "exit 1")
    r = sh(f"source {LIB}/ray_cluster.sh; ray_worker_wait_and_exit 2 8 127.0.0.1 1", stub_bin, RAY_JOIN_ATTEMPTS="2")
    assert r.returncode == 1 and "could not join Ray head" in r.stdout


def test_a_worker_leaves_as_soon_as_the_head_is_done(tmp_path: Path, stub_bin: StubBin, open_port: int):
    stub_bin.add("ray")
    rdv = tmp_path / "rdv"
    rdv.mkdir()
    (rdv / "ray_head_done").write_text("rc=0\n")
    t0 = time.time()
    r = sh(f"source {LIB}/ray_cluster.sh; ray_worker_wait_and_exit 2 8 127.0.0.1 1", stub_bin,
           VOA_RDV_DIR=str(rdv), RAY_PORT=str(open_port))
    assert r.returncode == 0 and "head done" in r.stdout and time.time() - t0 < 10


def test_a_worker_whose_head_died_without_cleanup_does_not_wait_for_the_timeout(tmp_path: Path, stub_bin: StubBin,
                                                                                 open_port: int):
    stub_bin.add("ray")
    rdv = tmp_path / "rdv"
    rdv.mkdir()
    (rdv / "ray_head_alive").write_text(str(int(time.time()) - 1000))   # port still open, script long gone
    r = sh(f"source {LIB}/ray_cluster.sh; ray_worker_wait_and_exit 2 8 127.0.0.1 1", stub_bin,
           VOA_RDV_DIR=str(rdv), RAY_PORT=str(open_port), RAY_HEARTBEAT_STALE_S="600")
    assert r.returncode == 1 and "last heartbeat is" in r.stdout


def test_a_worker_whose_head_is_gone_exits_cleanly(stub_bin: StubBin):
    stub_bin.add("ray")
    with socket.socket() as s:                   # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = sh(f"source {LIB}/ray_cluster.sh; ray_worker_wait_and_exit 2 8 127.0.0.1 1", stub_bin, RAY_PORT=str(port))
    assert r.returncode == 0 and "head gone" in r.stdout


# --- rendezvous (engine/lib/rendezvous.sh) ----------------------------------------------------------
def test_a_rendezvous_file_from_an_earlier_attempt_is_not_taken_for_this_one(tmp_path: Path, stub_bin: StubBin):
    f = tmp_path / "judge_endpoint"
    f.write_text("http://10.0.0.9:8000/v1\n")
    old = time.time() - 3600
    os.utime(f, (old, old))
    r = sh(f"source {LIB}/rendezvous.sh; DISPATCH_T0=$(date +%s); rdv_wait {f} 1", stub_bin, RDV_POLL_S="0.2")
    assert r.returncode == 1                                               # stale: ignored until the deadline
    r = sh(f"source {LIB}/rendezvous.sh; DISPATCH_T0=$(date +%s); rdv_put {f} http://10.0.0.1:8000/v1; "
           f"rdv_wait {f} 1", stub_bin, RDV_POLL_S="0.2")
    assert r.returncode == 0 and r.stdout.strip().endswith("http://10.0.0.1:8000/v1")
    assert not list(tmp_path.glob("*.tmp.*"))


def test_training_waits_for_the_judges_whole_budget(tmp_path: Path):
    math = REPO / "usecases/math/air/4_train.yaml"
    r = cc.render(math, tmp_path, 29870, {"JUDGE_WAIT_TIMEOUT": "2400"})       # the old, shorter number
    assert r["returncode"] != 0 and "shorter than the judge's own budget" in r["stderr"]
    ok = cc.render(math, tmp_path, 29880)
    assert "training waits <=6600s for the endpoint" in ok["stdout"]        # 3600 staging + 2400 health + 600


# --- the judge server (engine/serve/serve_judge.sh) -----------------------------------------------
def judge(tmp_path: Path, stub_bin: StubBin, server: str, *, timeout: float = 30, **env: str):
    stub_bin.add("vllm", server)
    stub_bin.add("curl", "exit 0")                            # /health answers at once
    port = str(20000 + os.getpid() % 20000)
    e = {**dict(JUDGE_ENGINE="vllm", JUDGE_MODEL_ID="org/judge", JUDGE_PORT=port, JUDGE_NNODES="1",
                JUDGE_RENDEZVOUS=str(tmp_path / "rdv" / "judge_endpoint"), JUDGE_WATCH_POLL_S="0.2",
                JUDGE_EXIT_SENTINEL=str(tmp_path / "rdv" / "training_done")), **env}
    return run(["bash", str(ENGINE / "serve/serve_judge.sh")], env=stub_bin.env(**e), timeout=timeout)


def test_the_judge_exits_0_when_training_says_it_is_done(tmp_path: Path, stub_bin: StubBin):
    (tmp_path / "rdv").mkdir()
    (tmp_path / "rdv" / "training_done").write_text("done\n")
    r = judge(tmp_path, stub_bin, "sleep 30")
    assert r.returncode == 0 and "stopped because training is done" in r.stdout
    assert (tmp_path / "rdv" / "judge_endpoint").read_text().strip().endswith("/v1")      # published, whole
    assert not list((tmp_path / "rdv").glob("*.tmp.*"))


def test_a_judge_that_dies_while_serving_is_a_judge_failure(tmp_path: Path, stub_bin: StubBin):
    r = judge(tmp_path, stub_bin, "sleep 1; exit 7")
    assert r.returncode == 3 and "died while serving (rc=7)" in r.stdout


def test_a_judge_that_outlives_its_lifetime_says_so(tmp_path: Path, stub_bin: StubBin):
    r = judge(tmp_path, stub_bin, "sleep 30", JUDGE_MAX_LIFETIME="1")
    assert r.returncode == 4 and "JUDGE_MAX_LIFETIME=1s ran out" in r.stdout


def test_staging_has_its_own_deadline(tmp_path: Path, stub_bin: StubBin):
    model = fake_hf_model(tmp_path / "judge-model")
    stub_bin.add("cp", "sleep 20")
    r = judge(tmp_path, stub_bin, "sleep 30", JUDGE_MODEL_PATH=str(model), JUDGE_MODEL_ID="",
              JUDGE_LOCAL_CACHE=str(tmp_path / "nvme"), JUDGE_STAGE_TIMEOUT="1")
    assert r.returncode == 1 and "did not finish within JUDGE_STAGE_TIMEOUT=1s" in r.stdout
    assert not stub_bin.calls("vllm")


def test_multi_node_sglang_is_refused_up_front(tmp_path: Path, stub_bin: StubBin):
    stub_bin.add("ray")
    r = judge(tmp_path, stub_bin, "sleep 30", JUDGE_ENGINE="sglang", JUDGE_NNODES="2")
    assert r.returncode == 1 and "no native SGLang multi-node launch" in r.stdout
    assert not stub_bin.calls("ray") and not stub_bin.calls("vllm")


# --- the pre-train judge ping (engine/serve/judge_ping.py) ----------------------------------------
@pytest.mark.parametrize("served,completion,want", [
    ("judge", (200, chat_completion("OK"), 0), 0),
    ("other", (200, chat_completion("OK"), 0), 1),                    # listed under another name
    ("judge", (500, {"error": "boom"}, 0), 1),                        # listed, cannot generate
])
def test_the_judge_must_answer_before_training_starts(served, completion, want):
    def reply(path, payload, n):
        if path.endswith("/models"):
            return 200, {"data": [{"id": served}]}, 0
        return completion
    with FakeOpenAIServer(reply) as srv:
        r = run(["python3", str(ENGINE / "serve/judge_ping.py"), srv.url, "judge"],
                env={**os.environ, "PING_TIMEOUT_S": "0"})
    assert r.returncode == want, r.stdout
