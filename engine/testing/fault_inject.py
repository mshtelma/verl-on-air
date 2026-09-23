#!/usr/bin/env python3
"""ACCEPTANCE TESTING ONLY: kill the Trainer after its first checkpoint (acceptance run A5b).

    fault_inject.py kill-trainer-after-save <ckpt_dir> [--step N] [--timeout-s S]

Started in the background by run_grpo_fully_async.sh when FAULT_INJECT=kill-trainer-after-save.
Waits until ``<ckpt_dir>/global_step_<N>/actor/ckpt_contents.json`` exists (rank 0 writes it last),
then SIGKILLs the FullyAsyncTrainer actor's process on whichever node it runs. The run cannot then
reach its final version, and A5b passes when the launcher reports FAILED with the checkpoint
version it did reach. Never set FAULT_INJECT on a real training job.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


def _kill_pid(pid: int) -> str:
    try:
        os.kill(pid, signal.SIGKILL)
        return f"killed pid {pid}"
    except ProcessLookupError:
        return f"pid {pid} already gone"


def kill_trainer(timeout_s: float) -> str:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.util.state import list_actors

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        actors = list_actors(filters=[("class_name", "=", "FullyAsyncTrainer"), ("state", "=", "ALIVE")])
        if actors:
            a = actors[0]
            killer = ray.remote(num_cpus=0)(_kill_pid).options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=a.node_id, soft=False))
            return ray.get(killer.remote(int(a.pid)), timeout=120)
        time.sleep(5)
    raise TimeoutError("no ALIVE FullyAsyncTrainer actor found")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["kill-trainer-after-save"])
    ap.add_argument("ckpt_dir")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=4 * 3600)
    args = ap.parse_args(argv)

    marker = Path(args.ckpt_dir) / f"global_step_{args.step}" / "actor" / "ckpt_contents.json"
    deadline = time.monotonic() + args.timeout_s
    while not marker.is_file():
        if time.monotonic() >= deadline:
            print(f"[fault_inject] {marker} never appeared; nothing injected", file=sys.stderr)
            return 1
        time.sleep(10)
    print(f"[fault_inject] {marker} exists -> killing the FullyAsyncTrainer (acceptance A5b)", flush=True)
    print(f"[fault_inject] {kill_trainer(600)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
