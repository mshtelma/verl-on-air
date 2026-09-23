"""Stands in for `python -m verl.experimental.fully_async_policy.fully_async_main` in the exit-guard
tests: FAKE_SCENARIO picks what the "training run" leaves behind and how it exits. Checkpoints are
written the way verl writes them -- the pieces, then actor/ckpt_contents.json, then the tracker."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                            # tests/ -> support
sys.path.insert(0, str(HERE.parents[1] / "engine" / "lib"))     # run_control
from support import fake_train_checkpoint  # noqa: E402

import run_control  # noqa: E402

CKPT = Path(os.environ["FAKE_CKPT_DIR"])
FINAL = int(os.environ["FAKE_FINAL"])


def save(step: int) -> None:
    fake_train_checkpoint(CKPT, step)
    (CKPT / "latest_checkpointed_iteration.txt").write_text(str(step))


def say(*lines: str) -> None:
    print("\n".join(lines), flush=True)


def main() -> int:
    sc = os.environ["FAKE_SCENARIO"]
    if sc == "exception_with_finally_marker":            # REVIEW.md R02 reproduction 1
        say("Traceback (most recent call last):", "ValueError: injected training failure",
            "[ASYNC MAIN] Training completed or interrupted")
        return 42
    if sc == "partial_dir_then_disk_full":                # REVIEW.md R02 reproduction 2
        (CKPT / "global_step_10" / "actor").mkdir(parents=True)
        say("OSError: [Errno 28] No space left on device")
        return 42
    if sc == "crash_after_intermediate_save":
        save(FINAL // 2)
        say("Traceback (most recent call last):", "RuntimeError: rollout worker died")
        return 1
    if sc == "swallowed_crash_exits_zero":                # rollouter gather(return_exceptions=True)
        save(FINAL // 2)
        say("[FullyAsyncTrainer] Training stopped by queue termination signal",
            "[ASYNC MAIN] One component completed successfully")
        return 0
    if sc == "nothing_new":
        return 0
    if sc == "final_without_manifest":
        fake_train_checkpoint(CKPT, FINAL, manifest=False)
        (CKPT / "latest_checkpointed_iteration.txt").write_text(str(FINAL))
        return 1
    if sc == "complete_but_aborted":
        save(FINAL)
        run_control.request_abort("judge failure budget exhausted", "test-reward-worker")
        return 0
    if sc == "complete_then_teardown_cancelled":
        save(FINAL)
        say("ray.exceptions.RayTaskError: RuntimeError: cancelled")
        return 1
    if sc == "complete_then_oom":
        save(FINAL)
        say("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        return 1
    if sc == "complete_clean":
        save(FINAL)
        return 0
    if sc == "hang":
        say("[fake] training forever")
        time.sleep(120)
        return 0
    raise SystemExit(f"unknown FAKE_SCENARIO {sc!r}")


if __name__ == "__main__":
    sys.exit(main())
