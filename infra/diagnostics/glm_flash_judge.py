#!/usr/bin/env python3
"""Serve the pinned Flash judge, then exercise the real math reward client.

Checks known grading cases and concurrent requests with strict final-content JSON.
The server and its children are stopped on success, failure, or interruption.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "usecases/math"))
sys.path.insert(0, str(ROOT / "infra/diagnostics"))
import judge_selfcheck  # noqa: E402
import probe_verdict  # noqa: E402
import reward  # noqa: E402


async def exercise() -> dict:
    concurrency = int(os.environ.get("GLM_FLASH_PROBE_CONCURRENCY", "16"))
    repeats = int(os.environ.get("GLM_FLASH_PROBE_REPEATS", "4"))
    if concurrency < 1 or repeats < 1:
        raise ValueError("probe concurrency and repeats must be positive")
    sem = asyncio.Semaphore(concurrency)

    async def grade(case):
        name, question, reference, working, want = case
        async with sem:
            started = time.monotonic()
            try:
                score = await asyncio.wait_for(
                    reward.call_judge(question, working, reference), reward._deadline_s()
                )
                ok = (score >= 0.5) == want
                result = {"case": name, "ok": ok, "score": score,
                          "latency_s": time.monotonic() - started}
            except (reward.JudgeError, asyncio.TimeoutError) as exc:
                result = {"case": name, "ok": False, "error": str(exc),
                          "latency_s": time.monotonic() - started}
            print("JUDGE_CASE " + json.dumps(result), flush=True)
            return result

    try:
        # Match the training dispatcher's gate before testing concurrent traffic.
        calibration = []
        for case in judge_selfcheck.CASES:
            calibration.append(await grade(case))
        if not all(r["ok"] for r in calibration):
            return {"ok": False, "calibration": calibration, "requests": []}
        started = time.monotonic()
        results = await asyncio.gather(*(grade(case) for case in judge_selfcheck.CASES * repeats))
        elapsed = time.monotonic() - started
        latency = sorted(r["latency_s"] for r in results)
        return {
            "ok": all(r["ok"] for r in results), "calibration": calibration,
            "requests": results, "concurrency": concurrency, "wall_s": elapsed,
            "requests_per_s": len(results) / elapsed,
            "median_latency_s": statistics.median(latency),
            "p95_latency_s": latency[min(len(latency) - 1, int(len(latency) * 0.95))],
        }
    finally:
        await reward.close_sessions()


def main() -> int:
    run_id = os.environ.get("RUN_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_root = os.environ.get("GLM_FLASH_PROBE_OUT_ROOT", "/tmp/glmflash-results")
    os.environ.setdefault("PROBE_VERDICT_OUT", str(Path(out_root) / run_id / "qualification.json"))
    report = {"ok": False, "model_path": os.environ["JUDGE_MODEL_PATH"],
              "vllm": importlib.metadata.version("vllm"),
              "transformers": importlib.metadata.version("transformers"),
              "image_tag": os.environ.get("VERL_ON_AIR_IMAGE_TAG"),
              "run_id": run_id,
              "git_sha": os.environ.get("GIT_SHA"),
              "reasoning_effort": os.environ.get("JUDGE_REASONING_EFFORT")}
    with tempfile.TemporaryDirectory(prefix="glmflash-judge-") as tmp:
        endpoint, stop = Path(tmp) / "judge_endpoint", Path(tmp) / "training_done"
        env = {**os.environ, "JUDGE_NNODES": "1", "JUDGE_RANK": "0",
               "JUDGE_RENDEZVOUS": str(endpoint), "JUDGE_EXIT_SENTINEL": str(stop),
               "VOA_RDV_DIR": tmp}
        startup_started = time.monotonic()
        server = subprocess.Popen(
            ["bash", str(ROOT / "engine/serve/serve_judge.sh")], env=env, start_new_session=True
        )
        try:
            startup_budget = (int(env.get("JUDGE_STAGE_TIMEOUT", "1200"))
                              + int(env.get("JUDGE_HEALTH_TIMEOUT", "600")) + 60)
            deadline = time.monotonic() + startup_budget
            next_progress = time.monotonic()
            server_log = Path(f"/tmp/judge_{env['JUDGE_ENGINE']}_{env.get('JUDGE_PORT', '8000')}.log")
            while not endpoint.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"judge exited before publishing its endpoint: {server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("judge startup exceeded the staging and health budgets")
                if time.monotonic() >= next_progress:
                    recent = []
                    if server_log.exists():
                        tail = subprocess.run(["tail", "-n", "4", str(server_log)],
                                              capture_output=True, text=True, timeout=10)
                        recent = tail.stdout.splitlines()[-3:]
                    print("JUDGE_STARTUP " + json.dumps({
                        "elapsed_s": round(time.monotonic() - startup_started),
                        "recent": recent,
                    }), flush=True)
                    next_progress = time.monotonic() + 30
                time.sleep(2)
            os.environ["JUDGE_BASE_URL"] = endpoint.read_text().strip()
            report["startup_s"] = time.monotonic() - startup_started
            print("judge endpoint ready", flush=True)
            report.update(asyncio.run(exercise()))
            if server.poll() is not None:
                raise RuntimeError(f"judge exited during qualification: {server.returncode}")
            report["gpu_status"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=20,
            ).stdout.strip().splitlines()
        except Exception as exc:
            report["ok"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(report["error"], flush=True)
        finally:
            stop.write_text("qualification finished\n")
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
    return probe_verdict.emit(
        "glm_flash_judge", report["ok"],
        reasons=[] if report["ok"] else [report.get("error", "judge calibration or concurrent grading failed")],
        report=report,
    )


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
