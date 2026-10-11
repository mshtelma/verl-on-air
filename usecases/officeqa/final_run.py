#!/usr/bin/env python3
"""Freeze, run and audit the final OfficeQA experiment on the explicitly selected df1 profile.

Run ``prepare`` from a clean, committed checkout, then ``run`` in a persistent terminal.
Receipts, logs, frozen source and evaluations live under .cache/officeqa-final/run/.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zlib

import yaml

import metrics

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / ".cache" / "officeqa-final" / "run"
PLAN = ROOT / "plan.json"
STATE = ROOT / "state.json"
PROFILE = "df1"
EXPERIMENT = "officeqa-final-20261011-v1"
TERMINAL_FAILURES = {"FAILED", "CANCELED", "CANCELLED", "TIMED_OUT", "SKIPPED", "INTERNAL_ERROR"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def envelopes(raw):
    result = []
    for line in raw.splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def submission_receipt(raw):
    for envelope in envelopes(raw):
        candidate = envelope.get("data", envelope)
        if isinstance(candidate, dict) and str(candidate.get("run_id", "")).isdigit():
            return candidate
    raise RuntimeError("AIR returned no structured submission receipt")


def command(args, *, cwd=None, timeout=120):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}")
    return result.stdout


def prepare():
    if PLAN.exists():
        raise RuntimeError("the final experiment is already frozen; run it or use a new experiment ID")
    if command(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("commit the experiment before freezing its source")
    sha = command(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "source.tar"
    subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", sha], cwd=REPO, check=True)
    source = ROOT / "source"
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    stages = {}
    artifact_root = "/Volumes/main/mshtelma/verl/eval/officeqa-final"
    train_id = EXPERIMENT + "-train"
    checkpoint_root = f"/Volumes/main/mshtelma/verl/ckpt/qwen3_5-35b-officeqa-final/{train_id}"
    checkpoint = checkpoint_root + "/global_step_100"
    names = {"prep": "1_prep_final.yaml", "baseline": "3_baseline_final.yaml",
             "train": "4_train_final.yaml", "eval": "5_eval_final.yaml", "train-export": "2_toolcheck.yaml"}
    for name, filename in names.items():
        path = source / "usecases" / "officeqa" / "air" / filename
        spec = yaml.safe_load(path.read_text())
        spec["code_source"]["snapshot"]["root_path"] = str(source)
        spec["env_variables"].update(RUN_ID=EXPERIMENT + "-" + name, GIT_SHA=sha,
                                     VOA_IMAGE=spec["environment"]["docker_image"]["url"])
        if name == "eval":
            spec["env_variables"]["EVAL_MODEL_PATH"] = checkpoint
        if name == "train-export":
            spec["experiment_name"] = "verl010-officeqa-final-train-export"
            spec["env_variables"].update(OQ_EXPORT_DIR=f"{artifact_root}/{train_id}", OQ_EXPORT_CKPT_DIR=checkpoint_root)
            spec["command"] = "python3 ${CODE_SOURCE_PATH}/usecases/officeqa/export_artifacts.py"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        dry = command(["air", "run", "--file", str(path), "--profile", PROFILE, "--json", "--dry-run"], cwd=source)
        (ROOT / f"{name}-dry-run.jsonl").write_text(dry)
        if name == "train":
            preflight = command([sys.executable, str(source / "scripts" / "preflight_job.py"), str(path)], cwd=source)
            (ROOT / "train-preflight.json").write_text(preflight)
        stages[name] = {"file": str(path), "idempotency_key": "mshtelma-" + EXPERIMENT + "-" + name,
                        "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "timeout_minutes": spec["timeout_minutes"], "gpus": spec["compute"]["num_accelerators"],
                        "artifact_dir": f"{artifact_root}/{EXPERIMENT}-{name}"}
    image = json.loads((source / "docker" / "IMAGE.lock").read_text())["images"]["v13-verl010-officeqa-bm25"]
    plan = {"created_at": now(), "experiment": EXPERIMENT, "profile": PROFILE, "git_sha": sha,
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "source": str(source),
            "image": image, "data_dir": yaml.safe_load(Path(stages["prep"]["file"]).read_text())["env_variables"]["OQ_DATA_DIR"],
            "checkpoint_root": checkpoint_root, "checkpoint": checkpoint, "total_updates": 100,
            "training_prompt_groups": 400, "training_trajectories": 1600, "train_questions": 203,
            "hard_train_questions": 90, "hard_heldout_questions": 43, "eval_episodes": 236,
            "heldout_used_for_selection": False, "stages": stages,
            "gpu_hours_upper_bound": sum(stage["gpus"] * stage["timeout_minutes"] / 60 for stage in stages.values())}
    write_json(PLAN, plan)
    write_json(STATE, {"created_at": now(), "status": "PREPARED", "stages": {}, "plan": str(PLAN)})
    print(json.dumps({"prepared": str(PLAN), "git_sha": sha, "gpu_hours_upper_bound": plan["gpu_hours_upper_bound"]}), flush=True)


def stage_run(plan, state, name):
    stage = plan["stages"][name]
    path = Path(stage["file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != stage["yaml_sha256"]:
        raise RuntimeError(f"frozen {name} YAML changed")
    state.update(status="RUNNING", phase=name, updated_at=now())
    write_json(STATE, state)
    if name not in state["stages"]:
        budget_environment = {**os.environ, "BUDGET_OK": "1"}
        subprocess.run([sys.executable, str(Path(plan["source"]) / "scripts" / "preflight_job.py"), "--budget-gate", str(path)],
                       env=budget_environment, cwd=plan["source"], check=True)
        receipt_path = ROOT / f"{name}-submission.jsonl"
        if receipt_path.is_file():
            raw = receipt_path.read_text()
        else:
            raw = command(["air", "run", "--file", str(path), "--profile", PROFILE, "--json",
                           "--idempotency-key", stage["idempotency_key"]], cwd=plan["source"], timeout=600)
            receipt_path.write_text(raw)
        submitted = submission_receipt(raw)
        state["stages"][name] = {"run_id": str(submitted["run_id"]), "status": "SUBMITTED", "submitted_at": now()}
        write_json(STATE, state)
        print(json.dumps({"stage": name, **state["stages"][name]}), flush=True)
    receipt = state["stages"][name]
    if receipt["status"] == "SUCCESS":
        return
    if name == "train" and receipt["status"] == "FAILED" and (ROOT / "training-completion-review.json").is_file():
        receipt["completion_review"] = reviewed_training_completion(plan, receipt)
        write_json(STATE, state)
        print(json.dumps({"stage": name, "status": "CERTIFIED", "air_status": "FAILED", "run_id": receipt["run_id"]}), flush=True)
        return
    run_id = receipt["run_id"]
    log_path = ROOT / f"{name}-logs.jsonl"
    log_error = ROOT / f"{name}-logs.stderr"
    deadline = time.monotonic() + (stage["timeout_minutes"] + 45) * 60
    failures = 0
    with log_path.open("a") as out, log_error.open("a") as error:
        watcher = subprocess.Popen(["air", "logs", run_id, "--profile", PROFILE, "--json"], stdout=out, stderr=error)
        try:
            while time.monotonic() < deadline:
                try:
                    status = next(row["data"] for row in envelopes(command(["air", "get", "run", run_id, "--profile", PROFILE, "--json"])) if "data" in row)
                    failures = 0
                except (RuntimeError, subprocess.TimeoutExpired, StopIteration):
                    failures += 1
                    if failures > 12:
                        raise
                    time.sleep(20)
                    continue
                receipt.update(status)
                state["updated_at"] = now()
                write_json(STATE, state)
                if status["status"] == "SUCCESS":
                    print(json.dumps({"stage": name, "status": "SUCCESS", "run_id": run_id}), flush=True)
                    return
                if status["status"] in TERMINAL_FAILURES:
                    if name == "train" and status["status"] == "FAILED" and (ROOT / "training-completion-review.json").is_file():
                        receipt["completion_review"] = reviewed_training_completion(plan, receipt)
                        write_json(STATE, state)
                        return
                    raise RuntimeError(f"{name} ended {status['status']}: {status.get('termination_reason')}")
                time.sleep(20)
            raise TimeoutError(f"{name} exceeded its remote timeout plus the monitoring margin")
        finally:
            # A completed-run log fetch returns itself. Allow the last envelopes to flush.
            try:
                watcher.wait(timeout=30)
            except subprocess.TimeoutExpired:
                watcher.terminate()
                watcher.wait(timeout=15)


def fetch(remote, name):
    path = ROOT / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        command(["databricks", "fs", "cp", "dbfs:" + remote, str(temporary), "--profile", PROFILE, "--overwrite"], timeout=180)
        temporary.replace(path)
    return json.loads(path.read_text())


def reviewed_training_completion(plan, receipt):
    """Keep AIR's failure visible while accepting separately proved complete training.

    This requires an explicit evidence review, not merely a failed job with a
    checkpoint directory. It never resumes, repeats or changes the worker run.
    """
    review = json.loads((ROOT / "training-completion-review.json").read_text())
    if review.get("profile") != PROFILE or review.get("run_id") != receipt["run_id"] \
            or review.get("source_git_sha") != plan["git_sha"] or receipt["status"] != "FAILED":
        raise ValueError("training completion review has a different run, source or status")
    files = {}
    for name in ("certificate", "trace_audit", "node0", "node1", "air_task_output"):
        evidence = review["evidence"][name]
        path = ROOT / evidence["file"]
        if path.parent != ROOT or hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError(f"training completion evidence changed: {name}")
        files[name] = path.read_text()
    certificate = json.loads(files["certificate"])
    if not certificate.get("certified") or certificate["raw_rc"] != 0 or certificate["final_rc"] != 0 \
            or certificate.get("abort") or certificate.get("problems") or certificate.get("hard_error_matches") \
            or certificate["git_sha"] != plan["git_sha"] or certificate["run_id"] != plan["experiment"] + "-train" \
            or certificate["expected_final_version"] != plan["total_updates"] \
            or str(certificate["observed_version"]) != str(plan["total_updates"]) \
            or certificate["checkpoint"]["step"] != plan["total_updates"]:
        raise ValueError("training completion requires a clean final checkpoint certificate")
    audit = json.loads(files["trace_audit"])
    if audit["updates"] != plan["total_updates"] or audit["trajectories"] != plan["training_trajectories"] \
            or audit["groups"] != plan["training_prompt_groups"] or audit["unique_training_questions"] != plan["train_questions"] \
            or audit["infrastructure_errors"] or audit["heldout_questions"] or audit["unobserved_training_uids"]:
        raise ValueError("training completion requires every planned, permitted trajectory")
    for node in (0, 1):
        rows = [r for r in envelopes(files[f"node{node}"]) if r.get("node") == node]
        lines = [r.get("line", "") for r in rows]
        if not any(line.strip() == "+ exit 0" for line in lines) or any(line.strip() == "+ exit 1" for line in lines) \
                or not any("Execution Summary" in line for line in lines) or not any("Completed at " in line for line in lines):
            raise ValueError(f"training completion lacks a clean node-{node} command exit")
        marker = "[certificate] CERTIFIED:" if node == 0 else "head done (rc=0); exiting worker cleanly"
        if not any(marker in line for line in lines):
            raise ValueError(f"training completion lacks the node-{node} completion marker")
    output = json.loads(files["air_task_output"])
    details = output["metadata"]["status"]["termination_details"]
    if output.get("error") or output.get("error_trace") or output["metadata"]["job_run_id"] != int(receipt["run_id"]) \
            or output["metadata"]["state"]["result_state"] != "FAILED" \
            or details["type"] != "CLIENT_ERROR" or details["code"] != "RUN_EXECUTION_ERROR":
        raise ValueError("AIR reports an additional execution failure requiring investigation")
    return {"workload_completed": True, "air_status": "FAILED", "checkpoint_identity": certificate["checkpoint"]["identity"],
            "review_file": str(ROOT / "training-completion-review.json"), "evidence": review["evidence"],
            "reason": review["reason"], "reviewed_at": review["reviewed_at"]}


def audit_eval(artifact, manifest):
    if artifact.get("valid") is not True or artifact["n_scored"] != 236 or len(artifact["results"]) != 236:
        raise ValueError("final evaluation must score all 236 predetermined episodes")
    heldout = [row for row in artifact["results"] if row["split"] == "heldout"]
    if len(heldout) != 172 or {row["uid"] for row in heldout} != set(manifest["split"]["heldout_ids"]):
        raise ValueError("evaluation omitted or substituted a held-out question")
    if any(count != 4 for count in Counter(row["uid"] for row in heldout).values()):
        raise ValueError("each held-out question must have four samples")
    if any(row["status"] != "scored" or row["metrics"]["infrastructure_error"] for row in artifact["results"]):
        raise ValueError("evaluation contains infrastructure failures")


def export_payload(name):
    raw_log = (ROOT / f"{name}-logs.jsonl").read_text()
    chunks, receipt = {}, None
    for envelope in envelopes(raw_log):
        row = envelope
        if "line" in envelope:
            try:
                row = json.loads(envelope["line"])
            except ValueError:
                continue
        if "officeqa_artifact_chunk" in row:
            index = row["officeqa_artifact_chunk"]
            if index in chunks and chunks[index] != row["data"]:
                raise ValueError("conflicting export chunks")
            chunks[index] = row["data"]
        if row.get("officeqa_export") == "PASS":
            receipt = row
    if not receipt or set(chunks) != set(range(receipt["chunks"])):
        raise RuntimeError("training evidence export is incomplete; fetch the completed run's logs")
    raw = zlib.decompress(base64.b64decode("".join(chunks[i] for i in range(receipt["chunks"])), validate=True))
    if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("training evidence export checksum mismatch")
    (ROOT / "training-artifacts.json").write_bytes(raw)
    return json.loads(raw)


def run():
    plan = json.loads(PLAN.read_text())
    state = json.loads(STATE.read_text())
    if state["status"] == "COMPLETE":
        print((ROOT / "comparison.json").read_text())
        return
    if "error" in state:
        state.setdefault("controller_errors", []).append({"error": state.pop("error"), "time": state.pop("failed_at", None)})
    for name in ("prep", "baseline", "train", "train-export", "eval"):
        stage_run(plan, state, name)
        if name == "prep":
            manifest = fetch(plan["data_dir"] + "/DATA_MANIFEST.json", "DATA_MANIFEST.json")
            qualification = fetch(plan["stages"][name]["artifact_dir"] + "/qualification.json", "qualification.json")
            if qualification["status"] != "PASS" or qualification["git_sha"] != plan["git_sha"]:
                raise ValueError("image/protocol/judge qualification failed or uses another source")
            state["qualified_versions"] = qualification["versions"]
        elif name == "baseline":
            before = fetch(plan["stages"][name]["artifact_dir"] + "/eval.json", "baseline.json")
            audit_eval(before, manifest)
            state["baseline_summary"] = before["summary"]
        elif name == "train":
            certificate = fetch(plan["checkpoint_root"] + "/run_result.json", "run_result.json")
            if not certificate["certified"] or certificate["raw_rc"] != 0 or certificate["final_rc"] != 0 \
                    or certificate["expected_final_version"] != 100 or certificate["observed_version"] != "100":
                raise ValueError("training did not publish a certified step-100 checkpoint")
            state["checkpoint_identity"] = certificate["checkpoint"]["identity"]
        elif name == "train-export":
            payload = export_payload(name)
            selection = payload["files"]["signal-data/DATA_MANIFEST.json"]
            if selection["selection"] != "all_official_training_questions" or selection["heldout_used_for_selection"] \
                    or set(selection["selected_training_uids"]) != set(manifest["split"]["train_ids"]):
                raise ValueError("training did not use the frozen official split")
            signal = payload["training_signal"]
            if signal["infrastructure_errors"]:
                raise ValueError("training has unresolved infrastructure rewards")
            if any(row["question_uid"] not in manifest["split"]["train_ids"] for group in signal["groups"].values() for row in group):
                raise ValueError("training scored a held-out or unknown question")
            state["training_signal"] = {key: value for key, value in signal.items() if key != "groups"}
        elif name == "eval":
            after = fetch(plan["stages"][name]["artifact_dir"] + "/eval.json", "final.json")
            audit_eval(after, manifest)
            if after["model_identity"]["identity"] != state["checkpoint_identity"]:
                raise ValueError("final evaluation served a different checkpoint")
            comparison = metrics.compare(before, after)
            comparison.update(git_sha=plan["git_sha"], checkpoint=plan["checkpoint"],
                              checkpoint_identity=state["checkpoint_identity"], runs=state["stages"],
                              training_signal=state["training_signal"], qualified_versions=state["qualified_versions"])
            write_json(ROOT / "comparison.json", comparison)
            state.update(status="COMPLETE", finished_at=now(), comparison=str(ROOT / "comparison.json"))
        write_json(STATE, state)
    print(json.dumps({"officeqa_final": "COMPLETE", "comparison": str(ROOT / "comparison.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare() if args.mode == "prepare" else run()
        except Exception as error:
            if STATE.exists():
                state = json.loads(STATE.read_text())
                state.update(status="FAILED", error=f"{type(error).__name__}: {error}", failed_at=now())
                write_json(STATE, state)
            raise
