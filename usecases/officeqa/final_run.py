#!/usr/bin/env python3
"""Freeze, run and audit the 100-update OfficeQA experiment.

Run ``prepare --experiment <unique-name> --profile <profile>`` from a committed checkout,
then ``run --work-dir <prepared-directory>`` in a persistent terminal. ``status``
reads local receipts without contacting AIR. See README.md for the Make commands.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
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
RECORDED_EXPERIMENT = "officeqa-final-20261011-v1"
JOB_FILES = {
    "prep": "1_prep_final.yaml",
    "baseline": "3_baseline_final.yaml",
    "train": "4_train_final.yaml",
    "train-export": "4_export_final.yaml",
    "eval": "5_eval_final.yaml",
}
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


def experiment_name(value: str) -> str:
    """Keep experiment IDs safe as directory names, RUN_IDs and idempotency keys."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("experiment must be 1-80 letters, digits, underscores or hyphens, starting with a letter or digit")
    if value == RECORDED_EXPERIMENT:
        raise ValueError("the published experiment ID is reserved; choose a new name")
    return value


def stage_specs(source: Path, experiment: str, git_sha: str) -> dict[str, dict]:
    """Render a new run from the job templates without changing the worker policy.

    Workspace paths and the image come from the retargeted YAML, while the
    experiment name separates data, receipts and checkpoints from earlier runs.
    """
    experiment_name(experiment)
    jobs = {name: yaml.safe_load((source / "usecases/officeqa/air" / filename).read_text())
            for name, filename in JOB_FILES.items()}
    prep_env = jobs["prep"]["env_variables"]
    data_dir = str(Path(prep_env["OQ_DATA_DIR"]).parent / experiment)
    artifact_root = prep_env["OQ_ARTIFACT_ROOT"].rstrip("/")
    train_id = experiment + "-train"
    checkpoint_base = jobs["train"]["parameters"]["output_dir"].rstrip("/")
    checkpoint_root = checkpoint_base + "/" + train_id
    for name, spec in jobs.items():
        spec["code_source"]["snapshot"]["root_path"] = str(source)
        spec["env_variables"].update(
            RUN_ID=experiment + "-" + name, GIT_SHA=git_sha, OQ_DATA_DIR=data_dir, OQ_ARTIFACT_ROOT=artifact_root,
            VOA_IMAGE=spec["environment"]["docker_image"]["url"])
        if name == "train":
            spec["parameters"].update(train_files=data_dir + "/train.parquet", val_files=data_dir + "/test.parquet")
        elif name == "eval":
            spec["env_variables"]["EVAL_MODEL_PATH"] = checkpoint_root + "/global_step_100"
            spec["env_variables"]["EVAL_CKPT_ROOT"] = checkpoint_base
        elif name == "train-export":
            spec["env_variables"].update(OQ_EXPORT_DIR=f"{artifact_root}/{train_id}", OQ_EXPORT_CKPT_DIR=checkpoint_root)
    return jobs


def prepare(experiment: str, profile: str):
    experiment_name(experiment)
    if PLAN.exists():
        raise RuntimeError("this experiment is already frozen; use run/status or choose a new experiment ID")
    if command(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise RuntimeError("commit the experiment before freezing its source")
    sha = command(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "source.tar"
    source = ROOT / "source"
    if source.exists() or archive.exists():
        raise RuntimeError("an incomplete preparation exists here; choose a new experiment ID or work directory")
    subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", sha], cwd=REPO, check=True)
    source.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter="data")
    stages = {}
    jobs = stage_specs(source, experiment, sha)
    artifact_root = jobs["prep"]["env_variables"]["OQ_ARTIFACT_ROOT"].rstrip("/")
    train_id = experiment + "-train"
    checkpoint_root = jobs["train"]["parameters"]["output_dir"].rstrip("/") + "/" + train_id
    checkpoint = checkpoint_root + "/global_step_100"
    images = {spec["environment"]["docker_image"]["url"] for spec in jobs.values()}
    locked_images = json.loads((source / "docker/IMAGE.lock").read_text())["images"].values()
    image = next((entry for entry in locked_images if images == {entry["ref"]}), None)
    if image is None:
        raise RuntimeError("all stages must use one image recorded in docker/IMAGE.lock; build/push and record your image first")
    for name, spec in jobs.items():
        filename = JOB_FILES[name]
        path = source / "usecases" / "officeqa" / "air" / filename
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        dry = command(["air", "run", "--file", str(path), "--profile", profile, "--json", "--dry-run"], cwd=source)
        (ROOT / f"{name}-dry-run.jsonl").write_text(dry)
        if name == "train":
            preflight = command([sys.executable, str(source / "scripts" / "preflight_job.py"), str(path)], cwd=source)
            (ROOT / "train-preflight.json").write_text(preflight)
        stages[name] = {"file": str(path), "idempotency_key": "officeqa-" + experiment + "-" + name,
                        "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "timeout_minutes": spec["timeout_minutes"], "gpus": spec["compute"]["num_accelerators"],
                        "max_retries": int(spec.get("max_retries") or 0),
                        "artifact_dir": f"{artifact_root}/{experiment}-{name}"}
    plan = {"created_at": now(), "experiment": experiment, "profile": profile, "git_sha": sha,
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "source": str(source),
            "image": image, "data_dir": yaml.safe_load(Path(stages["prep"]["file"]).read_text())["env_variables"]["OQ_DATA_DIR"],
            "checkpoint_root": checkpoint_root, "checkpoint": checkpoint, "total_updates": 100,
            "training_prompt_groups": 400, "training_trajectories": 1600, "train_questions": 203,
            "hard_train_questions": 90, "hard_heldout_questions": 43, "eval_episodes": 236,
            "heldout_used_for_selection": False, "stages": stages,
            "gpu_hours_upper_bound": sum(stage["gpus"] * stage["timeout_minutes"] / 60 * (1 + stage["max_retries"])
                                         for stage in stages.values())}
    write_json(PLAN, plan)
    write_json(STATE, {"created_at": now(), "status": "PREPARED", "stages": {}, "plan": str(PLAN)})
    print(json.dumps({"prepared": str(PLAN), "git_sha": sha, "gpu_hours_upper_bound": plan["gpu_hours_upper_bound"]}), flush=True)


def stage_run(plan, state, name):
    profile = plan["profile"]
    stage = plan["stages"][name]
    path = Path(stage["file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != stage["yaml_sha256"]:
        raise RuntimeError(f"frozen {name} YAML changed")
    state.update(status="RUNNING", phase=name, updated_at=now())
    write_json(STATE, state)
    if name not in state["stages"]:
        subprocess.run([sys.executable, str(Path(plan["source"]) / "scripts" / "preflight_job.py"), "--budget-gate", str(path)],
                       cwd=plan["source"], check=True)
        receipt_path = ROOT / f"{name}-submission.jsonl"
        if receipt_path.is_file():
            raw = receipt_path.read_text()
        else:
            raw = command(["air", "run", "--file", str(path), "--profile", profile, "--json",
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
        watcher = subprocess.Popen(["air", "logs", run_id, "--profile", profile, "--json"], stdout=out, stderr=error)
        try:
            while time.monotonic() < deadline:
                try:
                    status = next(row["data"] for row in envelopes(command(["air", "get", "run", run_id, "--profile", profile, "--json"])) if "data" in row)
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


def fetch(remote, name, *, profile):
    path = ROOT / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        command(["databricks", "fs", "cp", "dbfs:" + remote, str(temporary), "--profile", profile, "--overwrite"], timeout=180)
        temporary.replace(path)
    return json.loads(path.read_text())


def reviewed_training_completion(plan, receipt):
    """Keep AIR's failure visible while accepting separately proved complete training.

    This requires an explicit evidence review, not merely a failed job with a
    checkpoint directory. It never resumes, repeats or changes the worker run.
    """
    review = json.loads((ROOT / "training-completion-review.json").read_text())
    if review.get("profile") != plan["profile"] or review.get("run_id") != receipt["run_id"] \
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
    profile = plan["profile"]
    if state["status"] == "COMPLETE":
        print((ROOT / "comparison.json").read_text())
        return
    if "error" in state:
        state.setdefault("controller_errors", []).append({"error": state.pop("error"), "time": state.pop("failed_at", None)})
    for name in JOB_FILES:
        stage_run(plan, state, name)
        if name == "prep":
            manifest = fetch(plan["data_dir"] + "/DATA_MANIFEST.json", "DATA_MANIFEST.json", profile=profile)
            qualification = fetch(plan["stages"][name]["artifact_dir"] + "/qualification.json", "qualification.json", profile=profile)
            if qualification["status"] != "PASS" or qualification["git_sha"] != plan["git_sha"]:
                raise ValueError("image/protocol/judge qualification failed or uses another source")
            state["qualified_versions"] = qualification["versions"]
        elif name == "baseline":
            before = fetch(plan["stages"][name]["artifact_dir"] + "/eval.json", "baseline.json", profile=profile)
            audit_eval(before, manifest)
            state["baseline_summary"] = before["summary"]
        elif name == "train":
            certificate = fetch(plan["checkpoint_root"] + "/run_result.json", "run_result.json", profile=profile)
            if not certificate["certified"] or certificate["raw_rc"] != 0 or certificate["final_rc"] != 0 \
                    or certificate["expected_final_version"] != 100 or certificate["observed_version"] != "100":
                raise ValueError("training did not publish a certified step-100 checkpoint")
            state["checkpoint_identity"] = certificate["checkpoint"]["identity"]
        elif name == "train-export":
            payload = export_payload(name)
            selection = payload["files"]["signal-data/DATA_MANIFEST.json"]
            if selection["selection"] != "all_official_training_questions" or selection["heldout_used_for_selection"] \
                    or set(selection["selected_training_uids"]) != set(manifest["split"]["train_ids"]):
                raise ValueError("training did not use the frozen experiment split")
            signal = payload["training_signal"]
            if signal["infrastructure_errors"]:
                raise ValueError("training has unresolved infrastructure rewards")
            if any(row["question_uid"] not in manifest["split"]["train_ids"] for group in signal["groups"].values() for row in group):
                raise ValueError("training scored a held-out or unknown question")
            state["training_signal"] = {key: value for key, value in signal.items() if key != "groups"}
        elif name == "eval":
            after = fetch(plan["stages"][name]["artifact_dir"] + "/eval.json", "final.json", profile=profile)
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


def status():
    """Show persisted progress without submitting jobs, taking a lock or writing files."""
    plan = json.loads(PLAN.read_text())
    state = json.loads(STATE.read_text())
    result = {"experiment": plan["experiment"], "profile": plan["profile"], "status": state["status"],
              "work_dir": str(ROOT), "gpu_hours_upper_bound": plan["gpu_hours_upper_bound"],
              "stages": {name: {key: receipt[key] for key in ("run_id", "status") if key in receipt}
                         for name, receipt in state["stages"].items()}}
    for key in ("error", "checkpoint_identity", "comparison"):
        if key in state:
            result[key] = state[key]
    print(json.dumps(result, indent=2))


def main(argv=None):
    global ROOT, PLAN, STATE
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare", help="freeze source and dry-run all stages; no GPU submission")
    prep.add_argument("--experiment", required=True, help="unique experiment name (letters, digits, underscores, hyphens)")
    prep.add_argument("--profile", required=True, help="Databricks profile to save in the experiment plan")
    prep.add_argument("--work-dir", type=Path, help="local evidence directory (default: .cache/officeqa-final/<experiment>)")
    for mode in ("run", "status"):
        child = modes.add_parser(mode, help="submit/resume the frozen plan" if mode == "run" else "read local progress")
        child.add_argument("--work-dir", type=Path, required=True,
                           help="directory containing the prepared experiment plan")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        try:
            experiment_name(args.experiment)
        except ValueError as error:
            parser.error(str(error))
        ROOT = (args.work_dir or REPO / ".cache/officeqa-final" / args.experiment).resolve()
    else:
        ROOT = args.work_dir.resolve()
    PLAN, STATE = ROOT / "plan.json", ROOT / "state.json"
    if args.mode == "status":
        status()
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(args.experiment, args.profile) if args.mode == "prepare" else run()
        except Exception as error:
            if args.mode == "run" and STATE.exists():
                state = json.loads(STATE.read_text())
                state.update(status="FAILED", error=f"{type(error).__name__}: {error}", failed_at=now())
                write_json(STATE, state)
            raise


if __name__ == "__main__":
    main()
