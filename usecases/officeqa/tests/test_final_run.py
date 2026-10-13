import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from support import load_module

controller = load_module(Path(__file__).resolve().parents[1] / "final_run.py")


def complete_evaluation():
    heldout = [f"hard-{i}" for i in range(43)]
    rows = [{"uid": uid, "id": f"{uid}/{sample}", "split": split, "status": "scored",
             "metrics": {"infrastructure_error": 0}}
            for split, uids in [("heldout", heldout), ("train_probe", [f"train-{i}" for i in range(16)])]
            for uid in uids for sample in range(4)]
    return {"valid": True, "n_scored": 236, "results": rows}, {"split": {"heldout_ids": heldout}}


def test_final_pipeline_rejects_a_missing_or_substituted_heldout_question():
    artifact, manifest = complete_evaluation()
    controller.audit_eval(artifact, manifest)
    changed = copy.deepcopy(artifact)
    changed["results"][0]["uid"] = "training-question"
    with pytest.raises(ValueError, match="substituted"):
        controller.audit_eval(changed, manifest)
    artifact["results"].pop()
    with pytest.raises(ValueError, match="236"):
        controller.audit_eval(artifact, manifest)


def test_final_pipeline_never_accepts_an_infrastructure_reward():
    artifact, manifest = complete_evaluation()
    artifact["results"][0]["metrics"]["infrastructure_error"] = 1
    with pytest.raises(ValueError, match="infrastructure"):
        controller.audit_eval(artifact, manifest)


def test_duplicate_samples_cannot_hide_an_incomplete_heldout_group():
    artifact, manifest = complete_evaluation()
    artifact["results"][0]["uid"] = "hard-1"
    with pytest.raises(ValueError, match="four samples"):
        controller.audit_eval(artifact, manifest)


@pytest.mark.parametrize("receipt", [
    {"v": 1, "ts": "2026-10-11T09:53:39Z", "data": {"run_id": "252353129579893", "status": "PENDING"}},
    {"type": "SUBMITTED", "run_id": 252353129579893},
    {"v": 1, "data": {"run_id": "252353129579893", "reused": True}},
])
def test_submission_receipts_cover_direct_watch_and_safe_retry_formats(receipt):
    result = controller.submission_receipt("[INFO] Using Profile: [customer-workspace]\n" + json.dumps(receipt))
    assert str(result["run_id"]) == "252353129579893"


def completion_review(tmp_path, monkeypatch, *, abort=None, trajectories=1600, worker_rc=0):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    plan = {"git_sha": "frozen-source", "profile": "customer-workspace", "experiment": "officeqa-test", "total_updates": 100,
            "training_trajectories": 1600, "training_prompt_groups": 400, "train_questions": 203}
    receipt = {"run_id": "123", "status": "FAILED"}
    certificate = {"certified": True, "raw_rc": 0, "final_rc": 0, "abort": abort,
                   "problems": [], "hard_error_matches": [], "git_sha": plan["git_sha"],
                   "run_id": "officeqa-test-train", "expected_final_version": 100,
                   "observed_version": "100", "checkpoint": {"step": 100, "identity": "trained-model"}}
    audit = {"updates": 100, "trajectories": trajectories, "groups": 400, "unique_training_questions": 203,
             "infrastructure_errors": 0, "heldout_questions": 0, "unobserved_training_uids": []}
    output = {"error": None, "error_trace": None,
              "metadata": {"job_run_id": 123, "state": {"result_state": "FAILED"},
                           "status": {"termination_details": {"type": "CLIENT_ERROR", "code": "RUN_EXECUTION_ERROR"}}}}
    values = {"certificate": json.dumps(certificate), "trace_audit": json.dumps(audit), "air_task_output": json.dumps(output)}
    for node, marker in ((0, "[certificate] CERTIFIED: version 100"), (1, "head done (rc=0); exiting worker cleanly")):
        code = worker_rc if node else 0
        values[f"node{node}"] = "\n".join(json.dumps({"node": node, "line": line}) for line in (
            marker, f"+ exit {code}", "Execution Summary", "Completed at 2026-10-11 20:23:40"))
    evidence = {}
    for name, text in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(text)
        evidence[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    review = {"profile": "customer-workspace", "run_id": "123", "source_git_sha": plan["git_sha"], "evidence": evidence,
              "reason": "Complete certified training; AIR returned a generic client error after clean exits.", "reviewed_at": "2026-10-11"}
    (tmp_path / "training-completion-review.json").write_text(json.dumps(review))
    return plan, receipt


def test_completed_training_can_be_evaluated_while_preserving_air_failure(tmp_path, monkeypatch):
    plan, receipt = completion_review(tmp_path, monkeypatch)
    result = controller.reviewed_training_completion(plan, receipt)
    assert result["workload_completed"] is True
    assert result["checkpoint_identity"] == "trained-model"
    assert result["air_status"] == receipt["status"] == "FAILED"


@pytest.mark.parametrize("changes", [
    {"abort": {"reason": "reward request failed"}},
    {"trajectories": 1584},
    {"worker_rc": 1},
])
def test_certificate_never_hides_abort_incomplete_training_or_worker_failure(tmp_path, monkeypatch, changes):
    plan, receipt = completion_review(tmp_path, monkeypatch, **changes)
    with pytest.raises(ValueError):
        controller.reviewed_training_completion(plan, receipt)


def test_completion_review_rejects_changed_evidence(tmp_path, monkeypatch):
    plan, receipt = completion_review(tmp_path, monkeypatch)
    (tmp_path / "certificate.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        controller.reviewed_training_completion(plan, receipt)


def test_new_experiments_use_the_retargeted_workspace_and_separate_data(tmp_path):
    jobs_dir = tmp_path / "usecases/officeqa/air"
    shutil.copytree(controller.REPO / "usecases/officeqa/air", jobs_dir)
    for path in jobs_dir.glob("*.yaml"):
        path.write_text(path.read_text().replace("/Volumes/main/mshtelma/verl", "/Volumes/team/research/runs"))
    first = controller.stage_specs(tmp_path, "first-seed", "committed-source")
    second = controller.stage_specs(tmp_path, "second-seed", "committed-source")
    data = first["prep"]["env_variables"]["OQ_DATA_DIR"]
    assert data == "/Volumes/team/research/runs/data/officeqa_miles/first-seed"
    assert all(job["env_variables"]["OQ_DATA_DIR"] == data for job in first.values())
    assert first["train"]["parameters"]["train_files"] == data + "/train.parquet"
    assert first["train"]["parameters"]["val_files"] == data + "/test.parquet"
    assert first["eval"]["env_variables"]["EVAL_MODEL_PATH"] == (
        "/Volumes/team/research/runs/ckpt/qwen3_5-35b-officeqa-final/first-seed-train/global_step_100")
    assert second["prep"]["env_variables"]["OQ_DATA_DIR"] != data
    assert second["eval"]["env_variables"]["EVAL_MODEL_PATH"] != first["eval"]["env_variables"]["EVAL_MODEL_PATH"]
    before = first["baseline"]["env_variables"].copy()
    after = first["eval"]["env_variables"].copy()
    before.pop("EVAL_MODEL_PATH")
    after.pop("EVAL_MODEL_PATH")
    after.pop("EVAL_CKPT_ROOT")
    before.pop("RUN_ID")
    after.pop("RUN_ID")
    assert before == after


@pytest.mark.parametrize("name", ["../another-run", "a/b", "a b", "", controller.RECORDED_EXPERIMENT])
def test_new_experiment_names_cannot_escape_their_directory_or_reuse_the_published_run(name):
    with pytest.raises(ValueError):
        controller.experiment_name(name)


def test_repreparing_an_existing_run_preserves_its_complete_state(tmp_path, monkeypatch):
    for key in ("ROOT", "PLAN", "STATE"):
        monkeypatch.setattr(controller, key, getattr(controller, key))
    (tmp_path / "plan.json").write_text("{}")
    state = json.dumps({"status": "COMPLETE", "stages": {}})
    (tmp_path / "state.json").write_text(state)
    with pytest.raises(RuntimeError, match="already frozen"):
        controller.main(["prepare", "--experiment", "my-run", "--profile", "customer-workspace", "--work-dir", str(tmp_path)])
    assert (tmp_path / "state.json").read_text() == state


@pytest.mark.parametrize("args", [["prepare", "--experiment", "my-run"], ["run"], ["status"]])
def test_cli_requires_an_explicit_workspace_or_prepared_plan_before_writing(args, tmp_path, monkeypatch):
    work = tmp_path / "experiment"
    monkeypatch.setattr(controller, "ROOT", work)
    with pytest.raises(SystemExit) as error:
        controller.main(args)
    assert error.value.code == 2
    assert not work.exists()


def test_status_is_read_only_and_preserves_the_failed_air_receipt(tmp_path, monkeypatch, capsys):
    for key in ("ROOT", "PLAN", "STATE"):
        monkeypatch.setattr(controller, key, getattr(controller, key))
    plan = {"experiment": "my-run", "profile": "customer-workspace", "gpu_hours_upper_bound": 641}
    state = {"status": "COMPLETE", "stages": {"train": {"run_id": "123", "status": "FAILED"}}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "state.json").write_text(json.dumps(state))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    controller.main(["status", "--work-dir", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE"
    assert result["stages"]["train"]["status"] == "FAILED"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def stage_fixture(tmp_path, monkeypatch, *, submitted):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "STATE", tmp_path / "state.json")
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump({"compute": {"num_accelerators": 16}}))
    plan = {"profile": "frozen-profile", "source": str(tmp_path), "stages": {"train": {
        "file": str(job), "yaml_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "timeout_minutes": 60, "idempotency_key": "stable-key"}}}
    state = {"stages": {"train": {"run_id": "123", "status": "RUNNING"}} if submitted else {}}
    return plan, state


def test_controller_does_not_supply_its_own_budget_acceptance(tmp_path, monkeypatch):
    plan, state = stage_fixture(tmp_path, monkeypatch, submitted=False)
    monkeypatch.delenv("BUDGET_OK", raising=False)

    def blocked_gate(args, **kwargs):
        assert "--budget-gate" in args
        assert "env" not in kwargs or kwargs["env"].get("BUDGET_OK") != "1"
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(controller.subprocess, "run", blocked_gate)
    with pytest.raises(subprocess.CalledProcessError):
        controller.stage_run(plan, state, "train")
    assert state["stages"] == {}


def test_resuming_uses_the_frozen_profile_without_submitting_again(tmp_path, monkeypatch):
    plan, state = stage_fixture(tmp_path, monkeypatch, submitted=True)
    calls = []

    def existing_run(args, **kwargs):
        calls.append(args)
        assert args[:3] == ["air", "get", "run"]
        return json.dumps({"data": {"run_id": "123", "status": "SUCCESS"}})

    class LogWatcher:
        def __init__(self, args, **kwargs):
            calls.append(args)

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(controller, "command", existing_run)
    monkeypatch.setattr(controller.subprocess, "Popen", LogWatcher)
    controller.stage_run(plan, state, "train")
    assert state["stages"]["train"]["status"] == "SUCCESS"
    assert all(args[args.index("--profile") + 1] == "frozen-profile" for args in calls)


@pytest.mark.parametrize("fault,last_stage,message", [
    ("certificate", "train", "certified step-100"),
    ("split", "train-export", "frozen experiment split"),
    ("checkpoint", "eval", "different checkpoint"),
])
def test_orchestration_stops_before_publishing_results_on_invalid_evidence(tmp_path, monkeypatch, fault, last_stage, message):
    for name, filename in (("ROOT", None), ("PLAN", "plan.json"), ("STATE", "state.json")):
        monkeypatch.setattr(controller, name, tmp_path / filename if filename else tmp_path)
    plan = {"profile": "customer-workspace", "git_sha": "frozen-source", "data_dir": "/data",
            "checkpoint_root": "/checkpoint", "checkpoint": "/checkpoint/global_step_100",
            "stages": {name: {"artifact_dir": f"/artifacts/{name}"} for name in controller.JOB_FILES}}
    controller.write_json(controller.PLAN, plan)
    controller.write_json(controller.STATE, {"status": "PREPARED", "stages": {}})
    before, manifest = complete_evaluation()
    manifest["split"]["train_ids"] = [f"train-{i}" for i in range(203)]
    before["summary"] = {}
    after = copy.deepcopy(before)
    after["model_identity"] = {"identity": "wrong-model" if fault == "checkpoint" else "trained-model"}
    artifacts = {"DATA_MANIFEST.json": manifest,
                 "qualification.json": {"status": "PASS", "git_sha": plan["git_sha"], "versions": {}},
                 "baseline.json": before, "final.json": after,
                 "run_result.json": {"certified": fault != "certificate", "raw_rc": 0, "final_rc": 0,
                                     "expected_final_version": 100, "observed_version": "100",
                                     "checkpoint": {"identity": "trained-model"}}}
    selection = {"selection": "all_official_training_questions", "heldout_used_for_selection": False,
                 "selected_training_uids": manifest["split"]["train_ids"].copy()}
    if fault == "split":
        selection["selected_training_uids"][0] = "hard-0"
    payload = {"files": {"signal-data/DATA_MANIFEST.json": selection},
               "training_signal": {"infrastructure_errors": 0, "groups": {"one": [{"question_uid": "train-1"}]}}}
    stages = []
    monkeypatch.setattr(controller, "stage_run", lambda plan, state, name: stages.append(name))
    monkeypatch.setattr(controller, "fetch", lambda remote, name, **kwargs: copy.deepcopy(artifacts[name]))
    monkeypatch.setattr(controller, "export_payload", lambda name: payload)
    monkeypatch.setattr(controller.metrics, "compare", lambda *args: pytest.fail("invalid evidence must not be compared"))
    with pytest.raises(ValueError, match=message):
        controller.run()
    assert stages == list(controller.JOB_FILES)[:list(controller.JOB_FILES).index(last_stage) + 1]
    assert not (tmp_path / "comparison.json").exists()
    assert json.loads(controller.STATE.read_text())["status"] != "COMPLETE"
