import copy
import hashlib
import json
from pathlib import Path

import pytest

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
    result = controller.submission_receipt("[INFO] Using Profile: [df1]\n" + json.dumps(receipt))
    assert str(result["run_id"]) == "252353129579893"


def completion_review(tmp_path, monkeypatch, *, abort=None, trajectories=1600, worker_rc=0):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    plan = {"git_sha": "frozen-source", "experiment": "officeqa-test", "total_updates": 100,
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
    review = {"profile": "df1", "run_id": "123", "source_git_sha": plan["git_sha"], "evidence": evidence,
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
