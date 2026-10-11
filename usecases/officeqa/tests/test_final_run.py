import copy
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
