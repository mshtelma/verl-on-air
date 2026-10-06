import pytest

from select_train import mixed_training_uids


def artifact(train_values, heldout_values=(0, 1, 0, 1)):
    return {"valid": True, "eval_policy": {"samples_per_question": 4}, "results": [
        {"uid": uid, "split": split, "status": "scored", "reward": value}
        for uid, split, values in [("train", "train_probe", train_values), ("heldout", "heldout", heldout_values)]
        for value in values
    ]}


def test_selects_only_mixed_training_questions():
    assert mixed_training_uids(artifact((0, 1, 0, 0)), {"train"}) == ["train"]


def test_heldout_success_cannot_rescue_zero_training_signal():
    with pytest.raises(ValueError, match="no mixed training"):
        mixed_training_uids(artifact((0, 0, 0, 0)), {"train"})


def test_wrong_split_or_incomplete_groups_are_rejected():
    with pytest.raises(ValueError, match="outside the training pilot"):
        mixed_training_uids(artifact((0, 1, 0, 0)), {"different"})
    with pytest.raises(ValueError, match="incomplete"):
        mixed_training_uids(artifact((0, 1)), {"train"})
