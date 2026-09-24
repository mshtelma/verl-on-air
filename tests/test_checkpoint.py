"""engine/lib/verify_checkpoint.py -- what counts as a complete, servable checkpoint (R05, R02, R15).

The interrupted-save case is not hypothetical: a real run on the Volume has
global_step_50/actor/{extra,model,optimizer} with no ckpt_contents.json and no weights, while its
latest_checkpointed_iteration.txt says 40. A directory's existence proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import ENGINE, fake_hf_model, fake_train_checkpoint, load_module, run, safetensors_bytes

vc = load_module(ENGINE / "lib" / "verify_checkpoint.py")
CLI = str(ENGINE / "lib" / "verify_checkpoint.py")


def test_hf_model_dir_verifies_with_identity(tmp_path: Path):
    hf = fake_hf_model(tmp_path / "m")
    ident = vc.verify(hf)
    assert ident["kind"] == "hf_model" and ident["step"] is None
    assert len(ident["shards"]) == 2
    assert ident["total_bytes"] == sum(p.stat().st_size for p in hf.glob("*.safetensors"))
    assert len(ident["identity"]) == 16


@pytest.mark.parametrize("suffix", ["", "actor", "actor/model", "actor/model/huggingface"])
def test_every_path_into_a_training_checkpoint_checks_its_manifest(tmp_path: Path, suffix: str):
    step = fake_train_checkpoint(tmp_path / "run", 20)
    ident = vc.verify(step / suffix if suffix else step)
    assert ident["kind"] == "train_checkpoint" and ident["step"] == 20
    assert ident["run_dir"] == str(tmp_path / "run")
    assert ident["hf_dir"].endswith("global_step_20/actor/model/huggingface")


@pytest.mark.parametrize("suffix", ["", "actor/model/huggingface"])
def test_interrupted_save_without_manifest_is_rejected(tmp_path: Path, suffix: str):
    # the real global_step_50 shape: actor/{extra,model,optimizer} + nothing else
    step = fake_train_checkpoint(tmp_path / "run", 50, manifest=False, hf=False)
    with pytest.raises(vc.CheckpointError, match="ckpt_contents.json"):
        vc.verify(step / suffix if suffix else step)


def test_manifest_with_an_empty_hf_dir_is_rejected(tmp_path: Path):
    # verl's path helper mkdirs model/huggingface as a side effect -- it can exist and be empty
    step = fake_train_checkpoint(tmp_path / "run", 10, hf=False)
    with pytest.raises(vc.CheckpointError, match="config.json missing"):
        vc.verify(step)


def test_missing_empty_and_truncated_shards_are_rejected(tmp_path: Path):
    hf = fake_hf_model(tmp_path / "a")
    next(hf.glob("*-00002-*")).unlink()
    with pytest.raises(vc.CheckpointError, match="missing"):
        vc.verify(hf)

    hf = fake_hf_model(tmp_path / "b")
    next(hf.glob("*-00001-*")).write_bytes(b"")
    with pytest.raises(vc.CheckpointError, match="empty"):
        vc.verify(hf)

    hf = fake_hf_model(tmp_path / "c")
    shard = next(hf.glob("*-00001-*"))
    shard.write_bytes(shard.read_bytes()[:-10])  # a partially copied shard
    with pytest.raises(vc.CheckpointError, match="truncated"):
        vc.verify(hf)

    hf = fake_hf_model(tmp_path / "d")
    next(hf.glob("*-00001-*")).write_bytes(b"x" * 64)  # not a safetensors file at all
    with pytest.raises(vc.CheckpointError, match="header"):
        vc.verify(hf)

    hf = fake_hf_model(tmp_path / "e")  # the index names a tensor its shard does not hold
    shard = next(hf.glob("*-00002-*"))
    shard.write_bytes(safetensors_bytes({"something.else": b"x" * 64}))
    with pytest.raises(vc.CheckpointError, match="lacks 1 tensor"):
        vc.verify(hf)


def test_an_index_that_declares_more_than_it_holds_is_accepted(tmp_path: Path):
    """Acceptance run A6: every mbridge export of Qwen3.5-35B-A3B (incl. the served pure-EM step 20)
    declares total_size 71,903,655,008 and holds 70,214,492,304 bytes, all tensors present."""
    hf = fake_hf_model(tmp_path / "m")
    idx = json.loads((hf / "model.safetensors.index.json").read_text())
    assert idx["metadata"]["total_size"] > sum(p.stat().st_size for p in hf.glob("*.safetensors"))
    assert vc.verify(hf)["total_bytes"] == sum(p.stat().st_size for p in hf.glob("*.safetensors"))


def test_tokenizer_and_config_are_required(tmp_path: Path):
    hf = fake_hf_model(tmp_path / "a")
    (hf / "tokenizer.json").unlink()
    with pytest.raises(vc.CheckpointError, match="tokenizer"):
        vc.verify(hf)
    hf = fake_hf_model(tmp_path / "b")
    (hf / "config.json").write_text("{not json")
    with pytest.raises(vc.CheckpointError, match="not valid JSON"):
        vc.verify(hf)


def test_manifest_step_mismatch_and_non_hf_format_are_rejected(tmp_path: Path):
    step = fake_train_checkpoint(tmp_path / "run", 20)
    m = step / "actor" / "ckpt_contents.json"
    data = json.loads(m.read_text())
    m.write_text(json.dumps({**data, "global_step": 30}))
    with pytest.raises(vc.CheckpointError, match="global_step=30"):
        vc.verify(step)
    data["contents"]["model"]["format"] = "megatron_dist_checkpoint"
    m.write_text(json.dumps(data))
    with pytest.raises(vc.CheckpointError, match="no HuggingFace export"):
        vc.verify(step)


def test_require_train_checkpoint_rejects_a_plain_model(tmp_path: Path):
    with pytest.raises(vc.CheckpointError, match="not a training checkpoint"):
        vc.verify(fake_hf_model(tmp_path / "m"), require_train_checkpoint=True)


def test_identity_is_stable_and_changes_with_the_weights(tmp_path: Path):
    a = fake_hf_model(tmp_path / "a")
    b = fake_hf_model(tmp_path / "b")
    assert vc.verify(a)["identity"] == vc.verify(b)["identity"]
    next(b.glob("*-00001-*")).write_bytes(safetensors_bytes({"layer.0.weight": b"y" * 65}))
    assert vc.verify(a)["identity"] != vc.verify(b)["identity"]


def test_list_complete_skips_incomplete_steps(tmp_path: Path):
    root = tmp_path / "ckpt"
    fake_train_checkpoint(root, 10)
    fake_train_checkpoint(root, 20, manifest=False)
    fake_train_checkpoint(root / "run-b", 5)  # one run dir below the root
    assert [s for s, _ in vc.list_complete(root)] == [5, 10]


def test_cli_exit_codes(tmp_path: Path):
    good = fake_train_checkpoint(tmp_path / "run", 20)
    r = run(["python3", CLI, str(good), "--print-hf-dir", "--json-out", str(tmp_path / "id.json")])
    assert r.returncode == 0 and r.stdout.strip().endswith("actor/model/huggingface"), r.stdout
    assert json.loads((tmp_path / "id.json").read_text())["step"] == 20
    bad = fake_train_checkpoint(tmp_path / "run", 30, manifest=False)
    r = run(["python3", CLI, str(bad)])
    assert r.returncode == 1 and "FAIL" in r.stdout


def test_same_metadata_different_weights_is_a_different_model(tmp_path: Path):
    # two checkpoints of one architecture: identical config, index and shard sizes
    a, b = fake_hf_model(tmp_path / "a"), fake_hf_model(tmp_path / "b")
    next(b.glob("*-00001-*")).write_bytes(safetensors_bytes({"layer.0.weight": b"y" * 64}))  # same size, other bytes
    assert vc.verify(a)["identity"] != vc.verify(b)["identity"]


def test_a_copy_matches_only_if_its_content_matches(tmp_path: Path):
    import shutil
    src = fake_hf_model(tmp_path / "src", shard_bytes=3 << 20)
    ident = vc.verify(src)
    good = Path(shutil.copytree(src, tmp_path / "copy"))
    vc.same_model(good, ident)
    shard = next(good.glob("*-00002-*"))
    data = bytearray(shard.read_bytes())
    data[-10] ^= 0xFF                                     # corrupt the tail
    shard.write_bytes(bytes(data))
    with pytest.raises(vc.CheckpointError, match="shard_samples_sha256 differs"):
        vc.same_model(good, ident)


def test_staged_revision_is_part_of_the_identity(tmp_path: Path):
    a, b = fake_hf_model(tmp_path / "a"), fake_hf_model(tmp_path / "b")
    (a / "STAGED.json").write_text(json.dumps({"model_id": "org/m", "revision": "1111"}))
    (b / "STAGED.json").write_text(json.dumps({"model_id": "org/m", "revision": "2222"}))
    ia, ib = vc.verify(a), vc.verify(b)
    assert ia["hub_revision"] == "1111" and ia["identity"] != ib["identity"]
