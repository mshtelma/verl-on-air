"""engine/stage_model.py: one resolved revision, content-verified files, no mixed snapshots (R17).

Before: every file was fetched from an unpinned revision, and resume trusted a matching byte
size -- a repo update or a same-sized changed shard produced a mixed snapshot."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from support import ENGINE, load_module

COMMIT = "c0ffee" + "0" * 34
SHARD = "model-00001-of-00001.safetensors"
FILES = {
    "config.json": b'{"architectures": ["X"]}',
    "model.safetensors.index.json": json.dumps({"weight_map": {"w": SHARD}}).encode(),
    SHARD: b"\x00" * 64 + b"tensor-bytes",
}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Hub:
    """A fake huggingface_hub: publishes FILES at COMMIT with the Hub's own content hashes."""

    def __init__(self, served: dict[str, bytes] | None = None):
        self.served = served or FILES     # what downloads actually return (can differ: corruption)
        self.downloads: list[tuple[str, str]] = []
        self.info_revisions: list[str] = []

    def module(self) -> types.ModuleType:
        hub = self

        class Api:
            def __init__(self, token=None):
                pass

            def model_info(self, repo_id, revision=None, files_metadata=False):
                hub.info_revisions.append(revision)
                sibs = []
                for name, data in FILES.items():
                    lfs = types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name.endswith(".safetensors") else None
                    sibs.append(types.SimpleNamespace(rfilename=name, size=len(data), lfs=lfs,
                                                      blob_id=None if lfs else git_blob(data)))
                return types.SimpleNamespace(sha=COMMIT, siblings=sibs)

        def hf_hub_download(repo_id, filename, revision, local_dir, token=None):
            hub.downloads.append((filename, revision))
            out = Path(local_dir) / filename
            out.write_bytes(hub.served[filename])
            return str(out)

        m = types.ModuleType("huggingface_hub")
        m.HfApi, m.hf_hub_download = Api, hf_hub_download
        return m


@pytest.fixture
def stage(tmp_path: Path, monkeypatch):
    dest, scratch = tmp_path / "models" / "m", tmp_path / "scratch"

    def go(hub: Hub | None = None, **env: str):
        hub = hub or Hub()
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub.module())
        mod = load_module(ENGINE / "stage_model.py", env_overrides={
            "MODEL_ID": "org/m", "MODEL_DIR": str(dest), "SCRATCH_DIR": str(scratch), **env})
        mod.main()
        return hub

    go.dest = dest
    return go


def test_every_file_is_fetched_at_the_one_resolved_commit(stage):
    hub = stage()
    assert hub.info_revisions == ["main"] and {rev for _, rev in hub.downloads} == {COMMIT}
    staged = json.loads((stage.dest / "STAGED.json").read_text())
    assert staged["revision"] == COMMIT and set(staged["files"]) == set(FILES)
    assert not (stage.dest / ".staging.json").exists() and not (stage.dest / ".staging.lock").exists()


def test_a_correct_file_already_there_is_verified_not_refetched(stage):
    stage.dest.mkdir(parents=True)
    (stage.dest / SHARD).write_bytes(FILES[SHARD])
    hub = stage()
    assert SHARD not in [f for f, _ in hub.downloads]


def test_a_same_sized_but_different_file_is_refetched(stage):
    stage.dest.mkdir(parents=True)
    (stage.dest / SHARD).write_bytes(b"\xff" * len(FILES[SHARD]))   # the old size-only check trusted this
    hub = stage()
    assert SHARD in [f for f, _ in hub.downloads]
    assert (stage.dest / SHARD).read_bytes() == FILES[SHARD]


def test_a_directory_holding_another_revision_is_refused(stage):
    stage.dest.mkdir(parents=True)
    (stage.dest / "STAGED.json").write_text(json.dumps({"model_id": "org/m", "revision": "deadbeef"}))
    with pytest.raises(SystemExit, match="holds org/m@deadbeef"):
        stage()


def test_a_corrupt_download_is_refused(stage):
    bad = dict(FILES)
    bad[SHARD] = b"\x01" * len(FILES[SHARD])
    with pytest.raises(SystemExit, match="does not match its sha256"):
        stage(Hub(served=bad))


def test_a_held_lock_stops_a_second_staging_job(stage):
    stage.dest.mkdir(parents=True)
    (stage.dest / ".staging.lock").write_text("host=other pid=1")
    with pytest.raises(SystemExit, match="another staging job"):
        stage()


def test_an_already_staged_revision_is_a_no_op(stage):
    stage()
    hub = stage()
    assert hub.downloads == []
