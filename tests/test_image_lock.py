"""W2.7/R21-R22: an image tag names one build, from known inputs, pushed as one digest.

The reviewer's case: re-pushing changed content under a tag AI Runtime has already registered keeps
serving the old digest, and `make stale-check` (mtime of the repo's scripts vs the image's creation
time) neither caught that nor looked at the files the image is actually built from. The check is by
content now: the build labels the image with the sha256 of its inputs, docker/IMAGE.lock records the
digest and inputs each tag was pushed with, and push/register refuse anything else.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from support import REPO, load_module

il = load_module(REPO / "scripts" / "image_lock.py")
IMAGE = "example/verl-megatron-air:v99"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


@pytest.fixture
def lock_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the build inputs, with image_lock pointed at it and docker faked."""
    for f in il.INPUT_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, tmp_path / f)
    monkeypatch.setattr(il, "REPO", tmp_path)
    monkeypatch.setattr(il, "LOCK", tmp_path / "docker" / "IMAGE.lock")
    return tmp_path


class FakeDocker:
    def __init__(self, labels=None, repo_digests=(), remote=DIGEST_A):
        self.labels, self.repo_digests, self.remote = labels, list(repo_digests), remote

    def __call__(self, *args: str) -> str:
        if args[:2] == ("image", "inspect"):
            if self.labels is None:
                raise il.LockError("No such image")
            fmt = args[-1]
            if "Labels" in fmt:
                return json.dumps(self.labels)
            if "RepoDigests" in fmt:
                return json.dumps(self.repo_digests)
            if "Created" in fmt:
                return "2026-09-23T00:00:00Z"
            if "Size" in fmt:
                return "123"
        if args[:3] == ("buildx", "imagetools", "inspect"):
            return f"Name:      {args[3]}\nMediaType: application/vnd.docker.distribution.manifest.v2+json\nDigest:    {self.remote}\n"
        raise AssertionError(f"unexpected docker call {args}")


def test_inputs_hash_covers_every_input_file_and_content_arg(lock_repo: Path):
    base = il.inputs_sha256({})
    assert base == il.inputs_sha256({}), "not deterministic"
    for f in il.INPUT_FILES:
        p = lock_repo / f
        orig = p.read_bytes()
        p.write_bytes(orig + b"\n# changed\n")
        assert il.inputs_sha256({}) != base, f"{f} is not covered"
        p.write_bytes(orig)
    for a in il.CONTENT_ARGS:
        assert il.inputs_sha256({a: "1"}) != base, f"{a} is not covered"
    # the index is not an input: the lock pins what gets installed whichever index serves it
    assert il.inputs_sha256({"PIP_INDEX_URL": "https://x/simple"}) == base


def test_certs_are_inputs(lock_repo: Path):
    base = il.inputs_sha256({})
    (lock_repo / "certs").mkdir()
    (lock_repo / "certs" / ".gitkeep").write_text("")
    assert il.inputs_sha256({}) == base, ".gitkeep must not change the hash"
    (lock_repo / "certs" / "corp-ca.crt").write_text("-----BEGIN CERTIFICATE-----\n")
    assert il.inputs_sha256({}) != base


def test_check_local_accepts_an_image_built_from_the_current_inputs(lock_repo, monkeypatch):
    cur = il.inputs_sha256({})
    monkeypatch.setattr(il, "docker", FakeDocker(labels={il.LABEL: cur}))
    assert il.check_local("v99", IMAGE, cur) == []


@pytest.mark.parametrize("labels", [None, {}, {il.LABEL: "0" * 64}],
                         ids=["no-image", "unlabelled", "other-inputs"])
def test_check_local_refuses_an_image_not_built_from_the_current_inputs(lock_repo, monkeypatch, labels):
    monkeypatch.setattr(il, "docker", FakeDocker(labels=labels))
    problems = il.check_local("v99", IMAGE, il.inputs_sha256({}))
    assert problems and re.search(r"build", problems[0])


def test_check_local_refuses_reusing_a_pushed_tag_for_new_inputs(lock_repo, monkeypatch):
    cur = il.inputs_sha256({})
    il.save({"schema": "voa.image_lock/v1",
             "images": {"v99": {"digest": DIGEST_A, "inputs_sha256": "f" * 64}}})
    monkeypatch.setattr(il, "docker", FakeDocker(labels={il.LABEL: cur}))
    problems = il.check_local("v99", IMAGE, cur)
    assert len(problems) == 1 and "make bump" in problems[0]


def test_record_writes_the_pushed_digest_and_refuses_a_second_one(lock_repo, monkeypatch):
    cur = il.inputs_sha256({})
    monkeypatch.setattr(il, "docker", FakeDocker(
        labels={il.LABEL: cur},
        repo_digests=[f"other/repo@{DIGEST_B}", f"example/verl-megatron-air@{DIGEST_A}"]))
    e = il.record("v99", IMAGE, cur)
    assert e["digest"] == DIGEST_A and e["inputs_sha256"] == cur
    assert json.loads(il.LOCK.read_text())["images"]["v99"]["digest"] == DIGEST_A
    assert il.record("v99", IMAGE, cur)["digest"] == DIGEST_A, "re-recording the same push is fine"

    monkeypatch.setattr(il, "docker", FakeDocker(
        labels={il.LABEL: cur}, repo_digests=[f"example/verl-megatron-air@{DIGEST_B}"]))
    with pytest.raises(il.LockError, match="must not be reused"):
        il.record("v99", IMAGE, cur)
    assert json.loads(il.LOCK.read_text())["images"]["v99"]["digest"] == DIGEST_A


def test_record_refuses_an_image_that_was_never_pushed(lock_repo, monkeypatch):
    monkeypatch.setattr(il, "docker", FakeDocker(labels={}, repo_digests=[]))
    with pytest.raises(il.LockError, match="pushed"):
        il.record("v99", IMAGE, il.inputs_sha256({}))
    assert not il.LOCK.exists()


def test_check_remote_requires_the_recorded_digest(lock_repo, monkeypatch):
    monkeypatch.setattr(il, "docker", FakeDocker(remote=DIGEST_A))
    with pytest.raises(il.LockError, match="no digest recorded"):
        il.check_remote("v99", IMAGE)
    il.save({"schema": "voa.image_lock/v1", "images": {"v99": {"digest": DIGEST_A}}})
    assert il.check_remote("v99", IMAGE) == DIGEST_A
    monkeypatch.setattr(il, "docker", FakeDocker(remote=DIGEST_B))
    with pytest.raises(il.LockError, match="re-pushed"):
        il.check_remote("v99", IMAGE)


def test_constraints_keep_exact_pins_only():
    freeze = "\n".join([
        "# comment", "aiohttp==3.13.2", "torch==2.11.0",
        "megatron-core @ file:///opt/vendor/src/megatron-core",
        "nvidia-cuda-nvcc==13.0.88", "opencv-python-headless==4.12.0.88", ""])
    assert il.constraints(freeze) == ["aiohttp==3.13.2", "torch==2.11.0"]
    with pytest.raises(il.LockError, match="exact pin"):
        il.constraints("aiohttp>=3")


def test_the_committed_lock_files_agree_with_each_other():
    """requirements.lock never pins what artifacts.lock provides (the wheel wins, or the build's
    lock-drift check fails), and IMAGE.lock names only the repo config.env points at."""
    art = [ln.split() for ln in (REPO / "docker/artifacts.lock").read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    assert {r[0] for r in art} <= {"wheel", "git"}
    for r in art:
        pin = r[4]
        assert re.fullmatch(r"[0-9a-f]{64}" if r[0] == "wheel" else r"[0-9a-f]{40}", pin), r
    norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()  # noqa: E731
    provided = {norm(r[1]) for r in art}
    pinned = {norm(ln.split("==")[0]) for ln in (REPO / "docker/requirements.lock").read_text().splitlines()
              if ln.strip() and not ln.startswith("#")}
    assert not provided & pinned, provided & pinned

    doc = json.loads((REPO / "docker/IMAGE.lock").read_text())
    assert doc["schema"] == "voa.image_lock/v1"
    assert doc["base"]["digest"] == il.base_digest(), "the Dockerfile's FROM digest is not the one recorded"
    for tag, e in doc["images"].items():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", e["digest"]), tag
        assert e["ref"].endswith(":" + tag), tag


def test_dockerfile_uses_no_nested_quotes_inside_parameter_expansions():
    """`${X:+--flag "${X}"}` inside a RUN broke the Dockerfile frontend's heredoc detection (the v9
    build failed to parse with "unknown instruction: import"); keep such expansions quote-free."""
    text = (REPO / "docker/Dockerfile").read_text()
    assert not re.search(r"\$\{[A-Za-z_]+:[+-][^}]*[\"']", text)
