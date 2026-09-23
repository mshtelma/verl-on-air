"""W2.7/R21: vendored build artefacts are accepted only at their pinned identity.

`make vendor` used to download "whatever the URL serves" and `git clone --depth 1 -b <branch>`, and
the Dockerfile COPYed the result unchecked. Now docker/artifacts.lock pins each wheel's sha256 and
each source's commit, and scripts/vendor_artifacts.sh enforces both (the Dockerfile re-checks).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from support import REPO, run


def _tree(tmp: Path, lock_lines: list[str]) -> Path:
    root = tmp / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "docker").mkdir()
    shutil.copy(REPO / "scripts/vendor_artifacts.sh", root / "scripts")
    (root / "docker/artifacts.lock").write_text("# kind name version source pin\n" + "\n".join(lock_lines) + "\n")
    return root


def _git_repo(tmp: Path) -> tuple[Path, str]:
    src = tmp / "upstream"
    src.mkdir()
    g = ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    (src / "setup.py").write_text("# fake\n")
    subprocess.run([*g, "add", "."], check=True)
    subprocess.run([*g, "commit", "-qm", "c1"], check=True)
    commit = subprocess.run([*g, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (src / "later.txt").write_text("after the pin\n")
    subprocess.run([*g, "add", "."], check=True)
    subprocess.run([*g, "commit", "-qm", "c2"], check=True)
    return src, commit


def test_a_wheel_and_a_source_at_their_pins_are_vendored(tmp_path: Path):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    whl.write_bytes(b"wheel bytes")
    src, commit = _git_repo(tmp_path)
    root = _tree(tmp_path, [
        f"wheel pkg 1.0 file://{whl} {hashlib.sha256(whl.read_bytes()).hexdigest()}",
        f"git   lib c1  {src} {commit}"])
    r = run(["bash", "scripts/vendor_artifacts.sh"], cwd=root)
    assert r.returncode == 0, r.stdout
    assert (root / "vendor/wheels" / whl.name).read_bytes() == b"wheel bytes"
    tree = root / "vendor/src/lib"
    assert (tree / ".voa-commit").read_text().strip() == commit
    assert (tree / "setup.py").is_file() and not (tree / "later.txt").exists(), "not the pinned commit"
    assert not (tree / ".git").exists()

    r2 = run(["bash", "scripts/vendor_artifacts.sh"], cwd=root)
    assert r2.returncode == 0 and "have  pkg" in r2.stdout and "have  lib" in r2.stdout, r2.stdout


def test_a_wheel_whose_sha256_differs_is_deleted_and_fails(tmp_path: Path):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    whl.write_bytes(b"tampered")
    root = _tree(tmp_path, [f"wheel pkg 1.0 file://{whl} {'0' * 64}"])
    r = run(["bash", "scripts/vendor_artifacts.sh"], cwd=root)
    assert r.returncode != 0 and "sha256" in r.stdout, r.stdout
    assert list((root / "vendor/wheels").iterdir()) == [], "a mismatching download was kept"


def test_a_stale_vendored_wheel_is_replaced_not_trusted(tmp_path: Path):
    whl = tmp_path / "pkg-1.0-py3-none-any.whl"
    whl.write_bytes(b"good")
    root = _tree(tmp_path, [f"wheel pkg 1.0 file://{whl} {hashlib.sha256(b'good').hexdigest()}"])
    (root / "vendor/wheels").mkdir(parents=True)
    (root / "vendor/wheels" / whl.name).write_bytes(b"left over from an older pin")
    r = run(["bash", "scripts/vendor_artifacts.sh"], cwd=root)
    assert r.returncode == 0, r.stdout
    assert (root / "vendor/wheels" / whl.name).read_bytes() == b"good"


def test_a_source_whose_commit_cannot_be_checked_out_fails(tmp_path: Path):
    src, _ = _git_repo(tmp_path)
    root = _tree(tmp_path, [f"git lib x {src} {'1' * 40}"])
    r = run(["bash", "scripts/vendor_artifacts.sh"], cwd=root)
    assert r.returncode != 0, r.stdout
    assert not (root / "vendor/src/lib/.voa-commit").exists()
