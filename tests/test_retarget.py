"""R04: pointing the job files at YOUR image, from config.env.

Reviewer reproduction: set DOCKERHUB_USER/IMAGE_NAME to your own, run `make bump` -> the old
script searched for the NEW user/name with the old tag, printed "0 YAML file(s) updated", exited
0, and every job still named the author's private image.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from support import REPO, load_module, run

rt = load_module(REPO / "scripts" / "retarget.py")
AUTHOR = "michaelshtelma587/verl-megatron-air"


def _customize(repo: Path, user: str = "reviewexample", name: str = "custom-rl") -> None:
    cfg = repo / "config.env"
    s = cfg.read_text()
    s = re.sub(r"^DOCKERHUB_USER=.*$", f"DOCKERHUB_USER={user}", s, flags=re.M)
    s = re.sub(r"^IMAGE_NAME=.*$", f"IMAGE_NAME={name}", s, flags=re.M)
    cfg.write_text(s)


def _images(repo: Path) -> dict[str, str | None]:
    out = {}
    for f in rt.job_files(repo):
        env = (yaml.safe_load(f.read_text()) or {}).get("environment") or {}
        out[str(f.relative_to(repo))] = (env.get("docker_image") or {}).get("url") if "docker_image" in env else None
    return out


def test_bump_after_customizing_the_account_retargets_every_custom_image_job(scratch_repo: Path):
    before = {f: f.read_bytes() for f in rt.job_files(scratch_repo)}
    _customize(scratch_repo)
    r = run(["make", "-C", str(scratch_repo), "--no-print-directory", "bump", "PIP_INDEX_URL="])
    assert r.returncode == 0, r.stdout
    imgs = _images(scratch_repo)
    custom = {f: u for f, u in imgs.items() if u is not None}
    assert len(custom) == 23 and set(custom.values()) == {"reviewexample/custom-rl:v9"}, custom
    assert "IMAGE_TAG=v9" in (scratch_repo / "config.env").read_text()
    # stock-environment jobs are byte-identical
    for f, u in imgs.items():
        if u is None:
            assert (scratch_repo / f).read_bytes() == before[scratch_repo / f]


def test_only_the_url_line_changes(scratch_repo: Path):
    f = scratch_repo / "usecases/agentic-search/air/4_train.yaml"
    old = f.read_text().splitlines()
    _customize(scratch_repo)
    assert rt.main(["--root", str(scratch_repo)]) == 0
    new = f.read_text().splitlines()
    diff = [(a, b) for a, b in zip(old, new) if a != b]
    assert len(old) == len(new) and len(diff) == 1
    assert diff[0] == (f"    url: {AUTHOR}:v8", "    url: reviewexample/custom-rl:v8")


def test_check_mode_reports_drift_without_writing(scratch_repo: Path):
    _customize(scratch_repo)
    snapshot = {f: f.read_bytes() for f in rt.job_files(scratch_repo)}
    assert rt.main(["--root", str(scratch_repo), "--check"]) == 1
    assert snapshot == {f: f.read_bytes() for f in rt.job_files(scratch_repo)}
    assert rt.main(["--root", str(scratch_repo)]) == 0
    assert rt.main(["--root", str(scratch_repo), "--check"]) == 0


def test_expect_change_fails_when_nothing_moved(scratch_repo: Path):
    assert rt.main(["--root", str(scratch_repo), "--expect-change"]) == 1


def test_a_failed_bump_restores_config_env(scratch_repo: Path):
    # every job already names the bumped image -> nothing to rewrite -> the bump must not stick
    for f in rt.job_files(scratch_repo):
        f.write_text(f.read_text().replace(f"{AUTHOR}:v8", f"{AUTHOR}:v9"))
    r = run(["bash", str(scratch_repo / "scripts/bump_image_tag.sh")], cwd=scratch_repo)
    assert r.returncode != 0 and "config.env restored" in r.stdout
    assert "IMAGE_TAG=v8" in (scratch_repo / "config.env").read_text()


def test_lint_fails_while_a_job_names_a_different_image(scratch_repo: Path):
    f = scratch_repo / "usecases/math/air/4_train.yaml"
    f.write_text(f.read_text().replace(f"{AUTHOR}:v8", "someone/else:v1"))
    r = run(["make", "-C", str(scratch_repo), "--no-print-directory", "lint", "PIP_INDEX_URL=",
             "ALLOW_NO_SHELLCHECK=1"])
    assert r.returncode != 0 and "MISMATCH usecases/math/air/4_train.yaml" in r.stdout


def test_the_shipped_tree_is_consistent():
    assert rt.main(["--check"]) == 0
