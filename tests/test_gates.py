"""R10: every advertised local gate FAILS when the thing it gates fails.

Each case is a reviewer reproduction (REVIEW.md R10) turned into a regression test: before the
fix, all of these exited 0.
"""
from __future__ import annotations

from pathlib import Path

from support import REPO, StubBin, run


def make(target: str, *args: str, env: dict[str, str] | None = None, repo: Path = REPO):
    return run(["make", "-C", str(repo), "--no-print-directory", target, "PIP_INDEX_URL=", *args], env=env)


# --- lint ----------------------------------------------------------------------------------
def test_lint_fails_when_shellcheck_finds_problems(stub_bin: StubBin):
    sc = stub_bin.add("shellcheck", 'echo "SC2086 injected"; exit 1')
    r = make("lint", f"SHELLCHECK={sc}")
    assert r.returncode != 0, r.stdout
    assert "shellcheck ok" not in r.stdout


def test_lint_fails_when_shellcheck_is_missing_unless_opted_out():
    r = make("lint", "SHELLCHECK=")
    assert r.returncode != 0 and "shellcheck not installed" in r.stdout
    r = make("lint", "SHELLCHECK=", "ALLOW_NO_SHELLCHECK=1")
    assert r.returncode == 0, r.stdout
    assert "NOT CHECKED" in r.stdout


def test_lint_fails_on_a_syntax_error_even_when_later_files_compile(scratch_repo: Path):
    # sorts FIRST, so the old loop's exit status (the last file's) would have been 0
    (scratch_repo / "engine" / "aaa_broken.py").write_text("def f(:\n    pass\n")
    r = make("lint", "ALLOW_NO_SHELLCHECK=1", repo=scratch_repo)
    assert r.returncode != 0, r.stdout
    assert "aaa_broken.py" in r.stdout


def test_lint_covers_every_shell_script():
    listed = run(["git", "ls-files", "*.sh"]).stdout.split()
    out = make("lint").stdout
    assert f"shellcheck ok  {len(listed)} scripts" in out, out


# --- volume ----------------------------------------------------------------------------------
def test_volume_fails_on_permission_denied(stub_bin: StubBin):
    stub_bin.add("databricks", 'echo "Error: PERMISSION_DENIED: User does not have CREATE VOLUME" >&2; exit 1')
    r = make("volume", env=stub_bin.env())
    assert r.returncode != 0, r.stdout
    assert "already exists" not in r.stdout and "PERMISSION_DENIED" in r.stdout


def test_volume_accepts_already_exists(stub_bin: StubBin):
    stub_bin.add("databricks", "echo \"Error: Volume 'main.x.verl' already exists\" >&2; exit 1")
    r = make("volume", env=stub_bin.env())
    assert r.returncode == 0 and "already exists" in r.stdout, r.stdout


def test_volume_reports_creation(stub_bin: StubBin):
    stub_bin.add("databricks", 'echo "{}"; exit 0')
    r = make("volume", env=stub_bin.env())
    assert r.returncode == 0 and "created volume" in r.stdout, r.stdout


# --- size ------------------------------------------------------------------------------------
def test_size_fails_when_the_image_cannot_be_inspected(stub_bin: StubBin):
    stub_bin.add("docker", 'echo "Error: No such image: x" >&2; exit 1')
    r = make("size", env=stub_bin.env())
    assert r.returncode != 0 and "OK" not in r.stdout, r.stdout


def test_size_fails_over_the_gate_and_passes_under_it(stub_bin: StubBin):
    stub_bin.add("docker", "echo 25000000000")
    assert make("size", env=stub_bin.env()).returncode != 0
    stub_bin.add("docker", "echo 5000000000")
    r = make("size", env=stub_bin.env())
    assert r.returncode == 0 and "under the gate" in r.stdout, r.stdout


def test_size_fails_on_garbage_output(stub_bin: StubBin):
    stub_bin.add("docker", "echo '<no value>'")
    assert make("size", env=stub_bin.env()).returncode != 0


# --- release ordering --------------------------------------------------------------------------
def test_image_and_release_run_their_steps_in_order():
    for target, first in (("image", "docker build"), ("release", "docker build")):
        out = make(target, "-n", "-j4").stdout
        idx = [out.find(s) for s in (first, "image_size.py", "docker push", "register image")]
        assert all(i >= 0 for i in idx) and idx == sorted(idx), (target, idx, out)


# --- doctor ------------------------------------------------------------------------------------
def test_doctor_counts_an_egress_failure_as_a_hard_failure(scratch_repo: Path, stub_bin: StubBin):
    scripts = scratch_repo / "scripts"
    (scripts / "check_dockerhub_push.sh").write_text('#!/bin/bash\necho "mock push scope OK"\n')
    (scripts / "detect_pypi_index.sh").write_text('#!/bin/bash\nprintf "https://example.invalid/simple\\tmock\\n"\n')
    stub_bin.add("docker", "\n".join([
        'case "$1" in',
        "  version) echo 99.0.0; exit 0;;",
        "  buildx) echo 'github.com/docker/buildx v99'; exit 0;;",
        f"  info) echo '{scratch_repo}'; exit 0;;",
        "  run) echo 'BAD https://example.invalid/simple :: injected DNS failure'; exit 0;;",
        "esac", "exit 1"]))
    stub_bin.add("df", 'echo "Filesystem 1024-blocks Used Available Capacity Mounted"; '
                       'echo "mock 900000000 1 899000000 1% /"')
    stub_bin.add("databricks", 'echo "{}"')
    stub_bin.add("air")
    r = run(["bash", str(scripts / "doctor.sh")], cwd=scratch_repo, env=stub_bin.env(PIP_INDEX_URL=""))
    assert "FAIL" in r.stdout and "unreachable from container" in r.stdout, r.stdout
    assert r.returncode != 0, r.stdout
    assert "no hard failures" not in r.stdout
