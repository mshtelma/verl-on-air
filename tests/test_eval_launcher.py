"""engine/serve/serve_and_eval.sh: preflight BEFORE staging or vLLM (REVIEW.md R01, R05).

Runs the real launcher with the env each shipped eval YAML produces -- env_variables values arrive
LITERALLY, `${CODE_SOURCE_PATH}` and all -- with vllm and the /health probe stubbed. Before the fix
the launcher started vLLM and only then died on `python: can't open file '.../${CODE_SOURCE_PATH}/...'`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from support import ENGINE, REPO, StubBin, fake_hf_model, fake_train_checkpoint, run

LAUNCHER = str(ENGINE / "serve" / "serve_and_eval.sh")
EVAL_YAMLS = sorted(REPO.glob("usecases/*/air/[35]_*eval.yaml"))


def _stubs(stub_bin: StubBin) -> StubBin:
    stub_bin.add("vllm", "exec sleep 30")
    # python3: pass /health, "run" the eval by recording what it was given, else the real python
    stub_bin.add("python3", "\n".join([
        'if [[ "${1:-}" == "-c" && "${2:-}" == *"/health"* ]]; then exit 0; fi',
        'if [[ "${1:-}" == */eval.py ]]; then echo "EVAL-RAN $1 MODEL_PATH=$MODEL_PATH '
        'IDENT=$EVAL_MODEL_IDENTITY_FILE"; exit 0; fi',
        f'exec {sys.executable} "$@"']))
    return stub_bin


def _job_env(stub_bin: StubBin, yaml_path: Path, tmp: Path, **extra: str) -> dict[str, str]:
    spec = yaml.safe_load(yaml_path.read_text())
    e = {k: str(v) for k, v in (spec.get("env_variables") or {}).items()}
    e.update(CODE_SOURCE_PATH=str(REPO), EVAL_LOCAL_CACHE=str(tmp / "nvme"), EVAL_HEALTH_TIMEOUT="5")
    e.update(extra)
    return stub_bin.env(**e)


@pytest.mark.parametrize("job", EVAL_YAMLS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_eval_job_runs_its_own_eval_script(job: Path, stub_bin: StubBin, tmp_path: Path):
    """The literal ${CODE_SOURCE_PATH}/.../eval.py from the YAML is resolved and executed."""
    model = (fake_train_checkpoint(tmp_path / "run", 20) if job.name.startswith("5_")
             else fake_hf_model(tmp_path / "base"))
    r = run(["bash", LAUNCHER], env=_job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(model)))
    usecase = job.parent.parent.name
    assert r.returncode == 0, r.stdout[-3000:]
    assert f"EVAL-RAN {REPO}/usecases/{usecase}/eval.py" in r.stdout
    hf = model / "actor/model/huggingface" if job.name.startswith("5_") else model
    assert f"MODEL_PATH={hf}" in r.stdout  # the eval's tokenizer follows the served model
    assert stub_bin.calls("vllm"), "vLLM was never started"


def test_missing_eval_script_fails_before_vllm(stub_bin: StubBin, tmp_path: Path):
    job = REPO / "usecases/agentic-search/air/3_baseline_eval.yaml"
    env = _job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(fake_hf_model(tmp_path / "m")),
                   EVAL_SCRIPT="${CODE_SOURCE_PATH}/usecases/no-such-usecase/eval.py")
    r = run(["bash", LAUNCHER], env=env)
    assert r.returncode == 2 and "EVAL_SCRIPT does not exist" in r.stdout
    assert f"{REPO}/usecases/no-such-usecase/eval.py" in r.stdout  # resolved in the message
    assert not stub_bin.calls("vllm")
    assert not (tmp_path / "nvme").exists(), "staged weights before failing"


def test_checkpoint_eval_has_no_default_and_lists_complete_steps(stub_bin: StubBin, tmp_path: Path):
    root = tmp_path / "ckpt"
    fake_train_checkpoint(root, 12)
    fake_train_checkpoint(root, 24, manifest=False)  # interrupted save: must not be offered
    job = REPO / "usecases/math/air/5_eval.yaml"
    env = _job_env(_stubs(stub_bin), job, tmp_path, EVAL_CKPT_ROOT=str(root))
    env.pop("EVAL_MODEL_PATH", None)
    r = run(["bash", LAUNCHER], env=env)
    assert r.returncode == 2 and "set EVAL_MODEL_PATH" in r.stdout
    assert f"12\t{root}/global_step_12" in r.stdout
    assert "global_step_24" not in r.stdout.split("complete checkpoints under")[-1]
    assert not stub_bin.calls("vllm")


@pytest.mark.parametrize("damage", ["no_manifest", "missing_shard"])
def test_incomplete_checkpoint_fails_before_staging(stub_bin: StubBin, tmp_path: Path, damage: str):
    step = fake_train_checkpoint(tmp_path / "run", 24, manifest=(damage != "no_manifest"))
    if damage == "missing_shard":
        next((step / "actor/model/huggingface").glob("*-00001-*")).unlink()
    job = REPO / "usecases/math/air/5_eval.yaml"
    r = run(["bash", LAUNCHER], env=_job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(step)))
    assert r.returncode == 2 and "not a complete, servable model" in r.stdout
    assert not stub_bin.calls("vllm")
    assert not (tmp_path / "nvme").exists()


def test_tokenizer_path_must_match_the_served_model(stub_bin: StubBin, tmp_path: Path):
    job = REPO / "usecases/agentic-search/air/3_baseline_eval.yaml"
    env = _job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(fake_hf_model(tmp_path / "a")),
                   MODEL_PATH=str(fake_hf_model(tmp_path / "b")))
    r = run(["bash", LAUNCHER], env=env)
    assert r.returncode == 2 and "is not the served model" in r.stdout
    assert not stub_bin.calls("vllm")


def test_an_existing_eval_artifact_stops_the_job_before_staging(stub_bin: StubBin, tmp_path: Path):
    out = tmp_path / "musique_base.json"
    out.write_text("{}")
    job = REPO / "usecases/agentic-search/air/3_baseline_eval.yaml"
    env = _job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(fake_hf_model(tmp_path / "m")),
                   EVAL_OUT=str(out))
    r = run(["bash", LAUNCHER], env=env)
    assert r.returncode == 2 and "already exists" in r.stdout
    assert not stub_bin.calls("vllm")
    env["EVAL_OVERWRITE"] = "1"
    assert run(["bash", LAUNCHER], env=env).returncode == 0


# --- the local NVMe cache (R15) ----------------------------------------------------------------
def _serve_cached(stub_bin: StubBin, tmp_path: Path, model: Path):
    job = REPO / "usecases/agentic-search/air/3_baseline_eval.yaml"
    env = _job_env(_stubs(stub_bin), job, tmp_path, EVAL_MODEL_PATH=str(model), EVAL_STAGE="1")
    return run(["bash", LAUNCHER], env=env)


def test_a_reused_cache_never_serves_the_previous_models_weights(stub_bin: StubBin, tmp_path: Path):
    # reviewer reproduction: model A, then model B through the same cache -> B was served A's weights
    a, b = fake_hf_model(tmp_path / "A"), fake_hf_model(tmp_path / "B")
    next(b.glob("*-00001-*")).write_bytes(b"B" * 64)
    assert _serve_cached(stub_bin, tmp_path, a).returncode == 0
    r = _serve_cached(stub_bin, tmp_path, b)
    assert r.returncode == 0, r.stdout[-2000:]
    served = [c.split()[2] for c in stub_bin.calls("vllm")]
    assert len(set(served)) == 2, served
    assert next(Path(served[-1]).glob("*-00001-*")).read_bytes() == b"B" * 64


def test_the_same_model_reuses_its_verified_copy(stub_bin: StubBin, tmp_path: Path):
    a = fake_hf_model(tmp_path / "A")
    assert _serve_cached(stub_bin, tmp_path, a).returncode == 0
    r = _serve_cached(stub_bin, tmp_path, a)
    assert r.returncode == 0 and "reusing the verified local copy" in r.stdout


def test_a_failed_copy_fails_the_job(stub_bin: StubBin, tmp_path: Path):
    a = fake_hf_model(tmp_path / "A")
    stub_bin.add("cp", "exit 1")              # every copy fails (was `cp -rn ... || true`)
    r = _serve_cached(stub_bin, tmp_path, a)
    assert r.returncode != 0 and not stub_bin.calls("vllm")
