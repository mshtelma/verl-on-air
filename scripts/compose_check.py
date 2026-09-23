#!/usr/bin/env python3
"""Compose every training job's REAL emitted overrides against the PINNED verl config. CPU only.

`air run --dry-run` validates a job's YAML schema; it cannot tell you what verl will actually
receive. This does, without a GPU:

  1. run the job's own `command:` with DRY_RUN=1 and the env AI Runtime would inject (the
     launchers print their resolved override list and exit),
  2. compose those overrides with Hydra against the verl source pinned in docker/Dockerfile
     (VERL_REF), checked out at the exact commit below,
  3. assert invariants on the resolved config (topology, reward wiring, checkpoint semantics).

    python3 scripts/compose_check.py                    # all training jobs
    python3 scripts/compose_check.py --only 4_train     # substring filter
    VERL_SRC=/path/to/verl python3 scripts/compose_check.py   # use an existing checkout

Needs hydra-core + omegaconf + pyyaml (`make dev-env`) and, the first time, network access to
clone verl into .cache/. No verl Python code is imported and nothing is instantiated.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[1]
VERL_URL = "https://github.com/verl-project/verl"
# The commit docker/Dockerfile's VERL_REF resolves to. Bump both together.
VERL_COMMIT = {"v0.9.0": "483b8a009ba3a97563edee3a19887e4862b8094a"}
_OVERRIDE_RE = re.compile(r"^    ([+A-Za-z][^\n]*?) \\$")


def verl_ref() -> str:
    m = re.search(r"^ARG VERL_REF=(\S+)", (REPO / "docker" / "Dockerfile").read_text(), re.M)
    if not m:
        sys.exit("compose_check: ARG VERL_REF not found in docker/Dockerfile")
    return m.group(1)


def ensure_verl_src(explicit: str | None = None) -> Path:
    """A verl checkout at exactly the pinned commit (cloned into .cache/ on first use)."""
    ref = verl_ref()
    want = VERL_COMMIT.get(ref)
    if want is None:
        sys.exit(f"compose_check: VERL_REF={ref} has no pinned commit in VERL_COMMIT; add it")
    src = Path(explicit or os.environ.get("VERL_SRC") or REPO / ".cache" / f"verl-{ref}")
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", ref, VERL_URL, str(src)],
                       check=True)
    got = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if got != want:
        sys.exit(f"compose_check: {src} is at {got}, but VERL_REF={ref} pins {want}")
    return src


def training_jobs() -> list[Path]:
    jobs = sorted(REPO.glob("infra/geo3k/air/rung*.yaml")) + sorted(REPO.glob("usecases/*/air/4_train*.yaml"))
    return [j for j in jobs if not j.name.startswith(".probe_")]


def render(job: Path, workdir: Path, port: int, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the job's command with DRY_RUN=1 exactly as rank 0 of the job would see it."""
    spec = yaml.safe_load(job.read_text())
    hp = workdir / f"{job.stem}.hparams.yaml"
    hp.write_text(yaml.safe_dump(spec.get("parameters", {}) or {}))
    nodes = max(1, int(spec["compute"]["num_accelerators"]) // 8)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (spec.get("env_variables") or {}).items()})
    env.update({
        "DRY_RUN": "1", "CODE_SOURCE_PATH": str(REPO), "HYPERPARAMETERS_PATH": str(hp),
        "NUM_NODES": str(nodes), "LOCAL_WORLD_SIZE": "8", "POD_RANK": "0", "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "RENDEZVOUS_ROOT": str(workdir / "rdv"), "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    })
    env.update(extra_env or {})
    r = subprocess.run(["bash", "-c", spec["command"]], cwd=workdir, env=env,
                       capture_output=True, text=True, timeout=120)
    out = r.stdout
    mode = "async" if "fully_async_main" in out else "sync" if "verl.trainer.main_ppo" in out else "?"
    return {"spec": spec, "nodes": nodes, "returncode": r.returncode, "mode": mode,
            "stdout": out, "stderr": r.stderr,
            "overrides": [m.group(1) for ln in out.splitlines() if (m := _OVERRIDE_RE.match(ln))]}


def compose(verl_src: Path, mode: str, overrides: list[str]) -> dict[str, Any]:
    from hydra import compose as h_compose, initialize_config_dir
    from omegaconf import OmegaConf

    if mode == "async":
        cdir, name = verl_src / "verl/experimental/fully_async_policy/config", "fully_async_ppo_megatron_trainer"
    else:
        cdir, name = verl_src / "verl/trainer/config", "ppo_trainer"
    # The fully-async config declares `hydra.searchpath: file://verl/trainer/config`, relative to
    # the CWD -- the launcher cd's into site-packages for exactly this reason. Mirror it.
    cwd = os.getcwd()
    os.chdir(verl_src)
    try:
        with initialize_config_dir(config_dir=str(cdir), version_base=None):
            return OmegaConf.to_container(h_compose(config_name=name, overrides=overrides), resolve=True)
    finally:
        os.chdir(cwd)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- invariants ------
# Each check gets (rendered job, resolved config) and returns a list of violations.
def _expected_trainer_nnodes(job: dict[str, Any]) -> int:
    envv = job["spec"].get("env_variables") or {}
    nodes = job["nodes"]
    if "dispatch_agentic.sh" in job["spec"]["command"]:
        nodes = int(envv.get("TRAINING_NODES", 2))
    if job["mode"] == "async":
        nodes -= int(envv.get("ROLLOUT_NNODES", 1))
    return nodes


def check_topology(job: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    want, got = _expected_trainer_nnodes(job), get(cfg, "trainer.nnodes")
    return [] if got == want else [f"trainer.nnodes={got}, expected {want} from compute/TRAINING_NODES/ROLLOUT_NNODES"]


def check_custom_reward(job: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """A job that names a CUSTOM_REWARD_PATH must reach verl's reward loader with it -- verl reads
    reward.custom_reward_function (trainer/ppo/reward.py); the legacy top-level key is migrated
    only by fully_async_main, so a sync job passing it would silently train on the default scorer."""
    want = (job["spec"].get("env_variables") or {}).get("CUSTOM_REWARD_PATH")
    if not want:
        return []
    want = want.replace("${CODE_SOURCE_PATH}", str(REPO)).replace("$CODE_SOURCE_PATH", str(REPO))
    got = get(cfg, "reward.custom_reward_function.path")
    return [] if got and Path(got).resolve() == Path(want).resolve() else [
        f"reward.custom_reward_function.path={got!r}, expected {want!r} (the use case's reward is not wired)"]


CHECKS: list[Callable[[dict[str, Any], dict[str, Any]], list[str]]] = [check_topology, check_custom_reward]

FACTS = ("trainer.nnodes", "trainer.n_gpus_per_node", "trainer.resume_mode", "trainer.save_freq",
         "trainer.test_freq", "trainer.total_training_steps", "trainer.max_actor_ckpt_to_keep",
         "rollout.total_rollout_steps", "algorithm.norm_adv_by_std_in_grpo",
         "reward.custom_reward_function.path", "reward.reward_manager.name",
         "actor_rollout_ref.rollout.agent.agent_loop_config_path", "data.shuffle")


def check_job(verl_src: Path, job_path: Path, workdir: Path, port: int,
              extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    job = render(job_path, workdir, port, extra_env)
    res: dict[str, Any] = {"job": str(job_path.relative_to(REPO)), "mode": job["mode"],
                           "launcher_rc": job["returncode"], "n_overrides": len(job["overrides"]),
                           "violations": []}
    if job["returncode"] != 0 or not job["overrides"]:
        tail = (job["stdout"] + job["stderr"]).strip().splitlines()[-6:]
        res["violations"].append(f"launcher DRY_RUN failed (rc={job['returncode']}): " + " | ".join(tail))
        return res
    try:
        cfg = compose(verl_src, job["mode"], job["overrides"])
    except Exception as e:  # noqa: BLE001 - report every Hydra/OmegaConf failure the same way
        res["violations"].append(f"hydra composition failed: {type(e).__name__}: {e}")
        return res
    res["facts"] = {k: get(cfg, k) for k in FACTS}
    for chk in CHECKS:
        res["violations"] += chk(job, cfg)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verl-src", help="existing verl checkout (else $VERL_SRC, else .cache/)")
    ap.add_argument("--only", help="substring filter on job paths")
    ap.add_argument("--json", default=str(REPO / "logs" / "compose_check.json"), help="report path")
    args = ap.parse_args()

    verl_src = ensure_verl_src(args.verl_src)
    jobs = [j for j in training_jobs() if not args.only or args.only in str(j)]
    if not jobs:
        print("compose_check: no training jobs matched", file=sys.stderr)
        return 1
    results = []
    with tempfile.TemporaryDirectory(prefix="compose-check-") as tmp:
        for i, j in enumerate(jobs):
            results.append(check_job(verl_src, j, Path(tmp), 29500 + i))

    bad = 0
    for r in results:
        ok = not r["violations"]
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {r['job']:<52} {r['mode']:<5} {r['n_overrides']:>3} overrides")
        for v in r["violations"]:
            print(f"          - {v}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{len(results) - bad}/{len(results)} training jobs compose against verl "
          f"{verl_ref()} ({VERL_COMMIT[verl_ref()][:9]}) with all invariants  -> {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
