#!/usr/bin/env python3
"""Compose every training job's REAL emitted overrides against the PINNED verl config. CPU only.

`air run --dry-run` validates a job's YAML schema; it cannot tell you what verl will actually
receive. This does, without a GPU:

  1. run the job's own `command:` with DRY_RUN=1 and the env AI Runtime would inject (the
     launchers print their resolved override list and exit),
  2. compose those overrides with Hydra against the verl source the image installs -- the
     `git verl` line of docker/artifacts.lock, checked out at exactly that commit,
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
_OVERRIDE_RE = re.compile(r"^    ([+A-Za-z][^\n]*?) \\$")
RUN_ID = "compose-check"   # what make would set per submission


def verl_pin() -> tuple[str, str]:
    """(version, commit) of verl as docker/artifacts.lock pins it for the image."""
    for line in (REPO / "docker" / "artifacts.lock").read_text().splitlines():
        f = line.split()
        if len(f) == 5 and f[0] == "git" and f[1] == "verl":
            return f[2], f[4]
    sys.exit("compose_check: no `git verl ...` line in docker/artifacts.lock")


def ensure_verl_src(explicit: str | None = None) -> Path:
    """A verl checkout at exactly the pinned commit (cloned into .cache/ on first use)."""
    ref, want = verl_pin()
    src = Path(explicit or os.environ.get("VERL_SRC") or REPO / ".cache" / f"verl-{ref}")
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", ref, VERL_URL, str(src)],
                       check=True)
    got = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if got != want:
        sys.exit(f"compose_check: {src} is at {got}, but docker/artifacts.lock pins verl {ref} = {want}")
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
        "RENDEZVOUS_ROOT": str(workdir / "rdv"), "PYTHONDONTWRITEBYTECODE": "1", "RUN_ID": RUN_ID,
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


def check_certifiable(job: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """A fully-async run's success is decided by its final checkpoint (engine/lib/run_certificate.py),
    so the RESOLVED config must produce one at an exact, predictable version."""
    if job["mode"] != "async":
        return []
    bad = []
    per_sync = (get(cfg, "actor_rollout_ref.actor.ppo_mini_batch_size", 0)
                * get(cfg, "async_training.require_batches", 0)
                * get(cfg, "async_training.trigger_parameter_sync_step", 0))
    total = get(cfg, "rollout.total_rollout_steps")
    if not per_sync or not isinstance(total, int) or total % per_sync:
        bad.append(f"rollout.total_rollout_steps={total} is not a multiple of samples/sync={per_sync}")
    if not isinstance(get(cfg, "trainer.save_freq"), int) or get(cfg, "trainer.save_freq") <= 0:
        bad.append(f"trainer.save_freq={get(cfg, 'trainer.save_freq')}: no checkpoint -> cannot certify")
    if get(cfg, "trainer.test_freq") == 0:
        bad.append("trainer.test_freq=0 divides by zero at the end of fit() and skips the final save")
    if get(cfg, "actor_rollout_ref.actor.checkpoint.async_save"):
        bad.append("actor checkpoint async_save=True: the tracker is written before the save finishes")
    return bad


def check_role_spans(job: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """A multi-turn job must run the role-span agent loop (the reward's record of what the MODEL
    wrote, engine/lib/role_spans.py), and Ray's workers must be able to import it by name."""
    if not get(cfg, "actor_rollout_ref.rollout.multi_turn.enable"):
        return []
    bad = []
    reg = get(cfg, "actor_rollout_ref.rollout.agent.agent_loop_config_path")
    try:
        entries = yaml.safe_load(Path(reg).read_text()) if reg else None
    except OSError:
        entries = None
    targets = {e.get("name"): e.get("_target_") for e in (entries or []) if isinstance(e, dict)}
    if targets.get("tool_agent") != "role_span_agent_loop.RoleSpanToolAgentLoop":
        bad.append(f"agent_loop_config_path={reg!r} does not register tool_agent as the role-span loop "
                   f"(got {targets.get('tool_agent')!r})")
    pp = str(get(cfg, "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH") or "").split(":")
    if str(REPO / "engine" / "train") not in pp:
        bad.append("Ray workers' PYTHONPATH (ray_kwargs.ray_init.runtime_env) lacks engine/train: "
                   "the agent-loop registry's _target_ would not import")
    return bad


def check_run_identity(job: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Each run gets its own <output_dir>/<RUN_ID>/, and resuming is an explicit choice: verl's
    default resume_mode=auto would silently continue whatever checkpoint sits in output_dir."""
    bad = []
    run = job["spec"].get("env_variables", {}).get("RUN_ID") or RUN_ID
    if not str(get(cfg, "trainer.default_local_dir", "")).endswith(f"/{run}"):
        bad.append(f"trainer.default_local_dir={get(cfg, 'trainer.default_local_dir')!r} is not <output_dir>/<RUN_ID>")
    if get(cfg, "trainer.resume_mode") != "disable":
        bad.append(f"trainer.resume_mode={get(cfg, 'trainer.resume_mode')!r}: a fresh run must not resume implicitly")
    return bad


CHECKS: list[Callable[[dict[str, Any], dict[str, Any]], list[str]]] = [
    check_topology, check_custom_reward, check_certifiable, check_role_spans, check_run_identity]

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
          f"{verl_pin()[0]} ({verl_pin()[1][:9]}) with all invariants  -> {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
