#!/usr/bin/env python3
"""Typed, cross-field preflight for a training job. Both launchers run it before anything is
computed from a knob and before Ray starts; DRY_RUN runs it too, so a dry run checks meaning,
not just spelling.

    preflight.py knobs --mode async|sync
        Every engine knob the job sets (its environment), against KNOBS: its type and allowed
        values, and whether this mode's launcher reads it at all. A knob only the OTHER launcher
        reads is an error -- here it would be silently ignored -- and so is an unknown name with an
        engine prefix (ROLLOUT_TEMPERATURE for ROLLOUT_TEMP). Prints `export NAME=True|False` for
        every boolean it saw, normalised from true/1/yes/on and false/0/no/off, for the launcher to
        eval: every launcher then compares against exactly True/False.

    preflight.py plan --mode async|sync [--json-out FILE] KEY=VALUE ...
        The launcher's resolved geometry and budget (KEY=VALUE: MODEL, TP, PP, ..., see plan()),
        checked against the model -- its config.json, else KNOWN_MODELS: TP divides every head count
        (attention, KV, linear attention), PP the layers, EP the experts; DP = trainer GPUs /
        (TP*PP*CP) is whole; EP*ETP*PP divides the trainer GPUs (Megatron's expert grid); the rollout
        TP divides the attention heads and the rollout GPUs; the batch splits evenly over DP; and no
        FSDP with CPU offload. Prints `PREFLIGHT_PLAN <json>`: roles, parallelism, budget,
        checkpoints.

Exit 0 when valid; otherwise 1, with every problem found, one per line.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ASYNC, SYNC = ("async",), ("sync",)
BOTH = ASYNC + SYNC
TRUE, FALSE = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}
# verl v0.9.0's registered tool parsers (verl/experimental/agent_loop/tool_parser.py).
TOOL_FORMATS = ("hermes", "gpt-oss", "qwen3_coder", "glm", "seed", "minimax", "kimi", "deepseek_v4", "gemma4")
# ...and reward managers (verl/workers/reward_manager, experimental/reward_loop/reward_manager).
REWARD_MANAGERS = ("naive", "prime", "batch", "dapo", "gdpo", "rate_limited", "remote")
# Names no launcher treats as a knob: the runtime's topology and the environment's plumbing.
PLUMBING = {"DRY_RUN", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "NGPUS_PER_NODE", "NNODES", "NODE_RANK",
            "NUM_NODES", "POD_RANK", "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF"}
# Prefixes only the engine uses: an unknown name with one of these is almost surely a typo.
ENGINE_PREFIXES = ("ROLLOUT_", "TRAINER_", "MEGATRON_", "OFFLOAD", "AGENT_", "TOOL_", "REWARD_",
                   "NORM_ADV_")


@dataclass(frozen=True)
class Knob:
    kind: str                       # int | float | bool | enum | str
    modes: tuple[str, ...]          # the launchers that read it
    lo: float | None = None         # inclusive bounds (int, float)
    hi: float | None = None
    choices: tuple[str, ...] = ()   # enum values
    lo_open: bool = False           # lo itself not allowed


def _int(lo=None, hi=None, modes=BOTH):
    return Knob("int", modes, lo=lo, hi=hi)


def _bool(modes=BOTH):
    return Knob("bool", modes)


KNOBS: dict[str, Knob] = {
    # parallelism + placement
    "TP": _int(1), "PP": _int(1), "CP": _int(1), "EP": _int(1), "ETP": _int(1), "GEN_TP": _int(1),
    "ROLLOUT_NNODES": _int(0), "N_GPUS_ROLLOUT": _int(1, modes=ASYNC),
    "MEGATRON_MODE": Knob("enum", SYNC, choices=("fsdp", "classic")),
    "OFFLOAD": Knob("enum", SYNC, choices=("auto", "0", "1")),
    "OFFLOAD_FRACTION": Knob("float", BOTH, lo=0, hi=1),
    "TRAINER_MODE": Knob("enum", SYNC, choices=("sync", "separate_async")),
    "USE_DIST_CKPT": _bool(), "DIST_CKPT_PATH": Knob("str", BOTH),
    # rollout engine
    "ROLLOUT_GPU_MEM_UTIL": Knob("float", BOTH, lo=0, hi=1, lo_open=True),
    "ROLLOUT_ENFORCE_EAGER": _bool(), "MAX_MODEL_LEN": _int(1),
    "ROLLOUT_TEMP": Knob("float", ASYNC, lo=0, lo_open=True),
    "ROLLOUT_PREFIX_CACHING": _bool(ASYNC), "ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE": _bool(ASYNC),
    # multi-turn agent loop + tools
    "MULTI_TURN": _bool(), "MAX_TURNS": _int(1), "TOOL_FORMAT": Knob("enum", BOTH, choices=TOOL_FORMATS),
    "AGENT_NUM_WORKERS": _int(1), "MAX_TOOL_RESPONSE_LEN": _int(1),
    "FUNCTION_TOOL_PATH": Knob("str", BOTH), "TOOL_CONFIG_PATH": Knob("str", BOTH),
    "AGENT_LOOP_CONFIG_PATH": Knob("str", BOTH),
    # reward
    "REWARD_MANAGER": Knob("enum", BOTH, choices=REWARD_MANAGERS),
    "CUSTOM_REWARD_PATH": Knob("str", BOTH), "CUSTOM_REWARD_NAME": Knob("str", BOTH),
    "REWARD_MAX_CONCURRENT": _int(1), "REWARD_MAX_RPM": _int(1), "REWARD_MAX_TPM": _int(1),
    "REWARD_TIMEOUT": Knob("float", BOTH, lo=0, lo_open=True),
    "REWARD_SOURCE": Knob("str", BOTH),          # the use case's reward reads it, in either mode
    "NORM_ADV_BY_STD_IN_GRPO": _bool(),
    # fully-async budget + the completion certificate
    "TRIGGER_SYNC_STEP": _int(1, modes=ASYNC), "REQUIRE_BATCHES": _int(1, modes=ASYNC),
    "STALENESS": Knob("float", ASYNC, lo=0), "PARTIAL_ROLLOUT": _bool(ASYNC),
    "LR_DECAY_STEPS": _int(1, modes=ASYNC), "ALLOW_UNCERTIFIED": _bool(ASYNC),
    "ABORT_POLL_S": Knob("float", ASYNC, lo=0, lo_open=True), "ABORT_GRACE_S": Knob("float", ASYNC, lo=0),
    "CERT_SETTLE_S": Knob("float", ASYNC, lo=0),
    # sync trainer
    "DATA_SHUFFLE": _bool(SYNC), "VAL_BEFORE_TRAIN": _bool(SYNC), "WEIGHT_BUCKET_MB": _int(1, modes=SYNC),
    "CKPT_ENGINE_BACKEND": Knob("str", SYNC), "PARAM_SYNC_STEP": _int(1, modes=SYNC),
    "MAX_OFF_POLICY": _int(0, modes=SYNC), "ASYNC_WARMUP_BATCHES": _int(0, modes=SYNC),
    # checkpoints + run identity
    "SAVE_FREQ": _int(), "TEST_FREQ": _int(), "MAX_CKPT_TO_KEEP": _int(1),
    "RUN_ID": Knob("str", BOTH), "RESUME": Knob("str", BOTH), "GIT_SHA": Knob("str", BOTH),
    "VOA_IMAGE": Knob("str", BOTH), "PROJECT_NAME": Knob("str", BOTH), "EXPERIMENT_NAME": Knob("str", BOTH),
}

# Architecture limits of the models the jobs name, from their config.json (text_config) on the Hub
# (checked 2026-09-23). Used when the model is not a local directory -- a dry run on a laptop.
KNOWN_MODELS: dict[str, dict[str, int]] = {
    "Qwen3.5-2B": {"heads": 8, "kv_heads": 2, "linear_k_heads": 16, "linear_v_heads": 16, "layers": 24, "experts": 0},
    "Qwen3.5-9B": {"heads": 16, "kv_heads": 4, "linear_k_heads": 16, "linear_v_heads": 32, "layers": 32, "experts": 0},
    "Qwen3.5-35B-A3B": {"heads": 16, "kv_heads": 2, "linear_k_heads": 16, "linear_v_heads": 32, "layers": 40,
                        "experts": 256},
}
_CONFIG_KEYS = {"heads": "num_attention_heads", "kv_heads": "num_key_value_heads",
                "linear_k_heads": "linear_num_key_heads", "linear_v_heads": "linear_num_value_heads",
                "layers": "num_hidden_layers", "experts": "num_experts"}


# --- phase 1: knobs --------------------------------------------------------------------------------
def check_value(name: str, knob: Knob, raw: str) -> tuple[str | None, str | None]:
    """-> (problem, normalised boolean or None)."""
    v = raw.strip()
    if knob.kind == "bool":
        if v.lower() in TRUE:
            return None, "True"
        if v.lower() in FALSE:
            return None, "False"
        return f"{name}={raw!r} is not a boolean (True/False)", None
    if knob.kind == "enum":
        return (None if v in knob.choices else f"{name}={raw!r}: expected one of {', '.join(knob.choices)}"), None
    if knob.kind in ("int", "float"):
        try:
            x = int(v) if knob.kind == "int" else float(v)
        except ValueError:
            return f"{name}={raw!r} is not {'an integer' if knob.kind == 'int' else 'a number'}", None
        if knob.lo is not None and (x < knob.lo or (knob.lo_open and x == knob.lo)):
            return f"{name}={raw!r}: must be {'>' if knob.lo_open else '>='} {knob.lo:g}", None
        if knob.hi is not None and x > knob.hi:
            return f"{name}={raw!r}: must be <= {knob.hi:g}", None
    return None, None


def check_knobs(mode: str, environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    problems, booleans = [], {}
    other = "sync" if mode == "async" else "async"
    for name in sorted(environ):
        raw = environ[name]
        knob = KNOBS.get(name)
        if knob is None:
            if name not in PLUMBING and name.startswith(ENGINE_PREFIXES):
                problems.append(f"{name}: not a knob either launcher reads (a typo?)")
            continue
        if mode not in knob.modes:
            problems.append(f"{name}: only the TRAIN_MODE={other} launcher reads it; this {mode} job would "
                            f"ignore it -- remove it")
            continue
        problem, norm = check_value(name, knob, raw)
        if problem:
            problems.append(problem)
        elif norm is not None:
            booleans[name] = norm
    if mode == "sync" and environ.get("TRAINER_MODE", "sync") == "sync" and environ.get("ROLLOUT_NNODES", "0") != "0":
        problems.append("ROLLOUT_NNODES: a sync job co-locates the rollout -- set 0 "
                        "(or TRAINER_MODE=separate_async)")
    return problems, booleans


# --- phase 2: the resolved plan ----------------------------------------------------------------------
def model_limits(model: str) -> tuple[dict[str, int] | None, str]:
    cfg_path = Path(model) / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        cfg = cfg.get("text_config") or cfg
        return {k: int(cfg.get(v) or 0) for k, v in _CONFIG_KEYS.items()}, str(cfg_path)
    name = model.rstrip("/").split("/")[-1]
    if name in KNOWN_MODELS:
        return dict(KNOWN_MODELS[name]), f"KNOWN_MODELS[{name!r}]"
    return None, "unknown"


def _divides(n: int, of: int) -> bool:
    return n >= 1 and of % n == 0


def plan(mode: str, kv: dict[str, str]) -> tuple[list[str], dict[str, Any]]:
    """kv: the launcher's resolved values. Both modes: MODEL, NUM_NODES (the job's), NODES (the
    training launcher's), GPUS_PER_NODE, TRAINER_NODES, TRAINER_GPUS, ROLLOUT_GPUS,
    TP PP CP EP ETP GEN_TP, PPO_MINI, ROLLOUT_N,
    MAX_TURNS, MULTI_TURN, SAVE_FREQ, CKPT_DIR, RUN_ID. async: TOTAL_ROLLOUT_STEPS,
    TRIGGER_SYNC_STEP, REQUIRE_BATCHES. sync: MEGATRON_MODE, OFFLOAD, TRAIN_BATCH_SIZE,
    TOTAL_TRAINING_STEPS."""
    problems: list[str] = []

    def num(key: str) -> int:
        try:
            return int(kv[key])
        except (KeyError, ValueError):
            problems.append(f"plan: {key}={kv.get(key)!r} is missing or not an integer")
            return 0

    tp, pp, cp, ep, etp, gen_tp = (num(k) for k in ("TP", "PP", "CP", "EP", "ETP", "GEN_TP"))
    trainer_gpus, rollout_gpus, per_node = num("TRAINER_GPUS"), num("ROLLOUT_GPUS"), num("GPUS_PER_NODE")
    ppo_mini, rollout_n = num("PPO_MINI"), num("ROLLOUT_N")
    if problems:
        return problems, {}

    limits, source = model_limits(kv.get("MODEL", ""))
    if limits is None:
        problems.append(f"model {kv.get('MODEL')!r}: no config.json there and not in KNOWN_MODELS -- its "
                        f"head/layer/expert limits cannot be checked (stage it, or add it to KNOWN_MODELS)")
    else:
        for key in ("heads", "kv_heads", "linear_k_heads", "linear_v_heads"):
            if limits[key] and limits[key] % tp:
                problems.append(f"TP={tp} does not divide the model's {limits[key]} {key.replace('_', ' ')}")
        if limits["layers"] % pp:
            problems.append(f"PP={pp} does not divide the model's {limits['layers']} layers")
        if limits["experts"] == 0 and ep > 1:
            problems.append(f"EP={ep} on a dense model (no experts)")
        elif limits["experts"] and limits["experts"] % ep:
            problems.append(f"EP={ep} does not divide the model's {limits['experts']} experts")
        if limits["heads"] % gen_tp:
            problems.append(f"GEN_TP={gen_tp} does not divide the model's {limits['heads']} attention heads (vLLM)")
    model_parallel = tp * pp * cp
    dp = trainer_gpus // model_parallel if _divides(model_parallel, trainer_gpus) else 0
    if not dp:
        problems.append(f"trainer GPUs {trainer_gpus} are not a whole multiple of TP*PP*CP = {tp}*{pp}*{cp} "
                        f"= {model_parallel}: no integral data-parallel size")
    if not _divides(ep * etp * pp, trainer_gpus):
        problems.append(f"EP*ETP*PP = {ep}*{etp}*{pp} does not divide the {trainer_gpus} trainer GPUs "
                        f"(Megatron's expert-parallel grid)")
    if not _divides(gen_tp, rollout_gpus):
        problems.append(f"GEN_TP={gen_tp} does not divide the {rollout_gpus} rollout GPUs: no whole replicas")
    if dp and ppo_mini % dp:
        problems.append(f"ppo_mini_batch_size={ppo_mini} does not split evenly over DP={dp}")
    if dp and (ppo_mini * rollout_n) % dp:
        problems.append(f"ppo_mini_batch_size*rollout_n = {ppo_mini * rollout_n} samples do not split evenly "
                        f"over DP={dp}")

    budget: dict[str, Any]
    if mode == "sync":
        batch, steps = num("TRAIN_BATCH_SIZE"), num("TOTAL_TRAINING_STEPS")
        if kv.get("MEGATRON_MODE") == "fsdp" and kv.get("OFFLOAD") == "1":
            problems.append("MEGATRON_MODE=fsdp with OFFLOAD=1: Megatron-FSDP crashes with CPU offload "
                            "(aten.is_pinned on DTensor) -- use MEGATRON_MODE=classic, or more GPUs and OFFLOAD=0")
        if batch and (batch * rollout_n) % trainer_gpus:
            problems.append(f"train_batch_size*rollout_n = {batch * rollout_n} is not divisible by the "
                            f"{trainer_gpus} trainer GPUs")
        if batch and ppo_mini and batch % ppo_mini:
            problems.append(f"train_batch_size={batch} is not a multiple of ppo_mini_batch_size={ppo_mini}")
        budget = {"optimizer_steps": steps, "prompt_groups": steps * batch,
                  "trajectories": steps * batch * rollout_n,
                  "updates_per_step": batch // ppo_mini if ppo_mini else None}
        final = steps
    else:
        total, trigger, require = num("TOTAL_ROLLOUT_STEPS"), num("TRIGGER_SYNC_STEP"), num("REQUIRE_BATCHES")
        per_sync = trigger * require * ppo_mini
        final = total // per_sync if per_sync and total % per_sync == 0 else None
        budget = {"prompt_groups": total, "trajectories": total * rollout_n,
                  "optimizer_updates": total // (require * ppo_mini) if require * ppo_mini else None,
                  "weight_syncs": final, "prompt_groups_per_sync": per_sync}
    save = int(kv.get("SAVE_FREQ") or 0)
    out = {
        "mode": mode, "run_id": kv.get("RUN_ID"),
        "model": {"path": kv.get("MODEL"), "limits": limits, "limits_from": source},
        "roles": {"job_nodes": int(kv.get("NUM_NODES") or 0), "training_nodes": int(kv.get("NODES") or 0),
                  "judge_nodes": int(kv.get("NUM_NODES") or 0) - int(kv.get("NODES") or 0),
                  "gpus_per_node": per_node,
                  "trainer": {"nodes": int(kv.get("TRAINER_NODES") or 0), "gpus": trainer_gpus},
                  "rollout": {"gpus": rollout_gpus, "replicas": rollout_gpus // gen_tp if gen_tp else None,
                              "co_located": mode == "sync"}},
        "parallelism": {"tp": tp, "pp": pp, "cp": cp, "ep": ep, "etp": etp, "dp": dp, "gen_tp": gen_tp},
        "episode": {"multi_turn": kv.get("MULTI_TURN") == "True", "max_turns": int(kv.get("MAX_TURNS") or 1),
                    "rollout_n": rollout_n},
        "budget": budget,
        "checkpoints": {"dir": kv.get("CKPT_DIR"), "save_freq": save, "final": final,
                        "saved": (final // save if final and save > 0 else None)},
    }
    return problems, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=("knobs", "plan"))
    ap.add_argument("--mode", required=True, choices=("async", "sync"))
    ap.add_argument("--json-out")
    ap.add_argument("values", nargs="*", help="plan: KEY=VALUE resolved by the launcher")
    args = ap.parse_intermixed_args(argv)

    if args.phase == "knobs":
        problems, booleans = check_knobs(args.mode, dict(os.environ))
        for p in problems:
            print(f"FATAL: preflight: {p}", file=sys.stderr)
        if problems:
            return 1
        for name, value in booleans.items():
            print(f"export {name}={shlex.quote(value)}")
        return 0

    kv = dict(v.split("=", 1) for v in args.values if "=" in v)
    problems, out = plan(args.mode, kv)
    for p in problems:
        print(f"FATAL: preflight: {p}", file=sys.stderr)
    if problems:
        return 1
    print("PREFLIGHT_PLAN " + json.dumps(out, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(args.json_out).with_name(f".{Path(args.json_out).name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
        os.replace(tmp, args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
