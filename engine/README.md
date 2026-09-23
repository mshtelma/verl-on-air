# engine — the shared RL platform (write once, reuse everywhere)

← [verl-on-air](../README.md) · [training-modes](../docs/training-modes.md) · [configuration](../docs/configuration.md) · [build your own use case](../docs/new-usecase.md)

All the genuinely hard infrastructure lives here and is **use-case-agnostic**. A use case
never edits the engine; it plugs in via a few env vars. If you find yourself wanting to
change a file in here to make *your task* work, that is a signal the seam is in the wrong
place — say so, rather than forking a launcher.

```
engine/
├─ train/
│   ├─ dispatch_agentic.sh        THE ENTRYPOINT for agentic jobs. Picks the training
│   │                             mode (TRAIN_MODE), splits nodes between training and
│   │                             judge-serving, resolves the use case's tool/reward
│   │                             paths, and derives PYTHONPATH.
│   ├─ run_grpo_fully_async.sh    fully-async GRPO: disjoint Rollouter/Trainer GPU pools,
│   │                             MessageQueue + NCCL weight sync, bounded staleness.
│   └─ run_grpo_megatron.sh       synchronous GRPO: rollout co-located with training.
│                                 Also holds MEGATRON_MODE (FSDP vs classic) + offload.
├─ serve/
│   ├─ serve_judge.sh             serve an LLM-as-judge as an OpenAI endpoint, single- or
│   │                             multi-node, and publish its URL to a rendezvous file.
│   └─ serve_and_eval.sh          serve any model + run a use case's eval.py against it.
├─ lib/
│   ├─ hparams.sh                 air `parameters:` (a YAML file) -> shell, via hp <key> <default>
│   └─ ray_cluster.sh             multi-node Ray head/worker bring-up + teardown traps
└─ stage_model.py                 HF -> Unity Catalog Volume model staging (resumable)
```

## The plugin seam

The engine reads these env vars, so a use case supplies only **files and values** — never
engine edits. Full list with defaults: [`../docs/configuration.md`](../docs/configuration.md).

| env var | what the use case supplies |
|---|---|
| `CUSTOM_REWARD_PATH` (+ `CUSTOM_REWARD_NAME`) | `reward.py` — verl imports it; its directory goes on `PYTHONPATH` |
| `FUNCTION_TOOL_PATH` | `tool.py` — the agent's `@function_tool` definitions |
| `TOOL_CONFIG_PATH` / `AGENT_LOOP_CONFIG_PATH` | stateful `BaseTool` config / a custom agent loop (optional) |
| `EVAL_SCRIPT` | `eval.py` — for `serve_and_eval.sh` |
| `MULTI_TURN` / `MAX_TURNS` / `TOOL_FORMAT` | the agent-loop shape |
| `REWARD_MANAGER` | rule reward (`naive`) vs judge reward (`rate_limited`, async + concurrent) |
| `TRAIN_MODE` | `async` or `sync` — which launcher runs |
| `TRAINING_NODES` | how many nodes train; the rest serve the judge |
| topology (`TP`/`EP`/`GEN_TP`/`ROLLOUT_NNODES`/…) | fixed per model+GPU count — [`../docs/tuning.md`](../docs/tuning.md) |

Two details that make this work and are easy to get wrong if you re-implement it:

- **air does not expand `${CODE_SOURCE_PATH}` inside `env_variables:`.** The dispatcher
  resolves those paths itself (`_resolve_path`), falling back to the repo root — which is
  also why the launchers work locally under `DRY_RUN=1`.
- **`PYTHONPATH` is derived from the resolved tool/reward directories**, so `reward.py`
  and `tool.py` import each other by bare name (`import reward`, `import tool`) and
  `eval.py` imports the *same* modules. Training and eval therefore share the exact
  scorer — they cannot drift.

## What the dispatcher actually does

AI Runtime runs a job's `command:` **once per node** with the topology injected
(`NUM_NODES`, `POD_RANK`, `MASTER_ADDR`, `MASTER_PORT`). `dispatch_agentic.sh` turns that
into roles:

```
POD_RANK <  TRAINING_NODES   ->  ${TRAIN_LAUNCHER}   (GRPO; rank 0 is the Ray head)
POD_RANK >= TRAINING_NODES   ->  serve_judge.sh      (judge at TP = 8 x judge nodes)
```

The two halves form **separate Ray clusters** (training on 6379, judge on 6380 with its
own Ray pin) and talk only over HTTP. The judge head publishes its endpoint to a Unity
Catalog rendezvous file; training waits for it (`JUDGE_WAIT_TIMEOUT`) and rank 0 writes a
`training_done` sentinel from an `EXIT` trap so the judge shuts itself down on success,
failure *or* signal. `TRAINING_NODES` equal to the node count means "no judge" — which is
what a rule-based use case wants.

Why one job rather than two: df1 has **no cross-job connectivity** and one image per job.

## Things the engine handles so a use case doesn't

- 35B MoE parallelism (`EP`/`TP`/`PP`/`CP`/`ETP`) and the Megatron-FSDP-vs-classic +
  offload decision, with the `CUDA_DEVICE_MAX_CONNECTIONS` trap handled per mode.
- Multi-node Ray bring-up, worker join, and teardown traps.
- The fully-async Rollouter/Trainer split, weight-sync cadence, and the explicit
  `lr_decay_steps` that streaming requires (without it Megatron's scheduler asserts).
- The **completion certificate** ([`lib/run_certificate.py`](lib/run_certificate.py)):
  verl's fully-async exit code is wrong both ways -- a finished run exits non-zero (the
  finishing component cancels the other) and a crashed one can exit 0 (the Rollouter
  swallows its own exception and sends the normal stop signal). So for **every** exit code
  the launcher certifies success only if this run wrote verl's checkpoint tracker at the
  exact planned final version, that checkpoint verifies, and no component raised the
  abort channel ([`lib/run_control.py`](lib/run_control.py), polled by a watchdog that
  stops the run). The verdict lands in `run_result.json` next to the checkpoints.
- Episode-length arithmetic for multi-turn, in both launchers, identically.
- vLLM workarounds: the custom-all-reduce graph-capture crash, KV-cache sizing, the
  prefix-cache-with-weight-sync question.
- Eval serving: NVMe pre-staging (UC FUSE random-read is slow), `/health` waiting,
  `PYTHONPATH` wiring.
- Resumable model staging (a retry skips complete shards by exact byte size).

## Extending the engine

Fair game, in rough order of usefulness:

1. **`MEGATRON_MODE` on the async launcher.** It is hard-wired to classic + CPU offload;
   FSDP there would likely free the trainer node's host RAM.
2. **An offline/off-policy mode** — see [`../docs/training-modes.md`](../docs/training-modes.md) §6.
3. **`REWARD_MANAGER` + `NORM_ADV_BY_STD_IN_GRPO` on the sync launcher**, so a
   judge-reward use case can run in sync mode too. Left out deliberately rather than
   guessed: the reward-manager config key differs between the two trainer paths, and an
   invented Hydra key aborts the run at config parse.
4. **A GRPO advantage hook** (e.g. dropping degenerate or unknown-reward groups from the
   batch). There was one here for a task that is not published; it was removed rather
   than shipped broken.

Training modes and how to switch: [`../docs/training-modes.md`](../docs/training-modes.md).
Every setting: [`../docs/configuration.md`](../docs/configuration.md).
