# engine

The shared, task-independent part of the repo. A use case never edits it; it plugs in through
environment variables. If you find you need to change a file here to make your task work, the
interface is probably in the wrong place, and that is worth fixing rather than forking a launcher.

```
engine/
├─ train/
│   ├─ dispatch_agentic.sh        entry point for agentic jobs: picks the mode (TRAIN_MODE), splits
│   │                             nodes between training and judge serving, resolves the tool and
│   │                             reward paths, sets PYTHONPATH, runs PRE_TRAIN_CHECK
│   ├─ run_grpo_fully_async.sh    fully-async GRPO on separate rollout and trainer GPUs
│   ├─ run_grpo_megatron.sh       synchronous GRPO, rollout co-located with training
│   ├─ role_span_agent_loop.py    verl's ToolAgentLoop plus a record of which characters the model wrote
│   └─ agent_loops.yaml           registers it as `tool_agent`
├─ serve/
│   ├─ serve_judge.sh             serves an LLM judge on one or more nodes and publishes its URL
│   ├─ judge_ping.py              checks the judge answers before training starts
│   ├─ serve_and_eval.sh          checks and serves a model, then runs a use case's eval.py against it
│   └─ eval_contract.py           readiness checks, per-question status, validity, artifact identity
├─ lib/
│   ├─ hparams.sh                 reads air `parameters:` in shell (hp <key> <default>)
│   ├─ paths.sh                   resolves ${CODE_SOURCE_PATH} inside env values
│   ├─ preflight.py               typed knob checks and the job plan
│   ├─ run_identity.sh            RUN_ID, the run's output directory, the RESUME choice
│   ├─ run_manifest.py            run_manifest.json, written at the start of every run
│   ├─ run_driver.sh              runs verl in its own process group with the abort watchdog
│   ├─ run_certificate.py         decides whether a training run really finished
│   ├─ run_control.py             the abort channel (ABORT.json)
│   ├─ ray_cluster.sh             multi-node Ray start-up and teardown
│   ├─ rendezvous.sh              run-scoped rendezvous files on the Volume
│   ├─ verify_checkpoint.py       checks that a model or checkpoint is complete and servable
│   ├─ data_manifest.py           pinned dataset revisions and DATA_MANIFEST.json
│   └─ role_spans.py              who wrote which characters of an episode
├─ testing/                       fault injectors used by the acceptance tests
└─ stage_model.py                 stages a Hugging Face model to the Volume (resumable)
```

## The interface

| env var | what the use case supplies |
|---|---|
| `CUSTOM_REWARD_PATH` (and `CUSTOM_REWARD_NAME`) | `reward.py` |
| `FUNCTION_TOOL_PATH` | `tool.py` with `@function_tool` definitions |
| `TOOL_CONFIG_PATH` / `AGENT_LOOP_CONFIG_PATH` | optional stateful tools or a custom agent loop |
| `EVAL_SCRIPT` | `eval.py`, run by `serve_and_eval.sh` |
| `MULTI_TURN` / `MAX_TURNS` / `TOOL_FORMAT` | the agent loop |
| `REWARD_MANAGER` | `naive` for a rule, `rate_limited` for a judge |
| `TRAIN_MODE` | `async` or `sync` |
| `TRAINING_NODES` / `JUDGE_NODES` | how many nodes train and how many serve the judge |
| `TP`, `EP`, `GEN_TP`, `ROLLOUT_NNODES`, ... | set by the model and GPU count ([docs/tuning.md](../docs/tuning.md)) |

The full list is in [docs/configuration.md](../docs/configuration.md). Three details matter if you
re-implement any of this:

- air does not expand `${CODE_SOURCE_PATH}` inside `env_variables:`, so the engine resolves those
  paths itself (`resolve_code_path` in `lib/paths.sh`). It falls back to the repo root, which is
  why the launchers also work locally with `DRY_RUN=1`.
- `PYTHONPATH` is built from the tool and reward directories, so `reward.py`, `tool.py` and
  `eval.py` import each other by bare name, and training and eval share one scorer.
- The reward knows who wrote what. The `tool_agent` loop turns verl's response mask into character
  spans over the decoded episode, so a reward can read the model's own answer and credit only real
  tool output.

## The dispatcher

AI Runtime runs a job's `command:` once per node and injects the topology (`NUM_NODES`,
`POD_RANK`, `MASTER_ADDR`, `MASTER_PORT`). `dispatch_agentic.sh` assigns the roles:

```
POD_RANK <  TRAINING_NODES   the training launcher (rank 0 is the Ray head)
POD_RANK >= TRAINING_NODES   serve_judge.sh (TP = 8 × JUDGE_NODES)
```

Training and judge are separate Ray clusters (ports 6379 and 6380) and talk only over HTTP. The
judge publishes its endpoint to a rendezvous file on the Volume; training waits for it, and rank 0
writes a `training_done` sentinel on exit so the judge shuts down. With `TRAINING_NODES` equal to
the node count there is no judge. Trainer and judge share one job because AI Runtime jobs cannot
reach each other.

## What the engine takes care of

- MoE parallelism, and the choice between Megatron-FSDP and classic Megatron with CPU offload.
- Multi-node Ray start-up, worker join and teardown.
- The fully-async rollout/trainer split, the weight-sync cadence, and the explicit
  `lr_decay_steps` that streaming needs.
- Deciding whether a run succeeded. verl's fully-async exit code is unreliable in both
  directions, so the launcher checks the checkpoints instead: a run succeeds only if it wrote the
  planned final checkpoint, that checkpoint verifies, and nothing raised an abort. The verdict
  goes to `run_result.json`.
- The multi-turn episode length, computed the same way in both launchers.
- vLLM settings: the custom all-reduce workaround, KV-cache sizing, prefix caching across weight
  syncs.
- Eval serving: copying the model to local NVMe, waiting for `/health`, setting `PYTHONPATH`.
- Model staging at one resolved Hub commit, with every file checked against the Hub's hash.

## Extending it

In rough order of usefulness:

1. `MEGATRON_MODE` on the async launcher, which is hard-wired to classic Megatron with CPU
   offload. FSDP there would likely free the trainer node's host RAM.
2. An offline, off-policy mode ([docs/training-modes.md](../docs/training-modes.md)).
3. A co-located judge in sync mode. The reward wiring reaches sync mode, but the judge topology
   has never run there, so the dispatcher refuses it.
4. A GRPO advantage hook, for example to drop degenerate groups from a batch.
