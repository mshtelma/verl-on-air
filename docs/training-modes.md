# Training modes

An RL job has to fit two workloads onto its GPUs: generation (the rollout) and optimisation
(the trainer). How you arrange them does not depend on the task, data or reward. The engine
ships two arrangements as separate launchers and picks one with `TRAIN_MODE`.

```
SYNCHRONOUS (co-located)                  FULLY-ASYNC (disaggregated)
┌──────────────────────────┐              ┌───────────────┐   ┌───────────────┐
│  all GPUs                │              │ Rollouter     │   │ Trainer       │
│  generate, score, update │              │ GPUs 0..7     │<->│ GPUs 8..15    │
│  in lockstep             │              │ generates     │   │ trains        │
└──────────────────────────┘              └───────────────┘   └───────────────┘
 on-policy; GPUs idle in turns             NCCL weight sync every N updates;
                                           overlap, with bounded staleness
```

| mode | launcher | `TRAIN_MODE` | layout | used by |
|---|---|---|---|---|
| synchronous | `engine/train/run_grpo_megatron.sh` | `sync` | rollout and training share the GPUs, on-policy per batch | the geo3k ladder |
| fully-async | `engine/train/run_grpo_fully_async.sh` | `async` (default) | disjoint GPU pools, a MessageQueue, NCCL weight sync, bounded staleness | both use cases |
| separate_async (verl v1) | `run_grpo_megatron.sh` with `TRAINER_MODE=separate_async` | | disjoint pools via verl's v1 trainer | not recommended |
| offline | not implemented | | train on a pre-collected, pre-scored buffer | |

## Switching

The modes need different node counts, backends and step budgets, so each has its own job file
with the same tool, reward and data:

```bash
make search-train        # usecases/agentic-search/air/4_train.yaml, fully-async, 16xH100
make search-train-sync   # usecases/agentic-search/air/4_train_sync.yaml, synchronous, 32xH100
```

The sync search recipe is experimental. It composes against the pinned verl, but on GPU it runs
out of memory in the first actor update as configured (12-turn episodes next to a co-located
vLLM); shorter `MAX_TURNS`, `PP=2` or more nodes are the next things to try.

| | `4_train.yaml` (async) | `4_train_sync.yaml` (sync) |
|---|---|---|
| nodes | 2: one Rollouter node, one Trainer node (`ROLLOUT_NNODES=1`) | 4, all training (`TRAINING_NODES=4`, `ROLLOUT_NNODES=0`) |
| backend | classic Megatron (ZeRO-1), CPU offload, `TP=2 EP=8` | the same layout, with vLLM on the same GPUs |
| budget | `total_rollout_steps: 3200` prompt groups, 100 weight syncs | `total_training_steps: 100` × `train_batch_size: 32` = 3200 prompt groups |
| data order | shuffled | shuffled (`DATA_SHUFFLE: 'True'`), not the same order |
| policy lag | bounded by `STALENESS` | none; a failed group is dropped, not retried |

`engine/train/dispatch_agentic.sh` runs the launcher for `TRAIN_MODE` and, on every rank before
any role starts, refuses: nodes left over after `TRAINING_NODES` unless `JUDGE_NODES` states
exactly that many; judge nodes with no judge configured, or a judge with no node to serve it;
sync mode with a co-located judge (never run); sync mode without an explicit
`parameters.total_training_steps` (the sync launcher's default is a 3-step smoke cap); and sync
mode with `ROLLOUT_NNODES` > 0. Without a cluster, `make config MODE=fsdp GPUS=16` prints the
sync launcher's resolved verl overrides, `make diff-modes` shows what differs between fsdp and
classic, and `make compose-check` composes every training job against the pinned verl.

## Synchronous

One loop on one GPU pool: generate a batch, score it, update, repeat. The shipped jobs set
`ppo_mini_batch_size = train_batch_size`, so every update is on-policy. It is the simplest mode
(one process tree, one Ray cluster, no staleness), but generation and training never overlap and
both must fit in memory at once. vLLM's weights and KV cache sit next to the optimizer state, and
the actor-to-vLLM weight sync adds a transient peak on top. That transient is why the co-located
35B needs 32 GPUs, not 16 ([sizing.md](sizing.md)). The geo3k rungs call `run_grpo_megatron.sh`
directly and are the reference for this mode. Sync-only knobs: `MEGATRON_MODE` (`fsdp` = ZeRO-3,
`classic` = ZeRO-1), `OFFLOAD`, `train_batch_size`, `total_training_steps`, `VAL_BEFORE_TRAIN`,
`WEIGHT_BUCKET_MB`.

## Fully-async

A Rollouter and a Trainer run as separate processes on disjoint GPUs, joined by a MessageQueue,
and the Trainer pushes new weights to the Rollouter over NCCL every few updates. Utilisation is
higher; the price is staleness, since the Rollouter may be a few parameter versions behind. With
no shared GPUs there is no memory contention: the async trainer always uses classic Megatron
(ZeRO-1) with CPU offload on its own node, which is how the 35B trains on 8 trainer plus 8
rollout GPUs. The launcher does not expose `MEGATRON_MODE`. Async-only knobs: `ROLLOUT_NNODES`,
`N_GPUS_ROLLOUT`, `STALENESS`, `TRIGGER_SYNC_STEP`, `REQUIRE_BATCHES`, `PARTIAL_ROLLOUT`,
`total_rollout_steps`, `LR_DECAY_STEPS`.

| setting | meaning | example |
|---|---|---|
| `ROLLOUT_NNODES=0` | one node, split into `N_GPUS_ROLLOUT` rollout GPUs and the rest for training | `N_GPUS_ROLLOUT=4`: 4 generate, 4 train |
| `ROLLOUT_NNODES=1` | one whole node generates, the rest train | 2 nodes: 8 and 8 (agentic-search) |
| `ROLLOUT_NNODES=2` | two nodes generate | 4 nodes: 16 and 16 |

A 35B model wants whole-node splits: the trainer gets a full node's HBM and host RAM for offload,
and rollout tensor parallelism stays on NVLink (`GEN_TP=8`). With a judge,
`compute.num_accelerators` covers every role; `TRAINING_NODES` splits training from judging
first, then `ROLLOUT_NNODES` splits the training nodes. The math job's 32 GPUs:

```
4 nodes (32 GPUs)
├── TRAINING_NODES=2: ranks 0-1 run GRPO (ROLLOUT_NNODES=1: rank 0 generates, rank 1 trains)
└── JUDGE_NODES=2:    ranks 2-3 serve the judge at TP=16
```

### Weight syncs and checkpoints

```
samples between weight syncs = TRIGGER_SYNC_STEP × REQUIRE_BATCHES × ppo_mini_batch_size
total weight syncs           = total_rollout_steps / samples between weight syncs
```

`SAVE_FREQ` counts weight syncs in async mode and optimizer steps in sync mode. For
`usecases/math/air/4_train.yaml`: 2 × 1 × 16 = 32 samples per sync, 768 / 32 = 24 syncs, and
`SAVE_FREQ=12` saves at 12 and 24. The launcher prints this at startup. Both trainers always save
the final version when `SAVE_FREQ > 0`; that is the checkpoint the completion certificate
verifies, and a run that stops early is reported as failed.

### Staleness

`STALENESS` (`async_training.staleness_threshold`) bounds how far ahead the Rollouter may run.
`0` makes the Trainer wait for fresh samples; higher values buy overlap with older samples.
agentic-search uses `0.1` and math `0.5`, because generation, training and judging all overlap
there. `PARTIAL_ROLLOUT=True` keeps partly generated sequences across a weight sync. The
rollout:trainer ratio affects learning, not only throughput. A 2:1 split (two rollout nodes,
one trainer) raised effective staleness enough to cost about 2-3 EM points against 1:1 at the
same step count. To scale a run up, scale both sides.

## separate_async

`run_grpo_megatron.sh` also implements verl's v1 disaggregated trainer
(`trainer.v1.trainer_mode=separate_async`; knobs `PARAM_SYNC_STEP`, `ASYNC_WARMUP_BATCHES`,
`MAX_OFF_POLICY`, `CKPT_ENGINE_BACKEND`). It places the standalone rollout at
`start_rank=hybrid_num_replicas`, which is 0 when `hybrid_engine=False`, so the rollout lands on
the trainer's GPUs and vLLM runs out of memory. Use the fully-async recipe instead.

## Offline

Not implemented. It would be a collect job that scores the current policy's trajectories and
writes `(prompt, trajectory, reward)` records (mostly what `engine/serve/serve_and_eval.sh`
does already), and a train job on that buffer with no rollout GPUs. Iteration gets much
cheaper, but the data is off-policy and needs importance weighting or a method that tolerates it.

## What has been run

A setting only one mode reads is refused in the other (`engine/lib/preflight.py`).

| | sync | fully-async |
|---|---|---|
| single-turn GRPO, rule reward | yes: geo3k rungs 1-4, including the 35B MoE | yes |
| multi-turn agentic tool loop | tried: the search recipe runs out of memory as configured | yes, both use cases |
| LLM-judge reward (`REWARD_MANAGER=rate_limited`) | reward wiring composes; a co-located judge is refused | yes (math) |
| co-located judge nodes | refused by the dispatcher | yes |
| `NORM_ADV_BY_STD_IN_GRPO` | yes | yes |
| Megatron-FSDP (ZeRO-3) | yes (`MEGATRON_MODE=fsdp`) | no, always classic with CPU offload |
| CPU offload | yes (`OFFLOAD=1`), not with FSDP (preflight refuses it) | always on |

## Which mode

Use sync to debug a new reward or tool on a single-turn task: there is no staleness and failures
are easy to read. Use async to scale up a run that works, for agentic runs, and for any reward
that calls a service (`rate_limited` makes the calls concurrent and the judge overlaps with
training). For iterating on the reward itself, use a small `total_rollout_steps` and the
baseline probe until an offline mode exists. After the mode, the reward decides whether a run
learns anything; see [tuning.md](tuning.md).
