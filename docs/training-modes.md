# Training modes: synchronous, fully-async, and what "offline" would take

← [verl-on-air](../README.md) · [running-jobs](running-jobs.md) · [configuration](configuration.md) · [sizing](sizing.md)

RL post-training has to arrange two different workloads — **generation** (rollout) and
**optimisation** (trainer) — on a fixed pool of GPUs. How you arrange them is the single
biggest structural choice in an RL job, and it is independent of your task, your data
and your reward.

This repo ships both live arrangements as separate engine launchers and picks between
them with **one env var**.

```
SYNCHRONOUS (co-located)                  FULLY-ASYNC (disaggregated)
┌──────────────────────────┐              ┌───────────────┐   ┌───────────────┐
│  all GPUs                │              │ Rollouter     │   │ Trainer       │
│  generate → score → step │              │ GPUs 0..7     │   │ GPUs 8..15    │
│  generate → score → step │              │ generate      │◄─►│ optimise      │
│        (lockstep)        │              │ continuously  │MQ │ continuously  │
└──────────────────────────┘              └───────────────┘   └───────────────┘
 on-policy, simplest, GPUs                 NCCL weight sync every N updates;
 idle in turns                             overlap, at the cost of staleness
```

| mode | launcher | `TRAIN_MODE` | rollout ↔ trainer | policy | use when |
|---|---|---|---|---|---|
| **synchronous** (co-located) | `engine/train/run_grpo_megatron.sh` | `sync` | share the same GPUs, lockstep | strictly on-policy | simplest; fewest moving parts; every rollout comes from the current weights. **The geo3k ladder runs this.** |
| **fully-async** (disaggregated) | `engine/train/run_grpo_fully_async.sh` | `async` *(default)* | **disjoint** GPU pools + a MessageQueue + NCCL weight sync | bounded-stale | throughput: generation and training overlap. **Both use cases run this.** |
| *separate_async* (verl v1) | `run_grpo_megatron.sh` + `TRAINER_MODE=separate_async` | — | disjoint pools via verl's v1 trainer | bounded-stale | **not recommended** — see §5 |
| **offline** | *not implemented* | — | none: train on a pre-collected, pre-scored buffer | off-policy | reuse rollouts, iterate on reward cheaply. §6 sketches it |

---

## 1. How to switch

Switching an agentic job between modes is **not one knob**: the two modes need different
node counts, backends and step budgets, so each has its own job file.

```bash
# fully-async (the measured configuration): 16 GPUs = 1 rollout node + 1 trainer node
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch        # make search-train

# synchronous / on-policy: 32 GPUs, all 4 nodes train, Megatron-FSDP, 100 optimizer steps
air run --file usecases/agentic-search/air/4_train_sync.yaml -p df1 --watch   # make search-train-sync
```

> **`4_train_sync.yaml` is config-validated only.** It composes against the pinned verl
> (`make compose-check`) but has not run on GPUs; the measured result in
> [RESULTS.md](../RESULTS.md) comes from the fully-async job.

What differs between the two files, and why:

| | `4_train.yaml` (async) | `4_train_sync.yaml` (sync) |
|---|---|---|
| nodes | 2 = 1 Rollouter + 1 Trainer (`ROLLOUT_NNODES=1`) | 4, all training (`TRAINING_NODES=4`, `ROLLOUT_NNODES=0`) |
| backend | classic Megatron (ZeRO-1) + CPU-offloaded optimizer | Megatron-FSDP (ZeRO-3), no offload — the ladder's rung 4 |
| budget | `total_rollout_steps: 3200` prompt groups → 100 weight syncs | `total_training_steps: 100` × `train_batch_size: 32` = 3200 prompt groups |
| data order | shuffled (verl's default) | `DATA_SHUFFLE: 'True'` |
| policy lag | bounded (`STALENESS`) | none; a failed group is dropped, not retried |

The dispatcher (`engine/train/dispatch_agentic.sh`) reads `TRAIN_MODE`, execs the matching
launcher, and refuses — identically on every rank, before any role starts — the
combinations that used to fail late or silently:

- nodes left over for an LLM judge the job does not configure (e.g. `TRAIN_MODE=sync` +
  `compute.num_accelerators=32` with `TRAINING_NODES` still `2`);
- `TRAIN_MODE=sync` without an explicit `parameters.total_training_steps` (the sync
  launcher's default is a 3-step smoke cap);
- a co-located judge in sync mode (never run);
- `ROLLOUT_NNODES>0` in sync mode (would silently shrink `trainer.nnodes`).

The geo3k rungs call `run_grpo_megatron.sh` directly (no dispatcher, no tools, no
judge), which is why they are the clean reference for sync mode.

### Verify before you spend

Both launchers print their fully-resolved verl invocation and exit when `DRY_RUN=1`,
without starting Ray — this runs on a laptop:

```bash
make config MODE=fsdp GPUS=16          # sync launcher, resolved
make diff-modes                        # what actually differs between fsdp and classic

make compose-check                     # every training job's command, as rank 0 would run it,
                                       # composed against the pinned verl + invariants
```

---

## 2. Synchronous, concretely

One loop, one GPU pool: generate a batch → score it → take optimizer steps → repeat.
Every gradient is computed on samples from the **current** policy.

- **Pros.** On-policy, so no staleness to reason about and the most learning per
  sample. One process tree, one Ray cluster, one failure mode. Easiest to debug.
- **Cons.** Generation and training never overlap, so each is idle while the other runs.
  Worse, they must **coexist in memory**: vLLM's weights and KV cache sit on the same
  GPUs as the optimizer state, and the actor→vLLM weight sync creates a *transient* peak
  on top of both. That transient — not steady-state training — is what pushed the 35B
  co-located config from 16 GPUs to 32 ([sizing.md](sizing.md)).

Sync-only knobs: `MEGATRON_MODE` (`fsdp` = ZeRO-3 vs `classic` = ZeRO-1), `OFFLOAD`,
`train_batch_size`, `total_training_steps`, `VAL_BEFORE_TRAIN`, `WEIGHT_BUCKET_MB`.

## 3. Fully-async, concretely

A **Rollouter** and a **Trainer** run as separate processes on **disjoint** GPUs, joined
by a MessageQueue; the Trainer pushes fresh weights to the Rollouter over NCCL every so
often. Both run continuously, so utilisation is much higher — at the cost of
**staleness**: the Rollouter may be a few parameter versions behind.

Because the two sides no longer share GPUs, the memory fight disappears: the shipped
async trainer uses **classic Megatron (ZeRO-1) with CPU offload on its own node**, which
is how 35B trains with 8 trainer GPUs + 8 rollout GPUs. That is a different strategy from
sync's Megatron-FSDP, and the reason the same model needs 16 GPUs in one mode and 32 in
the other. (The async launcher does not currently expose `MEGATRON_MODE`; it is always
classic+offload. Adding FSDP there is an obvious extension.)

Async-only knobs: `ROLLOUT_NNODES` / `N_GPUS_ROLLOUT` (the split), `STALENESS`,
`TRIGGER_SYNC_STEP`, `REQUIRE_BATCHES`, `PARTIAL_ROLLOUT`, `total_rollout_steps`,
`LR_DECAY_STEPS`, plus `REWARD_MANAGER` and `NORM_ADV_BY_STD_IN_GRPO`.

### Sizing the split

`ROLLOUT_NNODES` decides the shape:

| setting | meaning | example |
|---|---|---|
| `ROLLOUT_NNODES=0` | **within-node** split: one node divided into `N_GPUS_ROLLOUT` rollout GPUs + the rest trainer | 1 node, `N_GPUS_ROLLOUT=4` → 4 gen + 4 train |
| `ROLLOUT_NNODES=1` | **whole-node** split: 1 entire node generates, the rest train | 2 nodes → 8 gen + 8 train (**agentic-search**) |
| `ROLLOUT_NNODES=2` | 2 nodes generate | 4 nodes → 16 gen + 16 train |

Whole-node splits are what a 35B model wants — the trainer gets a full node's HBM and
host RAM for offload, and rollout tensor-parallel stays intra-node on NVLink
(`GEN_TP=8`).

With a judge, the accounting stacks: `compute.num_accelerators` covers **all** roles, and
`TRAINING_NODES` splits train-vs-judge *before* `ROLLOUT_NNODES` splits train into
rollout-vs-trainer. The math job's 32 GPUs are:

```
4 nodes total (32 GPUs)
├── TRAINING_NODES=2  ──►  ranks 0-1: GRPO
│                          └── ROLLOUT_NNODES=1 → rank 0 generates, rank 1 trains
└── ranks 2-3: serve the judge at TP=16
```

### The cadence arithmetic

```
samples between weight syncs = TRIGGER_SYNC_STEP × REQUIRE_BATCHES × ppo_mini_batch_size
total weight syncs           = total_rollout_steps ÷ (that number)
SAVE_FREQ counts WEIGHT SYNCS in async mode (optimizer steps in sync mode)
```

Worked, for `usecases/math/air/4_train.yaml`: `2 × 1 × 16 = 32` samples per sync;
`768 / 32 = 24` syncs; `SAVE_FREQ=12` → checkpoints at 12 and 24. The launcher prints
exactly this banner at startup — read it before walking away, because picking a
`SAVE_FREQ` that does not divide the total is how a run finishes with **zero**
checkpoints.

### Staleness

`STALENESS` (`async_training.staleness_threshold`) bounds how far ahead the Rollouter may
run. `0` makes the Trainer wait for fresh samples (synchronous behaviour with the
disaggregated topology); higher values buy overlap and accept older samples.
agentic-search uses `0.1`; the math job uses `0.5` because it has *three* things to
overlap (generate ‖ train ‖ judge). `PARTIAL_ROLLOUT=True` keeps partially-generated
sequences across a sync instead of throwing that work away.

---

## 4. Measured gotcha: the split ratio is not learning-neutral

A 2:1 rollout:trainer split (2 rollout nodes, 1 trainer) raised effective staleness
enough to cost roughly **2–3 EM points** versus a 1:1 split at the same step count. More
generation capacity than the trainer can consume just produces staler samples.

**If you scale a deep run, scale both sides** (2 trainer + 2 rollout), not just the
rollout pool. Treat the split ratio as a hyperparameter with a learning effect, not
purely as a throughput dial.

---

## 5. `separate_async` — present, but not the path we recommend

`run_grpo_megatron.sh` also implements verl's **v1** disaggregated trainer
(`trainer.v1.trainer_mode=separate_async`, knobs `PARAM_SYNC_STEP`,
`ASYNC_WARMUP_BATCHES`, `MAX_OFF_POLICY`, `CKPT_ENGINE_BACKEND`). We use
`experimental.fully_async_policy` instead, for a concrete reason:

`separate_async` places the standalone rollout at `start_rank=hybrid_num_replicas`, which
is `0` when `hybrid_engine=False` — so the rollout lands on the **same GPUs as the
trainer** and vLLM OOMs against ~31 GiB of resident Megatron state.
`fully_async_policy` takes **top-level** `rollout.nnodes` / `rollout.n_gpus_per_node`
and keeps the trainer GPUs rollout-free, which is real disaggregation.

It is kept because it is a legitimate verl feature, its asserts are documented in the
launcher, and a future verl release may fix the placement. Do not reach for it first.

---

## 6. Offline / off-policy — not implemented, and what it would take

Nothing offline ships here. The shape would be two jobs:

1. **collect** — run the current policy over prompts, score the trajectories, and write
   `(prompt, trajectory, reward)` records to the Volume. This is the eval harness plus a
   writer: `engine/serve/serve_and_eval.sh` already serves a model and drives a
   use case's agent loop against it, which is most of the work.
2. **train** — optimise on that static buffer with no live generation, so the job needs
   no rollout GPUs at all.

Why it is attractive: rollout is the expensive half, so reusing trajectories makes
reward and algorithm iteration dramatically cheaper, and the reward function stops having
to keep up with generation. Why it is not free: the data is off-policy by construction,
so it needs importance weighting or a method tolerant of that; and a buffer collected
from one policy goes stale as the policy moves.

If you build it, the honest first milestone is reproducing a known on-policy result from
a collected buffer — not a new number.

---

## 7. Support matrix — what has actually been run

Being precise here matters more than looking complete.

| | sync (co-located) | fully-async (disaggregated) |
|---|---|---|
| single-turn GRPO, rule reward | ✅ **measured** — geo3k rungs 1–4, incl. 35B MoE | ✅ **measured** |
| multi-turn agentic tool loop | ⚠️ **wired + `DRY_RUN`-validated**, not run on GPU | ✅ **measured** — both use cases |
| LLM-judge reward (`REWARD_MANAGER=rate_limited`) | ❌ not plumbed — use async | ✅ **measured** — the math use case |
| co-located judge nodes (`TRAINING_NODES`) | ✅ dispatcher-level, mode-independent | ✅ **measured** |
| `NORM_ADV_BY_STD_IN_GRPO` | ❌ not plumbed — use async | ✅ |
| Megatron-FSDP (ZeRO-3) | ✅ `MEGATRON_MODE=fsdp` | ❌ always classic + CPU offload |
| CPU offload | ✅ `OFFLOAD=1` (**not** with FSDP: DTensor crash) | ✅ always on |

So: **if your use case needs a judge, use async.** If you want strictly on-policy
single-turn training, sync is the proven path. The agentic+sync corner is wired so the
switch is one knob, but treat its first run as a bring-up, not a reproduction.

---

## 8. Which mode should you use?

- **Starting out, or debugging a new reward/tool?** Sync. One process tree, no
  staleness, failures are legible.
- **Scaling up a run that works?** Async. Overlap is the whole point, and the shipped
  use cases are tuned for it.
- **Reward is an external service (a judge, an API)?** Async — `rate_limited` makes
  reward calls concurrent instead of serial, and the judge overlaps with training.
- **Iterating on the reward function itself?** You want offline (§6). Until it exists,
  use a small `total_rollout_steps` and the baseline probe.

Then read [tuning.md](tuning.md), because after the mode, the reward function is the
thing that decides whether the run learns anything.
