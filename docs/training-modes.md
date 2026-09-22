# Training modes: synchronous, fully-async (and offline)

RL post-training can run the rollout (generation) and the trainer (weight updates)
in different relationships. This repo ships both live modes as separate engine
launchers, and you pick one per job. This page explains the trade-off and **how to
switch**.

| mode | launcher | rollout ↔ trainer | policy | use when |
|---|---|---|---|---|
| **synchronous** (co-located) | `engine/train/run_grpo_megatron.sh` | share the same GPUs, lockstep | on-policy | simplest; fewer GPUs; maximum sample-efficiency per step. **The geo3k ladder runs this.** |
| **fully-async** (disaggregated) | `engine/train/run_grpo_fully_async.sh` | **disjoint** GPU pools + a MessageQueue + NCCL weight-sync | staleness-tolerant | throughput at scale — generation overlaps training. **Both use cases run this.** |
| **offline** (future) | *not implemented* | none — train on a pre-collected, pre-scored trajectory buffer | off-policy | reuse rollouts, cheap iteration, reward-model work. See the note at the bottom. |

## Synchronous vs fully-async, concretely

- **Synchronous** alternates: generate a batch → score it → take an optimizer step →
  repeat, all on the same GPUs. Every gradient is computed on data from the *current*
  policy (on-policy), which is the most sample-efficient regime, but generation and
  training never overlap so the GPUs idle in turns.
- **Fully-async** runs a **Rollouter** and a **Trainer** as separate processes on
  **disjoint** GPUs, connected by a MessageQueue; the Trainer pushes new weights to the
  Rollouter over NCCL every so often. Generation and training run at the same time, so
  utilisation is much higher — at the cost of *staleness*: the Rollouter may be a few
  updates behind the Trainer. A `staleness_threshold` bounds how far.

## How to switch

The mode is the **launcher your job's `command:` invokes**, plus a few env knobs:

- **Pick the launcher.** Synchronous jobs call `engine/train/run_grpo_megatron.sh`
  directly (see the geo3k rungs). Agentic jobs call `engine/train/dispatch_agentic.sh`,
  which currently hands off to `engine/train/run_grpo_fully_async.sh`.
- **Shape the async split** with env vars read by the launcher:
  - `ROLLOUT_NNODES` — `0` splits one node into rollout + trainer GPUs; `≥1` gives the
    Rollouter that many **whole** nodes and the Trainer the rest (true disaggregation).
  - `STALENESS` (`staleness_threshold`) — `0` behaves synchronously (Trainer waits);
    `>0` lets the Rollouter run ahead. We use `0.1`.
  - `TRIGGER_SYNC_STEP` × `REQUIRE_BATCHES` × `ppo_mini_batch_size` = samples consumed
    between weight syncs (the sync cadence).

> **Note (honest):** `dispatch_agentic.sh` currently hard-wires the async launcher. A
> single `TRAIN_MODE=sync|async` switch in the dispatcher would make the choice one knob
> instead of tribal knowledge — a small, welcome PR.

## Gotcha we hit

The rollout:trainer node ratio is **not** learning-neutral. A 2:1 split (2 rollout
nodes, 1 trainer) raised staleness enough to cost ~2–3 EM points versus a 1:1 split at
the same step count. If you scale a deep run, raise **both** sides (e.g. 2 trainer + 2
rollout), not just the rollout pool.

## Offline (future)

Nothing offline is implemented here. The shape would be: a **collect** job that runs
the current policy over prompts and writes `(trajectory, reward)` records, then a
**trainer** that optimises on that static buffer (no live generation). It's cheaper to
iterate on and decouples reward work from the GPU rollout loop. The OfficeQA
trajectory collector (on a separate branch) is a starting block for the collect half.
