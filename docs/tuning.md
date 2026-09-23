# The knobs that matter

← [verl-on-air](../README.md) · [the case study](../RESULTS.md) · [configuration](configuration.md) · [running-jobs](running-jobs.md)

RL has hundreds of knobs. Most you should never touch. This page is the curated list for
*these* use cases: what we set, why, and — the important part — **which knobs are yours
to tune** versus which are already solved for you.

Three companions: [configuration.md](configuration.md) is the exhaustive name/default
reference, [verl-config-reference.md](verl-config-reference.md) explains every individual
verl flag, and [sizing.md](sizing.md) has the memory arithmetic. This page sits above all
three and tells you *what to do*.

The single most useful idea: there are **two kinds of knob**, and you treat them
oppositely.

---

## Tier 1 — "make it run" (solved; leave alone unless you change model or GPU count)

These are about fitting a 35B MoE onto the GPUs and getting the rollout to work at all.
They are derived from the model and the hardware, **not** from your task. If you bring a
different base model or a different GPU count, revisit them starting at
[sizing.md](sizing.md). Otherwise don't.

| knob | our setting | why it's fixed |
|---|---|---|
| `TP` / `PP` / `CP` / **`EP`** / `ETP` | `EP=8`, `TP` 1–2 | MoE expert sharding; 92.5% of this model's weights are routed experts, so `EP` is the dominant lever. Follows from model + GPU count |
| backend + offload | sync: Megatron-FSDP no-offload · async: classic + CPU offload | two different answers to the same memory problem — see [training-modes.md](training-modes.md) §3 |
| `GEN_TP` | `8` | keeps rollout tensor-parallel intra-node (NVLink); extra nodes become DP replicas |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `True` | at intra-node `GEN_TP≤8` vLLM's custom all-reduce crashes CUDA-graph capture on H100. A correctness fix, not a dial |
| `TOOL_FORMAT` | `qwen3_coder` | Qwen3.5 emits **XML** tool calls, not JSON. The wrong parser silently zeroes all tool use |
| `use_remove_padding=False`, `use_dynamic_bsz=False` | fixed | Qwen3.5's Gated-DeltaNet has no THD packing → BSHD everywhere. Correctness, not tuning |
| `CUDA_DEVICE_MAX_CONNECTIONS` | launcher-managed | `1` for classic; **must be unset for FSDP** or collectives serialise behind compute. Do not set it in a YAML |
| episode length | `(prompt+response) × turns` | the whole multi-turn trajectory must fit vLLM's context |

---

## Tier 2 — "make it learn" (tune THESE)

| knob | our value | what it does / when to change |
|---|---|---|
| **the reward function** | rule-EM (search) · LLM-judge (math) | The #1 lever. This *is* your task; everything else is secondary. Start here, and make the eval scorer the same code |
| **reward variance** (the gate) | measure it first | GRPO's task-reward policy gradient comes *entirely* from reward differences within each group of `rollout_n` samples. A group where every sample scores alike carries **no task-reward signal** (only the KL term still moves the policy). Measure the fraction of groups with any spread **before** spending a training run (`infra/geo3k/air/2_baseline.yaml`) — aggregate pass@1 does not tell you: 95% can still leave mixed groups, 50% can leave none |
| `algorithm.norm_adv_by_std_in_grpo` | `True` (verl's default; what every shipped run used) | Whether each group's advantages `r − mean` are divided by the group's std. Dividing does **not** turn a graded reward into pass/fail: it rescales a group to unit spread and keeps its order and relative gaps — `[0, 0.05, 0.7, 1]` becomes `[−0.89, −0.79, 0.53, 1.14]`. What changes is the weight *between* groups: a group whose scores barely differ (judge noise around one value) gets advantages as large as a group with a clear winner. `False` keeps advantages in reward units, so near-ties count less. An empirical choice — ablate it; the math job sets `True` explicitly until one has |
| **`MAX_TURNS`** | `8 → 12` | ⭐ agentic-search's headline lever: the agent's tool/hop budget. 8→12 lifted base EM ~2 points and was recall-safe. Also the primary **backward-memory** cost at `micro_bsz=1` |
| `rollout_n` | `16` (search) / `4` (math) | GRPO group size → how dense the advantage signal is. Bigger = lower-variance advantage, linearly more compute. `16→32` did **not** help our plateau |
| `ROLLOUT_TEMP` | `1.0`–`1.2` | sampling temperature. Hotter = more diverse group = more reward variance (the thing GRPO needs) |
| `actor_lr` | `1e-6`–`2e-6` | step size. Higher risks collapse, lower wastes the run |
| `kl_loss_coef` | `0.01`, `low_var_kl` | how hard the policy is leashed to the reference model. Raise if the policy degenerates (repetition, format loss); lower if it cannot move |
| `STALENESS` / `ROLLOUT_NNODES` | `0.1` / 1:1 | async freshness vs throughput. **Gotcha:** a 2:1 rollout:trainer split cost ~2–3 EM — not learning-neutral ([training-modes.md](training-modes.md) §4) |
| `total_rollout_steps` / `total_epochs` | plateaued by ~step 20–40 | the training horizon. Watch the *eval* curve for a plateau rather than guessing a number |
| `SAVE_FREQ` | divisor of the total | not a learning knob, but the one that decides whether you *have* a model to evaluate |
| the **data** | MuSiQue not HotpotQA · MATH not GSM8K | dataset difficulty is a tuning knob: too easy saturates the reward and nothing learns. Both use cases picked their dataset for exactly this reason |

---

## The procedure (do it in this order)

Most wasted RL spend comes from doing these out of order.

1. **Write the reward — and score the eval with the same function.** If "what you optimise"
   and "what you measure" are two implementations, they will drift and you will not know
   which number to trust. Unit-test the reward on the CPU (`make test`: no GPU, seconds).
2. **Measure the baseline.** Run the eval job against the *untrained* model, at the exact
   settings you will use later. This is your only honest reference point, and it validates
   the whole harness for one node-hour.
3. **Check reward variance.** Does the reward actually separate samples within a group?
   If the base model is at 95% or at 2%, fix the *data* (difficulty band) or the *reward*
   (make it graded) before touching training.
4. **Smoke the training job.** Small `total_rollout_steps`, `SAVE_FREQ=1`. You are
   checking that tools fire, the reward is called, checkpoints write — not that it learns.
5. **Train, checkpointing a few times.** Then evaluate **several** checkpoints. The best
   held-out checkpoint is usually not the last (ours was step 20, then a plateau).
6. **Diagnose before you tune.** Decompose the metric (for retrieval:
   `EM = recall × conversion`, via `analyze_traces.py`). A diagnostic tells you *which*
   knob; without one you are guessing across a hundred dials.
7. **Change one thing.** Then re-evaluate at identical eval settings.

---

## Symptom → knob

| what you see | most likely cause | what to change |
|---|---|---|
| reward flat and high from step 1 | task saturated — no variance | harder data / graded reward. See math's MATH-vs-GSM8K note |
| reward flat and near zero | reward unreachable (a gate never passes) | run the reward on real model output; check format gates (e.g. geo3k scores 0, not 0.9, for a correct-but-unboxed answer) |
| graded reward behaves like pass/fail | the reward itself is near-binary (a judge that mostly says 0 or 1), or its partial credit barely varies within a group — *not* std-normalisation, which keeps a group's order and gaps | inspect the within-group score spread, not the mean reward |
| agent never calls its tools; log shows `Failed to decode tool call` | wrong `TOOL_FORMAT` | `qwen3_coder` for Qwen3.5; verify with `probe_tool_format.yaml` |
| trained model barely beats base, but turn budgets differ | eval mismatch | make `EVAL_MAX_TURNS` identical in baseline and trained eval |
| an added reward bonus changes nothing | the bonus fires on nearly every sample in the group | it lands in the group mean too — advantage is `(r − mean)/std`. Make the signal *discriminative*, not uniform |
| OOM in the actor backward | episode length × turns | lower `MAX_TURNS` (primary), or `ppo_max_token_len_per_gpu` |
| OOM at the weight sync (co-located) | FSDP full-tensor gather colliding with woken vLLM | more GPUs (thinner shards). `ROLLOUT_GPU_MEM_UTIL` is **not** the lever — it sizes the KV cache, which is asleep during the sync |
| `No available memory for the cache blocks` | KV cache vs episode length | lower `ROLLOUT_GPU_MEM_UTIL` or `MAX_MODEL_LEN` |
| judge is the bottleneck | `REWARD_MAX_CONCURRENT` defaults to **1** inside verl | set it (the math job uses 64) |
| run finished, no checkpoint | `SAVE_FREQ` never divided the total | pick a divisor of the sync/step count the launcher prints |
| slow multi-turn rollout | re-prefilling the shared prompt every turn | `ROLLOUT_PREFIX_CACHING=True` (safe: verl flushes on weight sync) |
| throughput collapsed on multi-node | NCCL fell back to TCP | grep for `Selected Provider is efa`; `NET/Socket` means no RDMA |

---

## Per-use-case headline knob

- **agentic-search** → `MAX_TURNS` (the retrieval hop budget) plus the pure rule-based EM
  reward. No judge, nothing to serve, cheapest agentic loop in the repo.
- **math** → the LLM-judge reward (a graded *surrogate*; deterministic MATH-500 correctness
  is the independent target), `REWARD_MAX_CONCURRENT` (judge throughput), and the calculator
  tool. `NORM_ADV_BY_STD_IN_GRPO` is `True`, as run, until an ablation says otherwise.
- **geo3k (infra)** → not a task to tune. It is the FSDP-vs-classic topology proof — a
  Tier-1 demonstration.

---

## The honest bit

Knobs move the number less than you would hope, and a single run cannot tell you by how
much. On agentic-search, a longer run, `rollout_n` 16→32 and a retrieval-shaped reward
each landed between 51% and 56% EM against a 54% base — single runs on one 200-question
development set, none significantly different from the base (see
[../RESULTS.md](../RESULTS.md) for the paired statistics). That is not evidence of a
ceiling; it is evidence that one run per setting cannot separate these levers. Before
concluding anything, repeat seeds and compare paired on a held-out split. Ideas worth
testing next — self-consistency at eval, a recall-weighted advantage, a harder-negative
curriculum — are not implemented here.
