# The knobs that matter

RL has hundreds of knobs. Most of them you should never touch. This page is the
curated short list for *these* use cases: what we set, why, and — the important part —
**which knobs are yours to tune** vs. which we already solved for you. The exhaustive
reference is [`verl-config-reference.md`](verl-config-reference.md); this page sits
above it.

The single most useful idea: there are **two kinds of knob**, and you treat them
oppositely.

## Tier 1 — "make it run" (we solved these; leave them unless you change model or GPU count)

These are about fitting a 35B MoE onto the GPUs and getting the rollout to work at all.
They're derived from the model + hardware, not from your task.

| knob | our setting | why it's fixed |
|---|---|---|
| `TP` / `PP` / `CP` / **`EP`** / `ETP` | `EP=8` @ 16 GPU | MoE expert sharding; follows from the model + GPU count — see [`sizing.md`](sizing.md). |
| FSDP vs classic Megatron | **Megatron-FSDP, no offload @ 16 GPU** | ZeRO-3 shards optimizer+grads+params; the smallest offload-free topology for this 35B model. |
| `GEN_TP` + `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `8`, `True` | at `GEN_TP≤8` vLLM's custom all-reduce crashes CUDA-graph capture on H100 — a correctness fix, not a dial. |
| `TOOL_FORMAT` | `qwen3_coder` | Qwen3.5 emits **XML** tool calls, not JSON — the wrong parser silently zeros all tool use. |
| episode length | `(prompt+response)×turns` | the whole multi-turn trajectory must fit vLLM's context window. |

If you bring a different base model or a different GPU count, revisit Tier 1 (start at
`sizing.md`). Otherwise, don't.

## Tier 2 — "make it learn" (tune THESE for your task)

| knob | our value | what it does / when to change |
|---|---|---|
| **the reward function** | rule-EM (search) · LLM-judge (math) | The #1 lever. This *is* your task; everything else is secondary. Start here. |
| `algorithm.norm_adv_by_std_in_grpo` | **`False` for graded rewards** | With std-normalisation on, a 0.05 and a 1.0 reward get the *same* within-group advantage — a graded reward collapses to binary. Turn it off (env: `NORM_ADV_BY_STD_IN_GRPO=False`) whenever reward is graded, not 0/1. |
| reward **variance** (the gate) | run the baseline probe first | GRPO's gradient comes *entirely* from reward variance within each group of `rollout_n` samples. A group where every sample scores alike teaches nothing. Measure the fraction of groups with non-zero variance **before** you spend a training run (`infra/geo3k/air/2_baseline.yaml` shows the pattern). |
| **`MAX_TURNS`** | `8 → 12` | ⭐ agentic-search's headline lever: the agent's tool/hop budget. Going 8→12 lifted base EM +2 and was recall-safe. Match it to how many hops your task needs. |
| `rollout_n` / `ROLLOUT_TEMP` | `16` / `1.0` (search), `1.2` (officeqa) | group size + sampling temperature → how dense/diverse the reward signal is. Bigger/hotter = lower-variance advantage, more compute. We found `16→32` didn't help *our* plateau. |
| `actor_lr` / `kl_loss_coef` | `2e-6` / `0.01` (low-var KL) | step size + how hard the policy is leashed to the reference model. |
| `STALENESS` / `ROLLOUT_NNODES` | `0.1` / 1:1 split | async throughput vs on-policy freshness — see [`training-modes.md`](training-modes.md). **Gotcha:** a 2:1 rollout:trainer split cost us ~2–3 EM (not learning-neutral). |
| `total_rollout_steps` | plateaued by ~step 20–40 | the training horizon. Watch the eval curve for a plateau rather than guessing a number. |

## Per-use-case headline knob

- **agentic-search** → `MAX_TURNS` (the retrieval hop budget) + the pure rule-based EM
  reward. No judge.
- **math** → the LLM-judge reward + `NORM_ADV_BY_STD_IN_GRPO=False` (the judge score is
  graded) + the calculator tool.
- **geo3k (infra)** → not a task to tune; it's the FSDP-vs-classic topology proof — a
  Tier-1 demonstration.

## The honest bit

Tuning Tier 2 moves the number, but not without limit. On agentic-search we found a
**capability ceiling**: once conversion (answer-given-retrieval) and recall were both
near their reachable band, more compute (deeper runs, denser groups) and a
retrieval-shaped reward did **not** break the plateau. Getting materially further needed
a signal/inference change, not more of the same knob. See [`../RESULTS.md`](../RESULTS.md).
