# Tuning

Most RL knobs should stay where they are. This page lists the ones that matter for the shipped
use cases, in two groups that you treat differently. Exact names and defaults are in
[configuration.md](configuration.md), individual verl flags in
[verl-config-reference.md](verl-config-reference.md), memory in [sizing.md](sizing.md).

## Knobs that make it run

These fit a 35B MoE onto the GPUs and make the rollout work. They follow from the model and the
GPU count, not from your task. Leave them alone unless you change one of those, and then start
from [sizing.md](sizing.md).

| knob | setting | why |
|---|---|---|
| `TP` / `PP` / `CP` / `EP` / `ETP` | `EP=8`, `TP` 1 or 2 | 92.5% of the weights are routed experts, so `EP` is the main memory lever |
| backend and offload | sync: Megatron-FSDP, no offload; async: classic Megatron with CPU offload | two answers to the same memory problem ([training-modes.md](training-modes.md)) |
| `GEN_TP` | `8` | keeps rollout tensor parallelism inside one node, on NVLink |
| `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE` | `True` | vLLM's custom all-reduce crashes CUDA-graph capture on H100 at intra-node `GEN_TP≤8` |
| `TOOL_FORMAT` | `qwen3_coder` | Qwen3.5 writes XML tool calls; the wrong parser silently drops every one |
| `use_remove_padding=False`, `use_dynamic_bsz=False` | fixed | Gated-DeltaNet has no packed-sequence (THD) support, so everything runs BSHD |
| `CUDA_DEVICE_MAX_CONNECTIONS` | set by the launcher | `1` for classic, unset for FSDP. Don't set it in a YAML |
| episode length | `(prompt + response) × turns` | the whole multi-turn trajectory has to fit vLLM's context |

## Knobs that make it learn

| knob | shipped value | what it does |
|---|---|---|
| the reward function | exact match (search), LLM judge (math) | the biggest lever, because it is the task. Score the eval with the same code |
| reward variance | measure it first | GRPO learns only from reward differences inside a group of `rollout_n` samples. A group where every sample scores the same carries no task signal; only the KL term still moves the policy. Measure the share of mixed groups before a training run (`infra/geo3k/air/2_baseline.yaml`, or `EVAL_N_SAMPLES` in the search eval). pass@1 doesn't tell you: 95% can still leave mixed groups, 50% can leave none |
| `NORM_ADV_BY_STD_IN_GRPO` | `True` (verl's default, used by every shipped run) | whether each group's advantages are divided by the group's std. That keeps a graded reward's order and relative gaps: `[0, 0.05, 0.7, 1]` becomes `[−0.89, −0.79, 0.53, 1.14]`. What changes is the weight between groups: a group whose scores barely differ gets advantages as large as a group with a clear winner. `False` keeps advantages in reward units. Worth an ablation |
| `MAX_TURNS` | `12` (up from 8) | the agent's hop budget and the main lever for agentic-search. Also the main cost in the actor backward at micro-batch 1 |
| `rollout_n` | 16 (search), 4 (math) | group size. Larger gives a steadier advantage at linear cost; going from 16 to 32 did not help here |
| `ROLLOUT_TEMP` | 1.0 to 1.2 | hotter sampling gives more varied groups |
| `actor_lr` | 1e-6 to 2e-6 | too high risks collapse, too low wastes the run |
| `kl_loss_coef` | 0.01, `low_var_kl` | raise it if the policy degenerates (repetition, lost format), lower it if the policy can't move |
| `STALENESS` / `ROLLOUT_NNODES` | 0.1, a 1:1 split | freshness against throughput. A 2:1 rollout:trainer split cost about 2 to 3 EM points ([training-modes.md](training-modes.md)) |
| `total_rollout_steps` | eval plateaued by step 20 to 40 here | watch the eval curve rather than guessing a horizon |
| `SAVE_FREQ` | 10 (search), 12 (math) | the interval between saved checkpoints; the final one is always saved, so this only sets the intermediates you can evaluate |
| the data | MuSiQue rather than HotpotQA, MATH rather than GSM8K | data that is too easy saturates the reward. Both use cases changed dataset for that reason |

## Order of work

1. Write the reward and score the eval with the same function. Unit-test it on a CPU
   (`make test`).
2. Run the eval job on the untrained model, at the settings you will use later.
3. Check reward variance. If the base model is at 95% or at 2%, fix the data or make the reward
   graded before touching training.
4. Smoke the training job with a small `total_rollout_steps` and `SAVE_FREQ=1`. You're checking
   that tools fire, the reward is called and checkpoints get written.
5. Train and save several checkpoints. Pick one on a dev split, then score that one once on a
   held-out test split. Picking on the number you report inflates it.
6. Read the traces before changing knobs. For retrieval, `analyze_traces.py` splits EM by
   whether a gold answer string appeared in a tool output. Use it to choose the next
   experiment, not as a finding.
7. Change one thing and re-evaluate with identical eval settings.

## Symptoms

| symptom | likely cause | change |
|---|---|---|
| reward flat and high from the first step | the task is saturated | harder data or a graded reward |
| reward flat near zero | a gate never passes | run the reward on real model output and check format gates (geo3k scores a correct but unboxed answer 0) |
| a graded reward acts like pass/fail | the reward is nearly binary, or its partial credit barely varies inside a group | look at the within-group spread, not the mean |
| the agent never calls tools, `Failed to decode tool call` in the log | wrong `TOOL_FORMAT` | `qwen3_coder` for Qwen3.5; check with `probe_tool_format.yaml` |
| trained model barely beats base, and the turn budgets differ | eval mismatch | use the same `EVAL_MAX_TURNS` for both evals |
| a reward bonus changes nothing | it fires on nearly every sample, so it only moves the group mean | make the bonus vary within the group |
| OOM in the actor backward | episode length | lower `MAX_TURNS` first, or `ppo_max_token_len_per_gpu` |
| OOM at the co-located weight sync | the FSDP full-tensor gather meets a woken vLLM | more GPUs. `ROLLOUT_GPU_MEM_UTIL` only sizes the KV cache, which is asleep during the sync |
| `No available memory for the cache blocks` | the KV cache can't hold one `MAX_MODEL_LEN` sequence | raise `ROLLOUT_GPU_MEM_UTIL` if nothing else is on those GPUs, raise `GEN_TP`, or lower `MAX_MODEL_LEN`. Lowering the utilisation makes it worse |
| the judge is the bottleneck | `REWARD_MAX_CONCURRENT` defaults to 1 in verl | set it (the math job uses 64) |
| run FAILED with "the plan reaches N" | it stopped before its final version (a swallowed crash, an abort, a timeout) | read `run_result.json` and the log |
| slow multi-turn rollout | the shared prompt is prefilled again every turn | `ROLLOUT_PREFIX_CACHING=True` (verl flushes the cache on weight sync) |
| multi-node throughput collapsed | NCCL fell back to TCP | grep the log for `Selected Provider is efa`; `NET/Socket` means no RDMA |

## Per use case

- agentic-search: `MAX_TURNS` and the plain exact-match reward. No judge, nothing to serve.
- math: the judge reward (a graded surrogate; MATH-500 correctness is the independent target),
  `REWARD_MAX_CONCURRENT` for judge throughput, and the calculator tool.
- geo3k: not a task to tune. It exists to prove the topology.

## What to expect

Knobs move the number less than you'd hope, and one run per setting can't tell them apart. On
agentic-search, a longer run, `rollout_n` 32 and a retrieval bonus each landed between 51% and
56% EM against a 54% base, on one 200-question dev set, and none differed significantly from the
base ([RESULTS.md](../RESULTS.md)). That isn't a ceiling, just the resolution of single runs.
Repeat seeds and compare paired on a held-out split before concluding anything.
