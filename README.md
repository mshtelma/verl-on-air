# verl-on-air

**Agentic reinforcement learning on Databricks AI Runtime — a showcase and a template.**

GRPO post-training of `Qwen3.5-35B-A3B` (MoE) with [verl](https://github.com/volcengine/verl)'s
Megatron backend, on AI Runtime serverless GPU. One shared engine does the hard part
once; a use case is a handful of small files on top. The point of this repo is to show
**how little it takes to run real RL on AI Runtime** — and to give you a template to copy.

> These are examples, not benchmark claims. Swap in your own data, reward, or tool and
> the same jobs run unchanged.

**Start here:** [docs/running-jobs.md](docs/running-jobs.md) — how to run every job ·
[docs/configuration.md](docs/configuration.md) — every setting ·
[docs/training-modes.md](docs/training-modes.md) — sync vs async ·
[docs/tuning.md](docs/tuning.md) — which knobs matter ·
[docs/new-usecase.md](docs/new-usecase.md) — bring your own task.

---

## Flagship: agentic-search

A multi-hop **search agent**. Given a question, `Qwen3.5-35B-A3B` runs a multi-turn tool
loop — search and read over a **Databricks Vector Search** index — and commits an answer.
GRPO trains it with a **rule-based exact-match reward**: no LLM judge, no reward model.

On 200 held-out MuSiQue questions, at a matched 12-turn eval budget — a reproducible
example from the template, **not** a leaderboard claim:

| | EM |
|---|---|
| base `Qwen3.5-35B-A3B` | 54% |
| GRPO-trained (best checkpoint) | **58.5%** |

The interesting part is *how* you get there: `EM = recall × conversion`, and the
diagnostic that decomposes it tells you which lever to pull (turns protect recall; GRPO
improves conversion). Full story, including the levers that **didn't** work and the
honest ceiling, in **[RESULTS.md](RESULTS.md)** and
[`usecases/agentic-search/`](usecases/agentic-search).

## Why it's easy: one engine, thin use cases

The genuinely hard infrastructure — 35B MoE parallelism, the disaggregated
rollout/trainer split, multi-node Ray, the multi-turn agent loop, judge serving — lives
once in [`engine/`](engine). A **new use case is just a few files**, wired in by env var:

| file | engine hook |
|---|---|
| `reward.py` — rule- or judge-based scorer | `CUSTOM_REWARD_PATH` |
| `tool.py` — the agent's tool(s) | `FUNCTION_TOOL_PATH` |
| `prep_data.py` — dataset → parquet | `train_files` / `val_files` |
| `eval.py` — held-out benchmark (reuses `reward.py`) | `EVAL_SCRIPT` |
| `air/*.yaml` — the jobs | — |

Every use case has the **same job types**: `prep → baseline eval → train → eval`
(→ a `deploy` spec). Same shape everywhere. Step-by-step:
[docs/new-usecase.md](docs/new-usecase.md).

## What's here

| path | what |
|---|---|
| [`usecases/agentic-search/`](usecases/agentic-search) ⭐ | the flagship — multi-hop RAG over Vector Search, rule-based reward, no judge |
| [`usecases/math/`](usecases/math) | peer template — MATH-500 + calculator tool + **LLM-judge** reward (the other reward pattern, judge served by the same job) |
| [`infra/`](infra) | platform validation — `diagnostics/` (health probes, run first) + `geo3k/` (the scaling ladder that proves the 35B config runs) |
| [`engine/`](engine) | the shared RL platform (launchers, dispatcher, judge/eval serving, libs) |
| [`docker/`](docker) | the image (the tested version set) |
| [`docs/`](docs) | see the doc map below |
| `scripts/` | host-side build & dev tooling (`make` helpers) |

## Quick start

```bash
# 0. One-time: build + register the image (x86_64 Linux box; everything else runs from a laptop)
make doctor                   # can THIS machine build? (arch / docker / disk / auth)
make image                    # build -> size gate -> push -> register     (docs/build-linux.md)
make volume                   # the Unity Catalog volume (~150 GB)

# 1. Free checks — linters + every job file against the real air CLI. No GPU, no cost.
make check

# 2. Prove the platform, cheapest first
make smoke                    # 1xA10 image pre-flight (~2 min)
air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch   # before ANY agentic job

# 3. Stage the base model once (~70 GB) — every job reads it from the Volume
air run --file infra/air/stage_model.yaml -p df1 --watch

# 4. Run a use case end to end
air run --file usecases/agentic-search/air/1_prep_data.yaml    -p df1 --watch   # data + corpus
air run --file usecases/agentic-search/air/2_build_index.yaml  -p df1 --watch   # Vector Search index
air run --file usecases/agentic-search/air/3_baseline_eval.yaml -p df1 --watch  # the "before" number
air run --file usecases/agentic-search/air/4_train.yaml         -p df1 --watch  # GRPO, 16xH100
air run --file usecases/agentic-search/air/5_eval.yaml          -p df1 --watch  # the "after" number
```

Every job also has a `make` target if you prefer not to remember paths —
`make search-prep search-index search-baseline search-train`,
`make search-eval CKPT=<hf_export_dir>`, and `make math-*` for the other use case.
`make help` lists them all.

Full walkthrough for both use cases, the infra ladder, monitoring and cost:
**[docs/running-jobs.md](docs/running-jobs.md)**.

## Run it in a different training mode

Rollout and training can be **co-located** (synchronous, strictly on-policy) or run on
**disjoint GPU pools** (fully-async, generation overlaps training). That is one env var —
same tool, same reward, same data:

```bash
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch \
  --override env_variables.TRAIN_MODE=sync env_variables.ROLLOUT_NNODES=0 \
             compute.num_accelerators=32
```

The trade-offs, how to size the rollout:trainer split, the weight-sync cadence
arithmetic, an honest support matrix of what has been measured versus wired, and what an
**offline** mode would take: [docs/training-modes.md](docs/training-modes.md).

## Set every setting

Three channels reach a job: `env_variables:` (the knobs), `parameters:` (model, data,
batch shape), and `compute:` (topology). Every one of them — name, default, which script
reads it, what it does — is catalogued in
**[docs/configuration.md](docs/configuration.md)**. The curated "what actually moves the
number, in what order" list is **[docs/tuning.md](docs/tuning.md)**.

Nothing needs an image rebuild: job files snapshot `engine/` and the use case with
`code_source`, so a launcher or reward edit ships with the next submit.

## Docs

| doc | read it when |
|---|---|
| [running-jobs.md](docs/running-jobs.md) | you want to run something — all 26 jobs, in order, with what to check |
| [configuration.md](docs/configuration.md) | you need a setting's exact name, default, or semantics |
| [training-modes.md](docs/training-modes.md) | choosing sync vs fully-async, or sizing the split |
| [tuning.md](docs/tuning.md) | the run works but doesn't learn |
| [new-usecase.md](docs/new-usecase.md) | bringing your own task |
| [setup.md](docs/setup.md) | first time: laptop → first GRPO run |
| [build-linux.md](docs/build-linux.md) | building/registering the image |
| [sizing.md](docs/sizing.md) | why this many GPUs (`python3 docs/sizing.py` reproduces every number) |
| [troubleshooting.md](docs/troubleshooting.md) | something failed |
| [verl-config-reference.md](docs/verl-config-reference.md) | why an individual verl flag is set that way |
| [ladder.md](docs/ladder.md) · [run-log-and-findings.md](docs/run-log-and-findings.md) | what was validated, and the measured numbers |

## The hard model, and the two topologies that run it

92.5% of this model's 34.8 B parameters are routed-expert weights, so expert parallelism
(`EP`) is the dominant lever and plain data parallelism is hopeless. The two modes solve
the memory problem differently, and it is worth being precise about which is which:

| | sync / co-located (the ladder) | fully-async (both use cases) |
|---|---|---|
| sharding | **Megatron-FSDP (ZeRO-3)**, no CPU offload | classic Megatron (ZeRO-1) **+ CPU offload** |
| GPUs | **32×H100** — 16 fits the *persistent* state, but the co-located weight-sync transient OOMs, so 32 is the smallest topology that completes a step | **16×H100** — 8 generate, 8 train, no sharing |
| why | classic ZeRO-1 replicates params+grads across DP, so it cannot reach this at all below 32 GPUs; FSDP shards all three | disjoint pools remove the co-location fight entirely |

That rung-3-vs-rung-4 contrast (same model, same data, same reward, two sharding
strategies) is itself part of the demo — see [`infra/`](infra). The full memory
arithmetic, including why `ROLLOUT_GPU_MEM_UTIL` is *not* the lever for a weight-sync
OOM, is in [docs/sizing.md](docs/sizing.md).

## Stack

The version set is taken verbatim from verl v0.9.0's tested `Dockerfile.stable.vllm` —
a mutually-tested combination; don't bump one alone.

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | df1 = AWS, CUDA 13.0.3 |
| torch | 2.11.0 / cu130 | matches the base's CUDA 13 |
| vllm | 0.24.0 | first with Qwen3.5 rollout support |
| transformers | 5.5.3 | `Qwen3_5MoeForConditionalGeneration` |
| verl | v0.9.0 | fully-async policy + agent loop |
| megatron-core / -bridge | `core_v0.18.0` / 0.5.2 | Megatron-FSDP |

Native wheels come prebuilt from verl's wheelhouse (pinned by URL) — nothing CUDA
compiles at build time. Details, and why df1/AWS not df2/Azure, in
[docs/build-linux.md](docs/build-linux.md).

## Credits

AI Runtime packaging patterns (image size limits, the FIPS/opencv trap, Ray multi-node
teardown, YAML hyperparameters) are adapted from
[hiouchiy/databricks-air-verl-qwen35](https://github.com/hiouchiy/databricks-air-verl-qwen35).
Training configuration follows verl's own `examples/grpo_trainer` recipes.
