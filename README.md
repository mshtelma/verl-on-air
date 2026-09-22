# verl-on-air

**Agentic reinforcement learning on Databricks AI Runtime — a showcase and a template.**

GRPO post-training of `Qwen3.5-35B-A3B` (MoE) with [verl](https://github.com/volcengine/verl)'s
Megatron backend, on AI Runtime serverless GPU. One shared engine does the hard part
once; a use case is a handful of small files on top. The point of this repo is to show
**how little it takes to run real RL on AI Runtime** — and to give you a template to copy.

> These are examples, not benchmark claims. Swap in your own data, reward, or tool and
> the same jobs run unchanged.

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

The genuinely hard infrastructure — 35B MoE on Megatron-FSDP (no offload, 16×H100),
fully-async disaggregated rollout, the multi-turn agent loop, judge serving — lives once
in [`engine/`](engine). A **new use case is just a few files**, wired in by env var:

| file | engine hook |
|---|---|
| `reward.py` — rule- or judge-based scorer | `CUSTOM_REWARD_PATH` |
| `tool.py` — the agent's tool(s) | `FUNCTION_TOOL_PATH` |
| `prep_data.py` — dataset → parquet | `train_files` / `val_files` |
| `eval.py` — held-out benchmark (reuses `reward.py`) | `EVAL_SCRIPT` |
| `air/*.yaml` — the jobs | — |

Every use case has the **same three job types**: `prep → train → eval` (→ a `deploy`
spec). Same shape everywhere.

## What's here

| path | what |
|---|---|
| [`usecases/agentic-search/`](usecases/agentic-search) ⭐ | the flagship — multi-hop RAG, rule-based reward, no judge |
| [`usecases/math/`](usecases/math) | peer template — MATH-500 + calculator tool + **LLM-judge** reward (the other reward pattern) |
| [`infra/`](infra) | platform validation — `diagnostics/` (health probes, run first) + `geo3k/` (the FSDP-vs-classic scaling ladder that proves the 35B config runs) |
| [`engine/`](engine) | the shared RL platform (launchers, judge/eval serving, libs) |
| [`docker/`](docker) | the image (the tested version set) |
| [`docs/`](docs) | [tuning.md](docs/tuning.md) (the knobs that matter) · [training-modes.md](docs/training-modes.md) (sync vs async) · [sizing.md](docs/sizing.md) (why 16-GPU FSDP) · build/troubleshooting |
| `scripts/` | host-side build & dev tooling (Makefile helpers) |

## Quick start

```bash
# Build & register the image (one x86_64 Linux box; laptop is fine for everything else).
make image                    # build -> size gate -> push -> register    (see docs/build-linux.md)
make check                    # lint + validate every air YAML against the real CLI (free)

# Validate the platform, cheapest first.
make smoke                    # 1xA10 image preflight (~2 min)
air run --file infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml -p df1 --watch   # the 35B headline topology

# Stage the base model once, then run a use case end-to-end.
air run --file infra/air/stage_model.yaml -p df1 --watch
#   -> usecases/agentic-search/README.md  (prep -> baseline -> train -> eval)
```

## The hard model, and why it runs here

92.5% of this model's 34.8 B parameters are routed-expert weights. The smallest
**offload-free** topology is **Megatron-FSDP (ZeRO-3) across 16×H100** — classic
Megatron's ZeRO-1 replicates params and grads, so more nodes never fix its OOM below 32
GPUs. That contrast (rung 3 vs rung 4 in [`infra/`](infra)) is itself part of the demo.
The full memory arithmetic is in [docs/sizing.md](docs/sizing.md) (`python3 docs/sizing.py`
reproduces every number).

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
