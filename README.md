# verl-on-air

**Agentic reinforcement learning on Databricks AI Runtime — a showcase, and a template
you can copy.**

Reinforcement learning on a 35B mixture-of-experts model is mostly *not* an algorithms
problem. It is expert parallelism, a rollout engine that has to hand fresh weights to a
trainer, multi-node Ray, a multi-turn agent loop, and — if your reward is an LLM judge — a
second large model that has to live inside the same job. That is the part that stops
teams before they ever get to the interesting question.

**This repo does that part once.** It sits in [`engine/`](engine), it is use-case-agnostic,
and it is wired to your task through a handful of environment variables. What you bring is
a reward, a tool, a data-prep and an eval — four small Python files and some YAML. Then you
run GRPO on `Qwen3.5-35B-A3B` across 16 or 32 H100s with `air run`.

> **No claims here.** The numbers below are one reproducible example on one dataset, not a
> benchmark result. The point is the machinery, and that swapping in your own data, reward
> or tool needs no engine changes.

**Three ways in:** &nbsp;🚀 [**Run it**](docs/running-jobs.md) &nbsp;·&nbsp;
🎛 [**Configure it**](docs/configuration.md) &nbsp;·&nbsp;
🧩 [**Bring your own task**](docs/new-usecase.md)

---

## The demo: teaching a 35B model to search

The flagship use case, [`usecases/agentic-search/`](usecases/agentic-search), trains a
**multi-hop retrieval agent**. The model is given a question and three tools over a
**Databricks Vector Search** index, and it has to find the answer by chaining hops —
because for a MuSiQue question, no single passage contains it.

An episode has this shape — 12 turns of budget, and `<answer>` is the only thing the reward
reads (illustrative sketch, not a captured transcript; the real tool calls are XML that verl
parses with `TOOL_FORMAT=qwen3_coder`):

```
user       Question: Who founded the company that manufactures the Bravia TV line?

assistant  vector_search("Bravia television manufacturer")    <- hop 1: find the entity
tool       -> "Bravia is a brand of TVs by Sony Corporation..."
assistant  read_article("Sony")                               <- hop 2: read it for the next fact
tool       -> "Sony Group Corporation ... founded in 1946 by ..."
assistant  <answer> Masaru Ibuka and Akio Morita </answer>    <- commit -- all the reward reads
```

GRPO trains that loop with a **rule-based exact-match reward** — no LLM judge, no reward
model, no human labels. The reward function is ~230 lines and the eval harness *imports the
same module*, so what we optimise and what we measure cannot drift apart.

On 200 held-out MuSiQue questions, at a matched 12-turn eval budget for both models:

| | exact match |
|---|---|
| base `Qwen3.5-35B-A3B` | 54% |
| GRPO-trained, best checkpoint | **58.5%** |

The number is not the interesting part — **finding out which lever to pull** is. `EM` factors
into `recall × conversion` (did retrieval surface the gold passage; given that it did, did the
model answer correctly), and a trace diagnostic measures both separately. That is what told
us turns protect recall while GRPO improves conversion — and it is also what let us prove
that a plausible-looking reward tweak did *nothing at all*.

**→ The full case study, including the levers that failed and the honest ceiling:
[RESULTS.md](RESULTS.md)**

## Why it's a template, not a monolith

One engine, a thin seam of environment variables, and use cases that stay small enough to
read in a sitting:

```
        ┌──────────────────── engine/ (write once, never fork) ─────────────────────┐
        │  dispatcher · sync + fully-async GRPO launchers · judge serving ·         │
        │  eval serving · multi-node Ray · MoE parallelism · exit-code guards       │
        └───────────────────────────────┬───────────────────────────────────────────┘
                                        │  the seam: env vars only
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
         usecases/agentic-search/          usecases/math/
           reward.py  rule-based EM          reward.py  LLM judge, graded
           tool.py    search + read          tool.py    calculator
           prep_data.py · eval.py            prep_data.py · eval.py
           air/*.yaml   (6 jobs)             air/*.yaml   (5 jobs)
```

A new use case is these four files plus its jobs — nothing else:

| your file | how the engine finds it |
|---|---|
| `reward.py` — scores a trajectory | `CUSTOM_REWARD_PATH` |
| `tool.py` — what the agent can do | `FUNCTION_TOOL_PATH` |
| `prep_data.py` — your data → verl parquet | `parameters.train_files` |
| `eval.py` — the held-out number (imports `reward.py`) | `EVAL_SCRIPT` |

The two shipped use cases differ deliberately on the **reward axis**, because that is what
decides how much infrastructure you need: agentic-search uses a free deterministic rule,
while [`math`](usecases/math) serves a **GLM-5.3 judge inside its own training job** and
optimises a graded score. Copy whichever is closer to your task.

**→ Step-by-step, with the contracts and a GPU-free checklist:
[docs/new-usecase.md](docs/new-usecase.md)**

## The repo in one screen

```
engine/          the shared platform — you should never need to edit this
  train/         dispatch_agentic.sh (mode + node roles) · the two GRPO launchers
  serve/         serve_judge.sh (LLM-as-judge) · serve_and_eval.sh (any model + any eval.py)
  lib/           air parameters → shell · multi-node Ray bring-up/teardown
usecases/
  agentic-search/  ⭐ the flagship: multi-hop RAG, rule reward, 6 jobs
  math/            the judge-reward pattern: MATH-500 + calculator, 5 jobs
infra/
  diagnostics/   8 cheap probes — run these before spending on GPUs
  geo3k/         the scaling ladder that proves the 35B topology actually trains
docker/          the image: one tested version set, no CUDA compiled at build time
docs/            see "Where to go next"
scripts/         host-side build tooling only (make helpers)
```

26 job files, all the same shape: **prep → baseline eval → train → eval** (→ deploy).

## Run it

Everything except the Docker build runs from a laptop.

```bash
# 0. one-time: image + storage      (x86_64 Linux host needed for the build only)
make doctor && make image && make volume

# 1. free checks — linters, plus every job file against the real air CLI. No GPU, no cost.
make check

# 2. prove the platform before paying for it
make smoke                                                    # 1xA10, ~2 min
air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch

# 3. stage the base model once (~70 GB); every job reads it from the Volume
air run --file infra/air/stage_model.yaml -p df1 --watch

# 4. the demo, end to end
make search-prep        # MuSiQue questions + the passage corpus
make search-index       # Vector Search index (kicks off; wait for ready)
make search-baseline    # the "before" number   <- never skip this
make search-train       # GRPO, fully-async, 16xH100
make search-eval CKPT=/Volumes/.../global_step_20/actor/model/huggingface
```

`make help` lists every target, including `make math-*` for the other use case. Each one is
just an `air run --file <job.yaml> -p df1 --watch`, so you can always drop to the CLI to add
`--override`.

**→ Every job in order, with prerequisites, what each produces, monitoring, capacity and
cost: [docs/running-jobs.md](docs/running-jobs.md)**

## Three things you change without touching the engine

**1 · The task.** Four files and a few env vars, as above.
→ [docs/new-usecase.md](docs/new-usecase.md)

**2 · How it trains.** Rollout and training can be **co-located** (synchronous, strictly
on-policy) or run on **disjoint GPU pools** (fully-async, so generation overlaps training).
That is one environment variable — same tool, same reward, same data:

```bash
air run --file usecases/agentic-search/air/4_train.yaml -p df1 --watch \
  --override env_variables.TRAIN_MODE=sync env_variables.ROLLOUT_NNODES=0 \
             compute.num_accelerators=32
```

→ [docs/training-modes.md](docs/training-modes.md) — the trade-off, how to size the
rollout:trainer split (the ratio is *not* learning-neutral), the weight-sync cadence
arithmetic, an honest matrix of what is measured versus merely wired, and what an
**offline** mode would take.

**3 · The knobs.** Three channels reach a job: `env_variables:` (the knobs),
`parameters:` (model, data, batch shape) and `compute:` (topology).
→ [docs/configuration.md](docs/configuration.md) catalogues every one — name, default,
which script reads it. → [docs/tuning.md](docs/tuning.md) is the short list that actually
moves the number, in the order to try it, with a symptom → knob table.

None of the three needs an image rebuild: every job snapshots `engine/` and one use case
with `code_source`, so an edit ships with your next submit.

## The part that was hard, and is already done

92.5% of this model's 34.8 B parameters are routed-expert weights. Expert parallelism is
therefore the dominant lever, plain data parallelism is hopeless, and the two training
modes solve the memory problem in genuinely different ways — worth being precise about,
because the GPU counts differ:

| | sync / co-located (the infra ladder) | fully-async (both use cases) |
|---|---|---|
| sharding | **Megatron-FSDP (ZeRO-3)**, no CPU offload | classic Megatron (ZeRO-1) **+ CPU offload** |
| GPUs | **32×H100** | **16×H100** — 8 generate, 8 train |
| the catch | 16 GPUs fits the *persistent* state, but the co-located weight-sync transient OOMs on top of it, so 32 is the smallest topology that completes a step | disjoint pools remove the co-location fight entirely |

Classic Megatron's ZeRO-1 **replicates** params and grads across data parallel, so it never
reaches this below 32 GPUs however many nodes you add; FSDP shards all three. That
contrast — same model, same data, same reward, two sharding strategies — is rung 3 versus
rung 4 of the ladder in [`infra/`](infra), and it is part of the demo.

**→ The per-GPU byte budget, and why `ROLLOUT_GPU_MEM_UTIL` is *not* the lever for a
weight-sync OOM: [docs/sizing.md](docs/sizing.md)** (`python3 docs/sizing.py` reproduces
every number) · **what each rung retired: [docs/ladder.md](docs/ladder.md)**

## The stack

Taken verbatim from verl v0.9.0's tested `Dockerfile.stable.vllm` — a mutually-tested
combination, so don't bump one component alone.

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | df1 = AWS, CUDA 13.0.3 |
| torch | 2.11.0 / cu130 | matches the base's CUDA 13 |
| vllm | 0.24.0 | first with Qwen3.5 rollout support |
| transformers | 5.5.3 | `Qwen3_5MoeForConditionalGeneration` |
| verl | v0.9.0 | fully-async policy + agent loop |
| megatron-core / -bridge | `core_v0.18.0` / 0.5.2 | Megatron-FSDP |

Native wheels come prebuilt from verl's wheelhouse, pinned by URL, so nothing CUDA compiles
at build time and the image stays under AI Runtime's 20 GB cap.
**→ [docs/build-linux.md](docs/build-linux.md)** (and why df1/AWS rather than df2/Azure).

## Where to go next

**I want to run something**
[running-jobs.md](docs/running-jobs.md) — all 26 jobs, in order ·
[setup.md](docs/setup.md) — first time, from an empty laptop ·
[build-linux.md](docs/build-linux.md) — the image ·
[troubleshooting.md](docs/troubleshooting.md) — when it breaks

**I want to change something**
[configuration.md](docs/configuration.md) — every setting ·
[tuning.md](docs/tuning.md) — the knobs that matter ·
[training-modes.md](docs/training-modes.md) — sync vs async ·
[new-usecase.md](docs/new-usecase.md) — your own task

**I want to understand why it is built this way**
[RESULTS.md](RESULTS.md) — the case study ·
[sizing.md](docs/sizing.md) — the memory arithmetic ·
[ladder.md](docs/ladder.md) — what was validated ·
[verl-config-reference.md](docs/verl-config-reference.md) — every verl flag, justified ·
[run-log-and-findings.md](docs/run-log-and-findings.md) — the measured record

**Reading the code**
[`engine/`](engine) — the platform and its plugin seam ·
[`usecases/`](usecases) — the two reward patterns compared ·
[`infra/`](infra) — probes and the scaling ladder

## Credits

AI Runtime packaging patterns (image size limits, the FIPS/opencv trap, Ray multi-node
teardown, YAML hyperparameters) are adapted from
[hiouchiy/databricks-air-verl-qwen35](https://github.com/hiouchiy/databricks-air-verl-qwen35).
Training configuration follows verl's own `examples/grpo_trainer` recipes.
