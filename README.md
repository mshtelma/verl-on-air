# verl-on-air

**Agentic reinforcement learning on Databricks AI Runtime — a template you can copy.**

RL post-training on a 35B mixture-of-experts model is mostly *not* an algorithms problem.
It is expert parallelism, a rollout engine handing fresh weights to a trainer, multi-node
Ray, a multi-turn agent loop, and — if your reward is an LLM judge — a second large model
that has to live inside the same job. That is what stops teams before they reach the
interesting question.

**This repo does that part once**, in [`engine/`](engine), use-case-agnostic and wired to
your task through a handful of environment variables. You bring a reward, a tool, a
data-prep and an eval — four small Python files and some YAML — then run GRPO on
`Qwen3.5-35B-A3B` across 16–32 H100s with `air run`.

> These are worked examples, not benchmark claims. There is one measured result
> ([RESULTS.md](RESULTS.md)) whose job is to show the loop really learns; everything else
> here is about *how to do it*.

🚀 [**Run it**](docs/running-jobs.md) · 🎛 [**Every setting**](docs/configuration.md) ·
🧩 [**Bring your own task**](docs/new-usecase.md) · 📊 [**Does it learn?**](RESULTS.md)

---

## What the demo does

[`usecases/agentic-search/`](usecases/agentic-search) trains a **multi-hop retrieval
agent**: given a question and three tools over a **Databricks Vector Search** index, it has
to chain hops, because no single passage holds the answer.

One episode, in shape (illustrative — the real tool calls are XML that verl parses with
`TOOL_FORMAT=qwen3_coder`):

```
user       Question: Who founded the company that manufactures the Bravia TV line?

assistant  vector_search("Bravia television manufacturer")    <- hop 1: find the entity
tool       -> "Bravia is a brand of TVs by Sony Corporation..."
assistant  read_article("Sony")                               <- hop 2: read it for the next fact
tool       -> "Sony Group Corporation ... founded in 1946 by ..."
assistant  <answer> Masaru Ibuka and Akio Morita </answer>    <- commit -- all the reward reads
```

GRPO trains that loop against a **rule-based exact-match reward**: no judge, no reward
model, no human labels. On 200 held-out questions at a matched 12-turn budget, the base
model scores **54%** and the best trained checkpoint **58.5%** — enough to show the loop
learns. The caveats, what *didn't* work, and the `recall × conversion` diagnostic that told
us which knob to turn: **[RESULTS.md](RESULTS.md)**.

## Why it's a template

```
        ┌──────────────────── engine/ (write once, never fork) ─────────────────────┐
        │  dispatcher · sync + fully-async GRPO launchers · judge serving ·         │
        │  eval serving · multi-node Ray · MoE parallelism · run certificates       │
        └───────────────────────────────┬───────────────────────────────────────────┘
                                        │  the seam: env vars only
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
         usecases/agentic-search/          usecases/math/
           reward.py  rule-based EM          reward.py  LLM judge, graded
           tool.py    search + read          tool.py    calculator
           prep_data.py · eval.py            prep_data.py · eval.py
           air/*.yaml   (7 jobs)             air/*.yaml   (5 jobs)
```

| your file | how the engine finds it |
|---|---|
| `reward.py` — scores a trajectory | `CUSTOM_REWARD_PATH` |
| `tool.py` — what the agent can do | `FUNCTION_TOOL_PATH` |
| `prep_data.py` — your data → verl parquet | `parameters.train_files` |
| `eval.py` — the held-out number (imports `reward.py`) | `EVAL_SCRIPT` |

The two use cases differ on the **reward axis**, because that decides how much
infrastructure you need: agentic-search uses a free deterministic rule;
[`math`](usecases/math) serves a **GLM-5.3 judge inside its own training job** and optimises
a graded score. Copy whichever is closer to your task —
**[docs/new-usecase.md](docs/new-usecase.md)**.

## The repo

```
engine/          the shared platform — you should never need to edit this
  train/         dispatch_agentic.sh (mode + node roles) · the two GRPO launchers
  serve/         serve_judge.sh (LLM-as-judge) · serve_and_eval.sh (any model + any eval.py)
  lib/           air parameters → shell · multi-node Ray bring-up/teardown
usecases/        agentic-search/ (rule reward, 7 jobs) · math/ (judge reward, 5 jobs)
infra/           diagnostics/ (8 cheap probes) · geo3k/ (the scaling ladder)
docker/          the image: one tested version set, no CUDA compiled at build time
docs/            see "Where to go next"
scripts/         host-side build tooling (make helpers)
```

27 job files, all the same shape: **prep → baseline eval → train → eval** (→ deploy).

## Make it yours

The job files carry concrete values so any one of them stays readable and
hand-submittable. Four things to point at your own workspace:

| what | where |
|---|---|
| **Docker image** — build and register your own; the one in the files is private | `config.env` → `DOCKERHUB_USER` / `IMAGE_NAME`, then `make bump` rewrites all 26 jobs |
| **Databricks profile** | `config.env` → `AIR_PROFILE` (and `-p <profile>` on any direct `air run`) |
| **Unity Catalog volume** — models, data, checkpoints | `config.env` → `UC_CATALOG`/`UC_SCHEMA`/`UC_VOLUME`, then grep `**/air/*.yaml` for the old path |
| **Vector Search endpoint + index** (agentic-search only) | `QA_VS_ENDPOINT` / `QA_VS_INDEX` in its job files |

You need a workspace with **AI Runtime serverless GPU** enabled, `GPU_8xH100` capacity, and
an x86_64 Linux box for the one-off image build. Details: [docs/setup.md](docs/setup.md) ·
[docs/build-linux.md](docs/build-linux.md).

## Run it

Everything except the Docker build runs from a laptop.

```bash
# 0. one-time: image + storage
make doctor && make image && make volume

# 1. free checks — lint, the CPU regression suite, every training job composed against the
#    pinned verl, and every job file against the real air CLI. No GPU, no cost.
make dev-env && make check

# 2. prove the platform before paying for it
make smoke                                                    # 1xA10, ~2 min
air run --file infra/diagnostics/air/probe_tool_format.yaml -p df1 --watch

# 3. stage the base model once (~70 GB); every job reads it from the Volume
air run --file infra/air/stage_model.yaml -p df1 --watch

# 4. the demo, end to end
make search-prep        # questions + passage corpus
make search-index       # Vector Search index (returns early; wait for status.ready)
make search-baseline    # the "before" number   <- never skip this
make search-train       # GRPO, fully-async, 16xH100
make search-eval CKPT=<…/global_step_20/actor/model/huggingface>
```

`make help` lists every target, including `make math-*`. Each is just an
`air run --file <job.yaml> -p df1 --watch`, so drop to the CLI whenever you want
`--override`. **→ [docs/running-jobs.md](docs/running-jobs.md)** covers every job, what it
produces, monitoring and cost.

## Three things you change without touching the engine

**1 · The task** — four files and a few env vars.
→ [docs/new-usecase.md](docs/new-usecase.md)

**2 · How it trains** — rollout and training either **co-located** (synchronous, strictly
on-policy) or on **disjoint GPU pools** (fully-async, generation overlapping training).
Same tool, same reward, same data — but not the same job: the modes need different node
counts, backends and step budgets, so sync has its own recipe:

```bash
make search-train        # fully-async, 16xH100 -- the measured configuration
make search-train-sync   # synchronous, 32xH100, the same 3200-prompt budget
```

> The sync recipe is **config-validated only** (it composes against the pinned verl, but
> has not run on GPUs). The dispatcher refuses the old one-override switch, which silently
> turned two nodes into an LLM judge the use case does not have.

→ [docs/training-modes.md](docs/training-modes.md) — the trade-off, how to size the
rollout:trainer split (the ratio is *not* learning-neutral), the weight-sync cadence
arithmetic, what has been measured versus merely wired, and what an offline mode would take.

**3 · The knobs** — `env_variables:` (the knobs), `parameters:` (model, data, batch shape),
`compute:` (topology). → [docs/configuration.md](docs/configuration.md) catalogues every
one with its default and which script reads it; → [docs/tuning.md](docs/tuning.md) is the
short list that moves the number, in the order to try it.

None of the three needs an image rebuild: every job snapshots `engine/` plus one use case
via `code_source`, so edits ship with your next submit.

## The part that was hard

92.5% of this model's 34.8 B parameters are routed-expert weights, so expert parallelism is
the dominant lever and plain data parallelism is hopeless. The two training modes solve the
memory problem differently, and the GPU counts differ as a result:

| | sync / co-located (the infra ladder) | fully-async (both use cases) |
|---|---|---|
| sharding | **Megatron-FSDP (ZeRO-3)**, no CPU offload | classic Megatron (ZeRO-1) **+ CPU offload** |
| GPUs | **32×H100** | **16×H100** — 8 generate, 8 train |
| the catch | 16 fits the *persistent* state, but the co-located weight-sync transient OOMs on top of it, so 32 is the smallest topology that completes a step | disjoint pools remove the co-location fight entirely |

Classic ZeRO-1 **replicates** params and grads across data parallel, so it never reaches
this below 32 GPUs however many nodes you add; FSDP shards all three. That contrast — same
model, data and reward, two sharding strategies — is rungs 3 and 4 of the ladder in
[`infra/`](infra). **→ [docs/sizing.md](docs/sizing.md)** has the per-GPU byte budget
(`python3 docs/sizing.py` reproduces every number) and why `ROLLOUT_GPU_MEM_UTIL` is *not*
the lever for a weight-sync OOM.

## The stack

Taken verbatim from verl v0.9.0's tested `Dockerfile.stable.vllm` — a mutually-tested
combination, so don't bump one component alone.

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | AWS workspace, CUDA 13.0.3 |
| torch | 2.11.0 / cu130 | matches the base's CUDA 13 |
| vllm | 0.24.0 | first with Qwen3.5 rollout support |
| transformers | 5.5.3 | `Qwen3_5MoeForConditionalGeneration` |
| verl | v0.9.0 | fully-async policy + agent loop |
| megatron-core / -bridge | `core_v0.18.0` / 0.5.2 | Megatron-FSDP |

Native wheels come prebuilt from verl's wheelhouse, pinned by URL, so nothing CUDA compiles
at build time and the image stays under AI Runtime's 20 GB cap.
**→ [docs/build-linux.md](docs/build-linux.md)**.

## Where to go next

**Run something** — [running-jobs.md](docs/running-jobs.md) (all 27 jobs, in order) ·
[setup.md](docs/setup.md) (from an empty laptop) · [build-linux.md](docs/build-linux.md) ·
[troubleshooting.md](docs/troubleshooting.md)

**Change something** — [configuration.md](docs/configuration.md) (every setting) ·
[tuning.md](docs/tuning.md) (the knobs that matter) ·
[training-modes.md](docs/training-modes.md) · [new-usecase.md](docs/new-usecase.md)

**Understand why it's built this way** — [RESULTS.md](RESULTS.md) (does it learn?) ·
[sizing.md](docs/sizing.md) (memory arithmetic) · [ladder.md](docs/ladder.md) (what was
validated) · [verl-config-reference.md](docs/verl-config-reference.md) (every verl flag)

**Read the code** — [`engine/`](engine) (the platform + its seam) ·
[`usecases/`](usecases) (the two reward patterns) · [`infra/`](infra) (probes + ladder)
