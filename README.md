# verl-on-air

GRPO training for tool-using agents on Databricks AI Runtime, set up as a template you can copy.

Most of the work in RL post-training a 35B mixture-of-experts model is infrastructure: expert
parallelism, moving fresh weights from the trainer to the rollout engine, multi-node Ray, a
multi-turn agent loop, and, if the reward is an LLM judge, a second large model inside the same
job. This repo does that once, in [`engine/`](engine). A task plugs in through a few environment
variables. You write a reward, a tool, a data-prep script and an eval, then run GRPO on
`Qwen3.5-35B-A3B` on 16 to 32 H100s with `air run`.

The use cases are worked examples, not benchmark results. [RESULTS.md](RESULTS.md) has the one
measured run and its statistics.

[Run it](docs/running-jobs.md) · [Settings](docs/configuration.md) ·
[New use case](docs/new-usecase.md) · [Results](RESULTS.md)

## The demo

[`usecases/agentic-search`](usecases/agentic-search) trains a multi-hop retrieval agent. It gets
a question and three tools over a Databricks Vector Search index, and has to chain lookups
because no single passage holds the answer. One episode, simplified (the real tool calls are XML
that verl parses with `TOOL_FORMAT=qwen3_coder`):

```
user       Question: Who founded the company that manufactures the Bravia TV line?
assistant  vector_search("Bravia television manufacturer")
tool       "Bravia is a brand of TVs by Sony Corporation..."
assistant  read_article("Sony")
tool       "Sony Group Corporation ... founded in 1946 by ..."
assistant  <answer> Masaru Ibuka and Akio Morita </answer>
```

The reward is exact match against the dataset's gold answers. There is no judge and no reward
model. In one run, the best of 13 checkpoints scored 58.5% on a 200-question development set,
against 54% for the base model. That checkpoint was picked on those same questions, so we also
scored it once on 500 held-out test questions: 37.6% against 34.8%, a gain that is not
statistically significant (p = 0.15). A second run of the same configuration, whose checkpoint
was named before it trained, scored 40.4% on the same test questions (59 gained, 31 lost,
p = 0.004). Every model scores lower on the test set because it mixes 2-, 3- and 4-hop
questions, while the dev set is all 2-hop, and without tools they all score about 5%, so the
score comes from retrieval. Two runs don't yet show how much the gain varies between runs;
details are in [RESULTS.md](RESULTS.md).

## How a use case plugs in

```
engine/                    written once, shared
  dispatcher, sync and fully-async GRPO launchers, judge and eval serving,
  multi-node Ray, MoE parallelism, run certificates
          |
          |  environment variables only
          v
usecases/agentic-search/   exact-match reward, search tools   (6 jobs)
usecases/math/             LLM-judge reward, calculator       (5 jobs)
```

| file | what it does | how the engine finds it |
|---|---|---|
| `reward.py` | scores a trajectory | `CUSTOM_REWARD_PATH` |
| `tool.py` | the agent's tools | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | turns your data into verl parquet | `parameters.train_files` |
| `eval.py` | the held-out number (imports `reward.py`) | `EVAL_SCRIPT` |

The two use cases differ in the reward, because the reward decides how much infrastructure you
need. agentic-search uses a free, deterministic rule. [`math`](usecases/math) serves a GLM-5.3
judge inside its own training job and optimises a graded score. Copy whichever is closer to your
task and follow [docs/new-usecase.md](docs/new-usecase.md).

## Layout

```
engine/      the shared platform; you should not need to edit it
  train/     dispatch_agentic.sh (mode and node roles) and the two GRPO launchers
  serve/     serve_judge.sh (LLM judge) and serve_and_eval.sh (any model, any eval.py)
  lib/       parameters, multi-node Ray, preflight checks, checkpoint verification
usecases/    agentic-search/ and math/
infra/       diagnostics/ (probes) and geo3k/ (the scaling ladder)
docker/      the image and its lock files
docs/        guides and reference
scripts/     host-side tooling used by make
```

There are 26 job files. Each use case runs in the same order: prep, baseline eval, train, eval.
Deployment is [not implemented](docs/deploy.md).

## Point it at your workspace

The job files carry concrete values so each one can be read and submitted by hand. They point at
the author's workspace; set yours in `config.env` and run `make retarget`, which rewrites every
job file (`make lint` fails while any job disagrees with `config.env`).

| what | `config.env` keys |
|---|---|
| Docker image (the one in the files is private; build and register your own) | `DOCKERHUB_USER`, `IMAGE_NAME`, `IMAGE_TAG` (22 jobs use the custom image, 4 use stock environments) |
| Databricks CLI profile | `AIR_PROFILE`; pass `-p <profile>` on any direct `air run` |
| Unity Catalog Volume for models, data and checkpoints | `UC_CATALOG`, `UC_SCHEMA`, `UC_VOLUME` |
| Vector Search endpoint and index (agentic-search) | `VS_ENDPOINT`, `VS_INDEX` |

You need a workspace with AI Runtime serverless GPUs, `GPU_8xH100` capacity, and an x86_64 Linux
machine for the image build. See [docs/setup.md](docs/setup.md) and
[docs/build-linux.md](docs/build-linux.md).

## Run it

Everything except the image build runs from a laptop.

```bash
# one-time: image and storage
make doctor && make image && make volume

# free checks: lint, the CPU tests, every training job composed against the pinned verl,
# and every job file validated by the air CLI
make dev-env && make check

# check the platform before paying for a real job
make smoke                                   # 1xA10, ~2 min

# stage the base model once (~70 GB); the tool-format probe reads its chat template
air run --file infra/air/stage_model.yaml -p <profile> --watch
air run --file infra/diagnostics/air/probe_tool_format.yaml -p <profile> --watch

# the demo
make search-prep                             # questions and passage corpus
make search-index WAREHOUSE_ID=<id>          # Vector Search index; returns before it is ready
make search-baseline                         # the base model's score
make search-train BUDGET_OK=1                # GRPO, fully-async, 16xH100 (at most 160 GPU-h)
make search-eval CKPT=<output_dir>/<RUN_ID>/global_step_20
```

`make help` lists every target, including the `math-*` ones. Each target is an
`air run --file <job.yaml> -p <profile> --watch`, so use the CLI directly when you need
`--override`. [docs/running-jobs.md](docs/running-jobs.md) covers every job, what it produces,
monitoring and cost.

## What you can change without touching the engine

- The task: four files and a few variables ([docs/new-usecase.md](docs/new-usecase.md)).
- How it trains. Rollout and training either share GPUs (synchronous, on-policy) or run on
  separate GPU pools (fully-async). The modes need different node counts, backends and step
  budgets, so each has its own job file: `make search-train` (fully-async, 16 GPUs, the measured
  setup) and `make search-train-sync` (32 GPUs). The sync recipe is experimental: as configured,
  it runs out of memory in the first update. See [docs/training-modes.md](docs/training-modes.md).
- The knobs: `env_variables:`, `parameters:` and `compute:` in the job files.
  [docs/configuration.md](docs/configuration.md) lists all of them, and
  [docs/tuning.md](docs/tuning.md) is the short list that affects results.

None of this needs an image rebuild. Every job uploads `engine/` and its use case as a code
snapshot, so edits ship with the next submit.

## Memory

92.5% of this model's 34.8B parameters are routed-expert weights, so expert parallelism does
most of the work and plain data parallelism does not fit. The two training modes handle memory
differently (the sync column is the geo3k ladder's validated setup):

| | sync (co-located, geo3k ladder) | fully-async (both use cases) |
|---|---|---|
| sharding | Megatron-FSDP (ZeRO-3), no CPU offload | classic Megatron (ZeRO-1) with CPU offload |
| GPUs | 32 | 16 (8 generate, 8 train) |
| limit | 16 GPUs hold the steady state, but the weight sync into the co-located vLLM runs out of memory | the pools are separate, so there is no such peak |

[docs/sizing.md](docs/sizing.md) has the per-GPU byte budget (`python3 docs/sizing.py`
reproduces it), and [`infra/`](infra) has the ladder of runs that validated it.

## The stack

Based on verl v0.9.0's `Dockerfile.stable.vllm`, with a few deviations listed at the top of
[`docker/Dockerfile`](docker/Dockerfile). The versions are tested together, so don't bump one
on its own.

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | CUDA 13.0.3 |
| torch | 2.11.0 / cu130 | |
| vllm | 0.24.0 | verl v0.9.0's tested rollout engine |
| transformers | 5.5.3 | upstream pins 5.3.0, which vLLM 0.24.0 and verl v0.9.0 reject |
| verl | v0.9.0 | fully-async policy and agent loop |
| megatron-core / megatron-bridge | `core_v0.18.0` / 0.5.2 | Megatron-FSDP |

TransformerEngine, apex and flash-attn come prebuilt from verl's wheelhouse and are checked
against sha256 hashes in `docker/artifacts.lock`, which keeps the image under AI Runtime's 20 GB
limit. Other packages are pinned in `docker/requirements.lock`, the base image by digest, and
each pushed tag's digest is recorded in `docker/IMAGE.lock`. See
[docs/build-linux.md](docs/build-linux.md).

## Docs

- Running: [running-jobs](docs/running-jobs.md), [setup](docs/setup.md),
  [build-linux](docs/build-linux.md), [troubleshooting](docs/troubleshooting.md)
- Changing: [configuration](docs/configuration.md), [tuning](docs/tuning.md),
  [training-modes](docs/training-modes.md), [new-usecase](docs/new-usecase.md)
- Background: [RESULTS](RESULTS.md), [sizing](docs/sizing.md), [ladder](docs/ladder.md),
  [verl-config-reference](docs/verl-config-reference.md), [security](docs/security.md)
