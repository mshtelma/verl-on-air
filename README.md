# verl-on-air

Train tool-using agents with GRPO on Databricks AI Runtime. The shared
[`engine/`](engine) handles training, rollout, weight synchronization and run
verification; each use case supplies its data, tools, reward and evaluation.

[Get started](docs/running-jobs.md) · [Results](RESULTS.md) ·
[Presentation](docs/showcase.md) · [Add a use case](docs/new-usecase.md)

| Use case | What the agent learns | Training compute | Measured held-out result |
|---|---|---|---|
| [OfficeQA](usecases/officeqa/README.md) | Research tables, calculate, and cite delivered evidence | 16 H100s + hosted judge | Supported pass@1 **8.72% → 20.93%** after 100 updates |
| [MuSiQue retrieval](usecases/agentic-search/README.md) | Chain searches and answer multi-hop questions | 16 H100s; rule reward | Exact match **34.8% → 40.4%** in the run with a predetermined checkpoint |
| [Math](usecases/math/README.md) | Solve problems with a calculator | 24 H100s including a self-hosted Flash judge | Recipe and judge qualification; no published learning gain |

OfficeQA's gain is +12.21 percentage points, with a paired 95% interval of
+4.65 to +20.35. MuSiQue's two runs gained +2.8 and +5.6 points; only the second
interval excludes zero. The tasks use different metrics and evaluation policies.
[RESULTS.md](RESULTS.md) gives the paired measurements, evidence and limitations.

The [OfficeQA report](results/officeqa/2026-10-11-final-report.md) includes
before/after examples and the unresolved AIR discrepancy: the training service
reported `FAILED` after clean node exits, while independent verification
certified the step-100 checkpoint used in the successful final evaluation.

The tested image pins **PyTorch 2.13.0**, **verl 0.10.0.dev0** at `8718ca30`,
and **vLLM 0.30.0**. This is a development snapshot of verl; its qualification
and compatibility changes are recorded in [the stack trial](docs/verl-0.10-trial.md).
The historical MuSiQue results use the earlier stack. An image update is
separate from the evidence for a particular training run.

## The retrieval demo

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
usecases/officeqa/         answer-and-evidence reward/tools    (hosted judge, shared loop)
```

| file | what it does | how the engine finds it |
|---|---|---|
| `reward.py` | scores a trajectory | `CUSTOM_REWARD_PATH` |
| `tool.py` | the agent's tools | `FUNCTION_TOOL_PATH` |
| `prep_data.py` | turns your data into verl parquet | `parameters.train_files` |
| `eval.py` | the held-out number (imports `reward.py`) | `EVAL_SCRIPT` |

The reward determines which infrastructure a task needs. Agentic-search uses a
deterministic rule. Math serves a judge inside its training job. OfficeQA calls
a hosted judge after checking the answer and tracking delivered evidence, and
uses a custom continuous-token loop. The [use-case index](usecases/README.md)
explains the choices; [document research](docs/document-research.md) shows how
to adapt OfficeQA to another collection.

## Layout

```
engine/      the shared platform; you should not need to edit it
  train/     dispatch_agentic.sh (mode and node roles) and the two GRPO launchers
  serve/     serve_judge.sh (LLM judge) and serve_and_eval.sh (any model, any eval.py)
  lib/       parameters, multi-node Ray, preflight checks, checkpoint verification
usecases/    agentic-search/, math/, and officeqa/
infra/       diagnostics/ (probes) and geo3k/ (the scaling ladder)
docker/      the image and its lock files
docs/        guides and reference
scripts/     host-side tooling used by make
```

Each use case runs in the same order: prep, baseline eval, train, eval.
Deployment is [not implemented](docs/deploy.md).

## Point it at your workspace

The job files carry concrete values so each one can be read and submitted by hand. They point at
the author's workspace; set yours in `config.env` and run `make retarget`, which rewrites every
job file (`make lint` fails while any job disagrees with `config.env`).

| what | `config.env` keys |
|---|---|
| Docker image (the one in the files is private; build and register your own) | `DOCKERHUB_USER`, `IMAGE_NAME`, `IMAGE_TAG` |
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

For the measured OfficeQA recipe, [its guide](usecases/officeqa/README.md)
documents prerequisites, cost and these named-experiment commands:

```bash
make officeqa-final-prepare EXPERIMENT=officeqa-my-run AIR_PROFILE=df1
make officeqa-final-run EXPERIMENT=officeqa-my-run BUDGET_OK=1
make officeqa-final-status EXPERIMENT=officeqa-my-run
```

`make help` lists every target. The individual job targets wrap `air run`; the
OfficeQA final controller freezes and validates the complete sequence first.
[docs/running-jobs.md](docs/running-jobs.md) covers inputs, outputs and monitoring.

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

| | sync (co-located, geo3k ladder) | fully-async (tool-agent recipes) |
|---|---|---|
| sharding | Megatron-FSDP (ZeRO-3), no CPU offload | classic Megatron (ZeRO-1) with CPU offload |
| GPUs | 32 | 16 (8 generate, 8 train) |
| limit | 16 GPUs hold the steady state, but the weight sync into the co-located vLLM runs out of memory | the pools are separate, so there is no such peak |

[docs/sizing.md](docs/sizing.md) has the per-GPU byte budget (`python3 docs/sizing.py`
reproduces it), and [`infra/`](infra) has the ladder of runs that validated it.

## The stack

This trial follows verl commit `8718ca30a3f0` (0.10.0.dev), with AIR adaptations listed at the top
of [`docker/Dockerfile`](docker/Dockerfile). The native extensions are built for the exact torch
ABI, so update the stack together. GPU qualification is tracked in
[docs/verl-0.10-trial.md](docs/verl-0.10-trial.md).

| component | version | note |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-runtime-cu13` | CUDA 13.0.3 |
| torch | 2.13.0 / cu130 | native wheels built for torch 2.13 |
| vllm | 0.30.0 | adds GLM-5.3-Flash to the model registry |
| transformers | 5.16.1 | Qwen3.5 and GLM-5.3-Flash configs |
| verl | 0.10.0.dev at `8718ca30a3f0` | experimental trial; immutable commit |
| megatron-core / megatron-bridge | `core_v0.19.2` / 0.6.2 | NVIDIA Bridge for both classic and Megatron-FSDP |
| Ray | 2.55.1 | judge nodes retain their separate 2.48.0 |

TransformerEngine, apex and flash-attn come prebuilt from verl's wheelhouse and are checked
against sha256 hashes in `docker/artifacts.lock`, which keeps the image under AI Runtime's 20 GB
limit. Other packages are pinned in `docker/requirements.lock`, the base image by digest, and
each pushed tag's digest is recorded in `docker/IMAGE.lock`. See
[docs/build-linux.md](docs/build-linux.md).

## Docs

- Running: [running-jobs](docs/running-jobs.md), [setup](docs/setup.md),
  [build-linux](docs/build-linux.md), [troubleshooting](docs/troubleshooting.md)
- Changing: [configuration](docs/configuration.md), [tuning](docs/tuning.md),
  [training-modes](docs/training-modes.md), [new-usecase](docs/new-usecase.md),
  [document-research](docs/document-research.md)
- Presenting: [showcase guide](docs/showcase.md), [held-out comparison](results/heldout-summary.png)
- Background: [RESULTS](RESULTS.md), [sizing](docs/sizing.md), [ladder](docs/ladder.md),
  [verl-config-reference](docs/verl-config-reference.md), [security](docs/security.md)
