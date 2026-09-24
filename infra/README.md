# infra

Checks to run before paying for a real training job: cheap probes first, then a ladder of short
GRPO runs. A failure on one A10 costs a few GPU-minutes; the same failure 40 minutes into a 32-GPU
job costs GPU-hours.

## Probes (`diagnostics/`, mostly 1×A10)

There are two kinds. Gates exit non-zero unless their check passes, and end with one line of JSON:
`PROBE_VERDICT {"probe": ..., "ok": ..., "status": PASS|FAIL|INCONCLUSIVE|ERROR, "reasons": [...]}`
(set `PROBE_VERDICT_OUT=<path>` to also write it to a file). INCONCLUSIVE counts as a failure.
Diagnostics only measure and print; their exit status means nothing.

| job | kind | checks | when to run it |
|---|---|---|---|
| `air/smoke_test.yaml` | gate | image imports, driver floor, an on-device bf16 matmul, CPU RAM, C compiler, and that `AutoBridge` resolves the model (needed for `MEGATRON_MODE=fsdp`; `SMOKE_REQUIRE_BRIDGE=0` downgrades it) | after every image build (`make smoke`) |
| `air/probe_tool_format.yaml` | gate | verl's parser for `PROBE_TOOL_FORMAT` (default `qwen3_coder`) reads the tool call the model's own chat template writes | before any agentic training job |
| `diagnostics/probe_cross_node_http.py` | gate | every pair of nodes can reach each other over HTTP | before a multi-node judge on a new cluster |
| `usecases/agentic-search/probe_vs_access.py` | gate | ANN and HYBRID queries on `QA_VS_INDEX` return rows; read-only | before the first search job |
| `air/diag_cuda.yaml` | diagnostic | CUDA, bf16 on device, driver floor | a node pool looks wrong |
| `air/diag_te.yaml` | diagnostic | TransformerEngine's multi-tensor path | TransformerEngine crashes |
| `air/probe_image_engines.yaml` | diagnostic | which model architectures this image's vLLM can serve | choosing a judge model |
| `air/probe_vllm_multinode.yaml` | diagnostic | serving one model across nodes | before a multi-node judge |
| `air/test_rollout_allreduce.yaml` (8×H100) | diagnostic | vLLM's custom all-reduce failure and the two fixes that keep CUDA graphs | rollout dies at start-up |
| `air/env_probe.yaml` | diagnostic | what the runtime injects (environment, PATH, venv) | something works locally but not in a job |

The tool-format probe catches the most expensive quiet failure in agentic RL: with the wrong
parser, tool calls fail to decode, nothing errors, and you pay for a run whose agent never uses
its tools. The smoke test's `cpu ram` line tells you whether CPU offload is possible at all; it
needs roughly 400–500 GB per node.

## The ladder (`geo3k/`)

Short GRPO runs on the small geo3k dataset that prove each topology works. Each rung changes
several settings at once ([docs/ladder.md](../docs/ladder.md) lists them), so a failure narrows the
problem to that rung's changes.

| job | model | backend | GPUs | offload | purpose | wall clock |
|---|---|---|---|---|---|---|
| `air/rung1_2b_fsdp_8gpu.yaml` | Qwen3.5-2B (dense) | Megatron-FSDP | 8 | no | the full code path, cheapest | ~911 s |
| `air/rung2_9b_fsdp_8gpu.yaml` | Qwen3.5-9B (dense) | Megatron-FSDP | 8 | no | co-located rollout memory | ~1000 s |
| `air/rung3_35b_classic_8gpu.yaml` | 35B-A3B (MoE) | classic Megatron | 8 | yes | verl's own tested configuration | ~1477 s |
| `air/rung4_35b_fsdp_32gpu.yaml` | 35B-A3B (MoE) | Megatron-FSDP | 32 | no | offload-free ZeRO-3 across 4 nodes | ~1443 s |

Rung 4 needs 32 GPUs even though 16 hold its steady-state memory. The weight sync into the
co-located vLLM adds a transient that runs out of memory at 16 ([docs/sizing.md](../docs/sizing.md)).
Rungs 3 and 4 use the same model, data and reward with two sharding strategies. Classic Megatron
replicates parameters and gradients across data-parallel ranks, so adding nodes doesn't fix its
memory; Megatron-FSDP shards them.

The use cases don't use this topology. They run fully-async: the trainer gets its own node with
classic Megatron and CPU offload, and generation runs on a separate node, 16 GPUs in total
([docs/training-modes.md](../docs/training-modes.md)).

Each rung is capped at `total_training_steps: 3`; set it to `0` for a real run. The 64-example
geo3k subset gives 2 steps per epoch, so a rung that prints "2/3" and succeeds is fine.

## The reward-variance gate

`geo3k/air/2_baseline.yaml` samples the base model and reports the fraction of prompt groups whose
rewards differ. A group where every sample scores the same has zero advantage and teaches GRPO
nothing, so this fraction is your effective batch size; `pass@1` does not tell you what it is. Run
it before committing to a dataset. Both use cases changed datasets because of it (HotpotQA to
MuSiQue, GSM8K to MATH).

`geo3k/reward.py` is a small tested reward that returns a dict, useful as a reference for the
reward contract. geo3k's accuracy term needs a `\boxed{}` answer, so a correct but unboxed answer
scores 0.00, not 0.90; `python3 infra/geo3k/reward.py` prints the cases.

## Run

```bash
make smoke                                    # image check
make prep && make stage                       # geo3k data and the base model
make baseline                                 # reward-variance gate
make rung1 && make rung2 && make rung3 && make rung4
air run --file infra/diagnostics/air/probe_tool_format.yaml -p <profile> --watch   # before agentic jobs
```

`infra/air/stage_model.yaml` stages `Qwen3.5-35B-A3B` (~70 GB) to the Volume once; every job reads
it from there.
