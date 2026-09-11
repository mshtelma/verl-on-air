# verl-on-air — Run Log & Findings (Qwen3.5 GRPO on df1)

Complete record of the validation-ladder + 122B async-vs-sync perf runs, so details
are recoverable later. Every number here is pulled from `air get run` and the MLflow
metric history, not from memory. Companion docs: `ladder.md` (rung design), `sizing.md`
(memory arithmetic). Last updated 2026-09-11.

## How to recover details

- **Workspace:** `https://dbc-559ffd80-2bfc.cloud.databricks.com` (org `o=2226288096546970`)
- **Auth / CLI:** `air get run <id> -p df1` · `air logs <id> -p df1 [--node N]` · `air list runs --all-status -p df1`. Use the `df1` profile (OAuth); the DEFAULT profile is a dead PAT.
- **MLflow run URL:** `<workspace>/ml/experiments/<exp>/runs/<mlflow_run_id>`
- **Per-step metrics** (throughput, reward, timing, memory) live in MLflow, NOT the driver log — the fully-async trainer logs from a Ray worker actor. Pull with:
  `databricks api get "/api/2.0/mlflow/metrics/get-history?run_id=<mlflow_run_id>&metric_key=perf/throughput" -p df1`
- **Driver console log** (tracebacks, vLLM, NCCL) = `air logs <id> --node 0`; worker node logs = `--node 1/2/...` or the MLflow artifact `logs/node_<n>/logs-0.chunk.txt`.

## MLflow experiments

| Purpose | air experiment | MLflow experiment id |
|---|---|---|
| Stage 122B model | verl-on-air-02b-stage-model-122b | 1342753651768356 |
| Convert 122B → mcore-dist | verl-on-air-02c-convert-122b-mcore-dist | 1342753651769464 |
| 35B fully-async (rung5b) | verl-on-air-31-qwen3_5-35b-fully-async | 1342753651768042 |
| **122B fully-async (air/40)** | verl-on-air-40-qwen3_5-122b-fully-async-metrics | 1342753651768441 |
| **122B classic sync (air/41)** | verl-on-air-41-qwen3_5-122b-classic-sync-metrics | 1342753651770432 |

---

## Master run table

Status is what `air` reports. "Outcome" is the verified reality (some `air` states are
misleading — see notes). `mlflow_run` combines with the experiment id above for the URL.

| # | run_id | what | GPUs | air status | dur | outcome | mlflow_run |
|---|---|---|---|---|---|---|---|
| 1 | 1115109074069926 | stage Qwen3.5-122B-A10B → UC | 1×A10 | SUCCESS | 3956s | 244 GB staged, 39 shards | 7a41fb4ca0ec4fde9ac3a940bccead76 |
| 2 | 261266183131425 | HF→mcore-dist convert (attempt 1) | 8 | FAILED | 413s | `torchrun --standalone` rendezvous on `node.host.local` unroutable (errno 113) | 0af183b70546402d8f7fa8292f1bb5c1 |
| 3 | **403338495784183** | HF→mcore-dist convert (fixed) | 8 | SUCCESS | 1244s | **245.2 GB dist-ckpt, 8 shards, in UC** (loopback rendezvous) | 467d8d76f9e64cbebc3b695eb40dd22e |
| 4 | 818798184847463 | 9B v1 separate_async (2 nodes) | 16 | FAILED | 433s | vLLM OOM — rollout collided onto trainer GPUs (wrong async path) | 7524227e682b48f298f42cfc17d79e8d |
| 5 | 109099216933559 | 9B fully_async (1 node 4+4) | 8 | FAILED | 273s | tied-embedding weight-load fail (`lm_head.weight` absent) → 9B is a dead-end | 6bce4e8ccf0540f9857ed232c6b875c7 |
| 6 | 211079120431601 | 35B fully-async (first) | 16 | FAILED | 593s | `OptimizerParamScheduler` assert — `lr_decay_steps` unset (streaming path) | b18a2c4d9f574eee9f7e95a841e7cede |
| 7 | 1040372797133433 | 35B fully-async (lr fixed) | 16 | FAILED | 3007s | **trained 2 steps OK** (reward 0.11→0.27); cosmetic non-zero teardown | c69235d5347b4b9b88f59b1a2ea86035 |
| 8 | **578317213649089** | 35B fully-async (long, 16 syncs) | 16 | SUCCESS | 5292s | **validated** — reward 0.155→0.492, ~95–101 tok/gpu/s | 8186ed14b1124a008fc6c3cbac340099 |
| 9 | 240241961488066 | 122B async (no ckpt) | 32 | CANCELED | 163s | cancelled to add full-state checkpointing | 4b09d404c04940008f16e120326679e7 |
| 10 | 511710866708108 | 122B async EP=8 | 32 | FAILED | 683s | init OOM — classic replicates params+grads, ~70 GiB/GPU at EP=8 | cba80cd4c8414df388ec82927e00b5b0 |
| 11 | 444713674103804 | 122B async EP=16, HF-save | 24 | FAILED | 6269s | **trained** (→50.9 tok/gpu/s, reward→0.31); OOM at HF-export ckpt save | 7b6508da957b4a32afeabe3ac7213e9b |
| 12 | 1005896590039082 | 122B async EP=16 + dist-ckpt | 24 | FAILED | 5849s | **trained + saved dist-ckpt** (on disk); cosmetic cross-node teardown FAILED | fd42aa450578432e91c7a7754a1050dc |
| 13 | 211655681315147 | 122B async (parallel w/ sync) | 24 | SUCCESS* | 5768s | **FALSE POSITIVE** — died at vLLM custom-all-reduce init, NEVER trained; stale ckpt false-passed the buggy exit-guard | 57a1369b33f944fb8e73c700724e63c1 |
| 14 | **889593001942611** | **122B async FINAL (enforce_eager)** | 24 | SUCCESS | 7922s | **genuine** — trained 3 steps, reward 0.17→0.30, dist-ckpt saved to -v2 | 612843d24aa8483493875560da3fc587 |
| 15 | 328511878149731 | 122B sync PP1 GEN_TP8 util.35 | 32 | FAILED | 1815s | vLLM KV: "No available memory for cache blocks" (weights 30 > 28 budget) | a06e9063a25f439686b256948c232eed |
| 16 | 970563027824989 | 122B sync PP2 GEN_TP16 util.3 | 32 | FAILED | 1243s | vLLM KV: 3.01 GiB needed > 2.21 avail (max_model_len defaulted to 262144) | 0ab4cdb7b6fe4718a7904b4f9f8f21c6 |
| 17 | 510104683616877 | 122B sync (dup submit) | 32 | CANCELED | 72s | accidental duplicate, cancelled | 574e10c056834f489e04b7361a7f9767 |
| 18 | **156278084180640** | **122B sync FINAL** (PP2 GEN_TP16 util.3 max_model_len=8192) | 32 | SUCCESS | 7431s | **genuine** — 6 steps, ~54 tok/gpu/s steady, 2 dist-ckpt saves | 8fa348548fe64079b7ec19ab2b32e985 |

\* run 13's SUCCESS is spurious — see Finding 6.

---

## Per-step metrics (runs that trained)

`throughput` = verl `perf/throughput` (tokens/GPU/s as verl reports it; async normalizes
by its own GPU count). `mem` = `actor/perf/max_memory_reserved_gb` (per trainer GPU).

### 35B-A3B fully-async, 2 nodes 8+8, TP2 EP8 GEN_TP8, classic+offload

- **run 1040372797133433** (short): steps [1,3] · throughput [31.8, 89.6] · reward [0.112, 0.267] · mem 44.1 GiB
- **run 578317213649089** (long, 16 syncs / 31 steps): reward **0.155 → 0.492** (steady climb); throughput ramps to ~101 then settles ~85–95 tok/gpu/s; MFU ~0.8%; mem 44.1 GiB. This is the validated 35B result.

### 122B-A10B (see comparison below for the two finalists)

- **run 444713674103804** (EP=16, HF-save path): steps [1,3,5] · throughput [18.2, 45.2, **50.9**] · reward [0.169, 0.309, 0.274] · mem 76.3 GiB. Died at the checkpoint save, not training.
- **run 1005896590039082** (EP=16 + dist-ckpt): trained and saved `global_step_4/actor/{model/dist_ckpt, optimizer, extra}` (verified on the UC volume) — first end-to-end proof the dist-ckpt SAVE path works. MLflow step metrics were not captured for this run; throughput ≈ run 444's since config is identical.

---

## 122B async-vs-sync comparison (the headline deliverable)

Both on Qwen3.5-122B-A10B, geo3k, classic Megatron + full CPU offload, dist-checkpointing.

| | **ASYNC** (889593001942611) | **SYNC** (156278084180640) |
|---|---|---|
| Topology | 24 GPU: 16 trainer + 8 rollout (disaggregated) | 32 GPU, co-located |
| Parallelism | TP2 PP1 EP16 · GEN_TP8 | TP2 PP2 EP16 · GEN_TP16 |
| vLLM | **enforce_eager** (no CUDA graphs) | CUDA graphs |
| **throughput (steady)** | **~20–24 tok/GPU/s** | **~49–57 tok/GPU/s (mean ~54)** |
| MFU (actor) | ~1.0% | ~1.4% |
| trainer mem reserved/GPU | 75.5 GiB | 38.8 GiB |
| `gen` time / step | 220–350 s | ~49 s |
| reward (few steps) | 0.169 → 0.302 | ~0.24–0.27 |
| checkpoint | dist-ckpt → `…-fully-async-v2/global_step_4` | dist-ckpt → `…-classic/global_step_{4,6}` |

**Interpretation (important):** sync ran ~2.5× faster per GPU here, but that is the cost of
the `enforce_eager` workaround on async — NOT an inherent async penalty. enforce_eager was
required to dodge a vLLM kernel bug (Finding 5) and it inflated async `gen` ~5–7×. Sync
avoided the bug via cross-node GEN_TP=16 (NCCL) and kept CUDA graphs. A fair async number
needs async run at cross-node GEN_TP≥16, or the kernel fixed. **Both paths are proven to
run 122B end-to-end at ≤32 GPU with sharded checkpointing; the throughput gap is an
artifact, not a verdict.**

Sync per-step throughput was [17.6, 48.7, 57.4, 7.8, 55.7, 10.0] — steps 4 & 6 are the
checkpoint-save steps (SAVE_FREQ=4 + final), not compute; steady compute is steps 2/3/5.

---

## Findings (root cause → fix → evidence)

**F1 — Use the fully_async_policy recipe, not v1 separate_async.**
`trainer.v1.trainer_mode=separate_async` with `hybrid_engine=False` places rollout at
`start_rank=0` → collides onto trainer GPUs → vLLM OOM (run 818798184847463). Use
`verl.experimental.fully_async_policy.fully_async_main` with top-level `rollout.nnodes` /
`rollout.n_gpus_per_node` for true disaggregation. → `scripts/run_grpo_fully_async.sh`.

**F2 — Qwen3.5-9B is a dead-end for the classic path (tied embeddings).**
9B ties `lm_head` to input embeddings; legacy mbridge wants a separate `lm_head.weight` and
fails at weight load (run 109099216933559). MoE targets (35B/122B) have untied embeddings →
fine. Small dense models need Megatron-Bridge (FSDP), not the classic path.

**F3 — Fully-async needs an explicit `lr_decay_steps`.**
Streaming mode (`data.train_batch_size=0`) gives verl no dataloader step count, so
Megatron's `OptimizerParamScheduler` asserts `lr_decay_steps > 0` and the trainer dies at
setup (run 211079120431601). Launcher sets `actor.optim.lr_decay_steps = total_rollout_steps`.

**F4 — 122B checkpoint save couples to `use_dist_checkpointing`; VL models block the stock converter.**
verl v0.9.0 (`megatron_checkpoint_manager.py:245`): with `use_dist_checkpointing=False` (default),
the `model` content is exported via a full-gather HF bridge write → **OOMs at 122B** (run
444713674103804, `_save_model_as_hf_via_bridge`, needed 3 GiB with 2.45 free). Setting it
`True` gives a sharded save AND switches INIT to load from `dist_checkpointing_path`. So a
dist-ckpt must be pre-built from HF — but the stock `scripts/converter_hf_to_mcore.py` can't:
Qwen3.5-35B/122B are multimodal (`Qwen3_5MoeForConditionalGeneration`, text_config+vision_config)
and fall into its text-only branch. **Fix:** custom `scripts/convert_hf_to_mcore_dist.py` reusing
verl's vanilla-mbridge build+load path → `dist_checkpointing.save` (run 403338495784183, 245 GB
in UC). Then train with `USE_DIST_CKPT=True` + `DIST_CKPT_PATH` (proven: runs 1005896590039082,
889593001942611, 156278084180640 all saved sharded ckpts). Details: memory `verl-122b-dist-checkpoint`.
Convert single-node with `torchrun --master_addr=127.0.0.1` (NOT `--standalone`: it binds the
TCPStore to the unroutable container hostname → errno 113, run 261266183131425).

**F5 — vLLM custom all-reduce kernel crashes at intra-node GEN_TP≤8 on df1 H100.**
During CUDA-graph capture every rollout worker hits `Failed: Cuda error
custom_all_reduce.cuh:455 'invalid argument'` → workers die → "Engine core initialization
failed" (run 211655681315147, GEN_TP=8; weights load and eager profiling pass first, so it's
the graph-capture path only). **Fixes:** `enforce_eager=True` (skips capture; CONFIRMED by run
889593001942611 — but ~5–7× slower gen), OR cross-node GEN_TP≥16 (NCCL all-reduce, keeps graphs;
sync run 156278084180640 ran clean). `VLLM_ALLREDUCE_USE_SYMM_MEM=0` does NOT prevent it.
Launcher: `ROLLOUT_ENFORCE_EAGER` toggle. Details: memory `vllm-custom-allreduce-h100`.

**F6 — Fully-async exit-guard stale-checkpoint false-positive (FIXED).**
Two layered issues: (a) fully_async normal completion returns non-zero (trainer finishes →
cancels rollouter → `RuntimeError: cancelled`), so the launcher has a guard to treat benign
teardown as success; (b) the guard credited ANY on-disk `global_step_*` — a STALE checkpoint
from a prior run in the same `output_dir` made run 211655681315147 report SUCCESS though it
died at vLLM init and never trained. **Fix in `scripts/run_grpo_fully_async.sh`:** snapshot
pre-existing checkpoints before launch and credit only a NEW one; add
`Engine core initialization failed` / `died unexpectedly` / `Cuda error.*invalid argument`
to the hard-error veto; use a fresh `output_dir` per run. **Lesson: verify SUCCESS against
MLflow step metrics, not the `air` status label.**

**F7 — 122B co-located (sync) vLLM memory is a two-sided squeeze.**
At 32 GPU co-located: PP1+GEN_TP8+util0.35 → vLLM weights (30 GiB) exceed the util budget (28),
zero KV (run 328511878149731). PP2+GEN_TP16+util0.3 fits weights (~15 GiB) but vLLM sized KV
for the model's 262144 config context → 3.01 GiB needed vs 2.21 avail (run 970563027824989).
**Fix:** cap `max_model_len` (we use 1024+2048; set 8192) → run 156278084180640 succeeded.
`util` is NOT the lever for a weight/KV-init failure; PP (trainer footprint), GEN_TP (vLLM
weight shard), and max_model_len (KV size) are.

**F8 — 122B classic parallelism: EP=16 is required at ≤32 GPU.**
Classic Megatron (ZeRO-1) replicates params+grads across DP; only the optimizer shards. At
EP=8 the 122B is ~70 GiB/GPU → init OOM (run 511710866708108). EP=16 halves expert
params+grads to ~41 GiB and fits. dist-ckpt is reshard-aware: convert at TP1/EP8, load at
TP2/EP16.

---

## Config knobs (env vars → launchers)

`scripts/run_grpo_fully_async.sh` (async) and `scripts/run_grpo_megatron.sh` (sync), both read air `env_variables:`:

| knob | async final (889…) | sync final (156…) | note |
|---|---|---|---|
| ROLLOUT_NNODES | 1 | — | async: whole rollout node count |
| TP / PP / CP | 2 / 1 / 1 | 2 / 2 / 1 | PP=2 halves co-located trainer footprint |
| EP / ETP | 16 / 1 | 16 / 1 | EP=16 mandatory ≤32 GPU (F8) |
| GEN_TP | 8 | 16 | 16 = cross-node, dodges F5 kernel bug |
| ROLLOUT_ENFORCE_EAGER | True | (n/a) | async workaround for F5 |
| MAX_MODEL_LEN | 8192 | 8192 | caps vLLM KV (F7) |
| ROLLOUT_GPU_MEM_UTIL | 0.7 | 0.3 | dedicated vs co-located |
| USE_DIST_CKPT / DIST_CKPT_PATH | True / …-mcore-dist | True / …-mcore-dist | F4 |
| SAVE_FREQ | 4 | 4 | sharded dist-ckpt save |
| OFFLOAD_FRACTION | 1 | 1 | Adam in host RAM (~1.6 TB/node) |

## Artifacts on the UC volume (`/Volumes/main/mshtelma/verl/`)

- `models/Qwen3.5-122B-A10B` — HF safetensors (39 shards, ~244 GB), staged by run 1.
- `models/Qwen3.5-122B-A10B-mcore-dist` — Megatron dist-ckpt (~245 GB, 8 shards), from run 3. Init source for `USE_DIST_CKPT=True`.
- `models/Qwen3.5-35B-A3B` — HF safetensors (staged earlier).
- `ckpt/qwen3_5-122b-fully-async/global_step_4` — from run 12 (dist-ckpt async).
- `ckpt/qwen3_5-122b-fully-async-v2/global_step_4` — from run 14 (async FINAL).
- `ckpt/qwen3_5-122b-classic/global_step_{4,6}` — from run 18 (sync FINAL).
</content>
