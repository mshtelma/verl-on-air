# Troubleshooting

Ordered by when you hit them. Entries marked **[inherited]** are documented
failures from the reference implementation
([hiouchiy/databricks-air-verl-qwen35](https://github.com/hiouchiy/databricks-air-verl-qwen35))
that this repo already works around — listed so you recognise them if a change
reintroduces one. Entries marked **[predicted]** are reasoned-about but not yet
observed on this stack.

## Image build / registration

| symptom | cause | fix |
|---|---|---|
| `air register image` hangs then times out | image >20 GB; the platform cannot replicate it | `make size` before pushing. Confirm `UV_NO_CACHE=1` applied (the uv cache alone is ~11 GB and pushed a comparable image to 31 GB). **[inherited]** |
| `no space left on device` at layer commit | large `COPY` into a layer | never `COPY` a wheelhouse; bind-mount it per-`RUN`. **[inherited]** |
| Job dies at ~1 s, `No module named pip` | AI Runtime's harness imports `pip` and `yaml` before your command runs; a bare base venv has neither | already handled — step 1 of the Dockerfile installs `pip` + `pyyaml`. **[inherited]** |
| `ImportError: undefined symbol: _ZN3c105Error...` | CXX11-ABI mismatch between torch and a CUDA extension | do not mix wheel sources. All native wheels here come from verl's wheelhouse, built against torch 2.11/cu130. Check `torch._C._GLIBCXX_USE_CXX11_ABI`. |
| build installs an unrelated `apex` | PyPI has a package literally named `apex` that is not NVIDIA's | already handled — the Dockerfile pins wheels by **direct URL**, never by index resolution. |
| `Python.h: No such file` | missing dev headers for megatron-core's pybind11 ext | already handled (`python3.12-dev`). **[inherited]** |

## Runtime — process dies immediately

| symptom | cause | fix |
|---|---|---|
| `FATAL FIPS SELFTEST FAILURE`, `Fatal Python error: Aborted` on `import cv2` | `opencv-python-headless` 5.x bundles a FIPS-enforcing `libcrypto`; `transformers` imports cv2 via `mistral_common` | already handled — pinned to `4.12.0.88` as the **last** pip op. `OPENSSL_*` env vars cannot fix it; the blob is vendored. **[inherited]** |
| `ssl.SSLError: [CRYPTO] unknown error (_ssl.c)` | air hosts run a FIPS kernel; non-FIPS crypto fails to init | `OPENSSL_FORCE_FIPS_MODE=0` (set in the image and every YAML). |
| Ray: `expected a valid path like mymodule.provider_class` | `RAY_RUNTIME_ENV_HOOK` set to `""` | never set it to empty. Unset it entirely. **[inherited]** |
| `No module named 'triton'` / GDN kernel compile failure | Triton JITs a host C launcher stub at runtime and needs `cc`/`gcc` | already handled — `build-essential` is deliberately **kept** in the image. The smoke test asserts a compiler is on PATH. |

## Runtime — Megatron / config

| symptom | cause | fix |
|---|---|---|
| `use_megatron_fsdp` appears to do nothing | `vanilla_mbridge=True` — verl only threads it through the Megatron-**Bridge** provider path, legacy mbridge silently ignores it | fsdp mode sets `vanilla_mbridge=False`. Never combine `MEGATRON_MODE=fsdp` with `vanilla_mbridge=True`. |
| FSDP throughput far below expectation, no error | `CUDA_DEVICE_MAX_CONNECTIONS=1` serialises FSDP collectives behind compute | the launcher **unsets** it in fsdp mode and exports `=1` only in classic. If you hand-edit, preserve this. |
| Megatron-FSDP crash mentioning gradient accumulation fusion | incompatible with Megatron-FSDP | `gradient_accumulation_fusion=False` (set in fsdp mode). |
| shape/stride errors in attention, or GDN complaining about packed input | Qwen3.5 GDN has no THD support in Megatron-LM | `use_remove_padding=False` on **both** `model.` and `actor.megatron.`, plus `use_dynamic_bsz=False` everywhere. Non-negotiable. |
| `real_train_batch_size (N) must be divisible by minimal possible batch (M)` | `train_batch_size × rollout_n` not divisible by world GPUs | the launcher pre-checks this and fails fast with the arithmetic. Adjust `train_batch_size` or `rollout_n`. **[inherited]** |
| `'set' object is not subscriptable` during model wrap | verl bug where `model._no_split_modules` resolves to a `set` | FSDP2-path bug from the reference repo; the Megatron path here does not hit it. If you switch to the FSDP2 backend, pass an explicit layer-class list. **[inherited]** |

## Runtime — memory

| symptom | cause | fix |
|---|---|---|
| OOM at optimizer construction, 8 GPUs, 35B | ZeRO-1 replicates params+grads; ~390 GB of Adam state has nowhere to go | this is expected. Use `OFFLOAD=1` (rung 3) or 16 GPUs + fsdp (rung 4). See [sizing.md](sizing.md). |
| Opaque host OOM / job killed during optimizer build with `OFFLOAD=1` | node has <~550 GiB RAM | read `cpu ram` from `make smoke`. If short, rung 4 (no offload) is the only option. |
| OOM only during rollout | vLLM `gpu_memory_utilization` too high alongside resident training state | lower `ROLLOUT_GPU_MEM_UTIL` (0.6 → 0.5). `free_cache_engine=True` is already on so the two peaks do not sum. |
| OOM in log-prob / entropy | vocab is 248320; un-chunked logits are ~3 GB per micro-batch | `entropy_from_logits_with_chunking=True` (already set for actor and ref). |
| `tensor too large to fit in the bucket` during weight sync | a single tensor exceeds the actor→vLLM transfer bucket; embedding is 248320×2048 | set `WEIGHT_BUCKET_MB=6144`. Left unset by default because the config path moved between verl releases — if hydra rejects `rollout.checkpoint_engine.update_weights_bucket_megabytes`, try `rollout.update_weights_bucket_megabytes`. **[inherited]** |

## Multi-node

| symptom | cause | fix |
|---|---|---|
| Job stays `RUNNING` forever after training finishes; billing continues | worker used `ray start --block` and never exits | already handled — workers poll the head's GCS port and `exit 0` when it disappears. Never reintroduce `--block`. **[inherited]** |
| Worker lingers after the head **fails** | cleanup placed after the launch line; `set -e` + `pipefail` skips it | already handled via `trap cleanup EXIT` on the head. **[inherited]** |
| `only N/16 GPUs registered after 15 min` | worker could not reach the head | check `make logs RUN=... NODE=1`. Confirm `MASTER_ADDR`/`MASTER_PORT` are injected (multi-node only — they are absent on single-node). |
| NCCL falls back to sockets; throughput collapses | EFA/`aws-ofi-nccl` did not bind | rung 4 sets `NCCL_DEBUG=INFO`; grep for `NET/OFI`. Do **not** flip the image's `OVERRIDE_NCCL=1` until you have confirmed EFA binds *without* it — swapping NCCL under the base image's `aws-ofi-nccl` 1.15.0 is the classic way to break this. |
| `NET/OFI ... initialization failed` WARN ×3 on A10 jobs | A10/G-family hosts have no EFA hardware; the base image ships the plugin anyway | harmless. Silence with `NCCL_NET_PLUGIN: "none"`. |

## Training signal

| symptom | cause | fix |
|---|---|---|
| Reward is flat; loss ~0; nothing learns | every sample in each group scores identically → advantage 0 → **no gradient** | not a bug. Run `make baseline`: it reports the fraction of groups with non-zero reward variance. Raise `rollout_n`, use harder data, or switch checkpoint. |
| Reward stuck at exactly 0.00 | responses contain no `\boxed{}`. geo3k's accuracy term is gated on `\boxed{}` extraction, so a *correct but unboxed* answer scores 0.00, not 0.90 | ensure the prompt carries the `<think>`/`\boxed{}` instruction (`scripts/prep_geo3k.py` injects it). Verify the surface with `python3 scripts/reward/custom_reward.py`. |
| Reward pinned near 1.0 from step 0 | model already solves the task | saturated — see `make baseline` verdict. Move to `math_dapo`/`aime` or the `-Base` checkpoint. |
| MLflow shows only `score` | you returned a float from a custom reward | return a dict with a `score` key; other keys become separate metrics. |

## Checkpointing (Megatron-FSDP specifics)

Per verl's Megatron-FSDP docs, on this path:

- checkpoints are **DTensor** checkpoints under `dist_ckpt`
- requires `use_distributed_optimizer=True` (verl's default)
- requires `CUDA_DEVICE_MAX_CONNECTIONS` unset or >1 (handled by the launcher)
- **optimizer state cannot be saved alone** — include `model` whenever
  `optimizer` is in `checkpoint.save_contents`
- `checkpoint.async_save=True` is not covered for FSDP DTensor checkpoints
- PEFT + Megatron-FSDP save/load is not covered upstream

The ladder ships with `SAVE_FREQ=-1` (no saving) to keep smoke runs fast. Set
`SAVE_FREQ` before a real run, and expect the first save to be where any of the
above surfaces.
