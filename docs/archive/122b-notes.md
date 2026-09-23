# Archive: Qwen3.5-122B-A10B notes

← [verl-config-reference](../verl-config-reference.md)

**Historical, not part of the tested template.** These notes come from 122B experiments that
were never published from this repository: the scripts they name (the HF→mcore converter, the
122B job files) are not in the tree, and nothing here was re-validated after the 2026-09
remediation. They are kept because the reasoning -- why classic Megatron with CPU offload, why a
sharded dist checkpoint, how the parallel sizes combine -- transfers to any model too large to
gather on one GPU. Treat every number as a lead to re-measure, not a result.

## Parallel sizes of the two 122B configurations

Worked examples from the two 122B finalists:

- **async:** 16 trainer GPU, TP2 PP1 EP16 → train-DP=8, expert-DP=1. (EP=16 spans both trainer nodes → cross-node expert all-to-all; a *conservative* async throughput — intra-node ETP=2/PP=2 would be fairer.)
- **sync:** 32 GPU, TP2 PP2 EP16 → train-DP=16, expert-DP=2.

## Why classic for 122B

> **Why classic for 122B.** ZeRO-3 (fsdp) 122B co-located would need ~128 GPU
> (`sizing.md`). Classic + **full CPU offload** (Adam in ~0.4–0.8 TB host RAM/node) is
> the only path that fits 122B at ≤32 GPU. Both 122B finalists are classic.

## Distributed checkpointing (`USE_DIST_CKPT`)

The single most important 122B-enabling feature, and a subtle one because **one flag
controls two things**.

`actor_rollout_ref.actor.megatron.use_dist_checkpointing` (default `False`):

| | `False` (default) | `True` |
|---|---|---|
| **checkpoint save** | `model` content exported as a **full-gather HF** file via mbridge (`_save_model_as_hf_via_bridge`) → gathers all weights onto one GPU → **OOMs at 122B** | **sharded** Megatron dist checkpoint, no gather |
| **init weight-load** | load from HF `model.path` | load from `dist_checkpointing_path` |

So to get the sharded save you also flip init onto the dist path — which means you must
**pre-build a dist checkpoint from HF first**. Enabled via env in both launchers:

```
USE_DIST_CKPT=True  DIST_CKPT_PATH=/Volumes/.../Qwen3.5-122B-A10B-mcore-dist
```

which appends `use_dist_checkpointing=True` + `dist_checkpointing_path=…` to **both** the
actor and ref arrays (the ref loads init weights too — §9).

**The bootstrap (a converter script, not published on this branch -- see git history):** the stock
`scripts/converter_hf_to_mcore.py` **cannot** convert Qwen3.5 — it dispatches by
architecture, and multimodal `Qwen3_5MoeForConditionalGeneration` isn't registered, so it
falls into the text-only branch that can't build the vision tower. Our converter instead
reuses verl's **exact vanilla-mbridge init path**:

```python
bridge   = AutoBridge.from_config(hf_config, dtype)   # verl-patched mbridge
tf_config = bridge.config;  tf_config.bf16 = True
module, _ = make_megatron_module(wrap_config, tf_config, hf_config, bridge=bridge, provider=None)
bridge.load_weights(module, hf_path)                  # HF safetensors -> megatron module
dist_checkpointing.save(unwrap_model(chunks[0]).sharded_state_dict(), out,
                        sharded_strategy=None, async_sharded_save=False)
```

This is byte-for-byte what verl's `load_mcore_dist_weights()` reads back at init.
`wrap_config` uses `wrap_with_ddp=False` / `use_distributed_optimizer=False` (pure
conversion — no optimizer, grads, or activations), and `share_embeddings_and_output_weights`
follows the HF `tie_word_embeddings` flag.

**Reshard-aware:** convert at **TP=1/EP=8** (8×H100, one node), train at
**TP=2/EP=16** — dist_checkpointing reshards on load. EP shards the 256 experts so no rank
holds all of them (~40–55 GiB/GPU during convert, comfortable on 80).

**Two operational gotchas in that conversion:**

1. **Rendezvous:** `torchrun --standalone` binds the TCPStore to the container hostname
   (`node.host.local`), unroutable back to itself in the air network (errno 113).
   Force `--master_addr=127.0.0.1 --master_port=29500` (not `--standalone`).
2. **UC write:** the FUSE mount rejects **parallel** range writes (torch_dist writes many
   shards at once). Write to node-local `/local_disk0` (fast NVMe) first, then a
   **sequential** `cp -r` onto the UC Volume.

Result: `…/models/Qwen3.5-122B-A10B-mcore-dist`, ~245 GB / 8 shards.

