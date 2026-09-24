# Archive: notes on Qwen3.5-122B-A10B

Not part of the tested template. These come from 122B experiments that were never published here
(the converter and the 122B job files are not in the tree) and have not been re-validated. The
reasoning applies to any model too large to gather on one GPU; re-measure every number.

## Why classic Megatron with CPU offload

Co-located Megatron-FSDP would need about 128 GPUs for 122B ([sizing.md](../sizing.md)). Classic
Megatron with full CPU offload (Adam state in roughly 0.4 to 0.8 TB of host RAM per node) was the
only layout that fit in 32 GPUs or fewer. The async run used 16 trainer GPUs at TP 2, EP 16
(expert all-to-all across both nodes); the sync run 32 GPUs at TP 2, PP 2, EP 16.

## One flag controls saving and loading

`use_dist_checkpointing` (`USE_DIST_CKPT`) switches the save from a gathered HF export, which puts
every weight on one GPU and runs out of memory at 122B, to a sharded Megatron checkpoint. It also
moves the initial load from `model.path` to `DIST_CKPT_PATH`, so a dist checkpoint has to be built
from the HF weights before training. Both launchers set it on the actor and the reference model.

## Building the dist checkpoint

verl's `scripts/converter_hf_to_mcore.py` cannot convert Qwen3.5: the multimodal
`Qwen3_5MoeForConditionalGeneration` takes its text-only branch, which cannot build the vision
tower. The converter used instead followed verl's mbridge load path, so its output is exactly what
`load_mcore_dist_weights()` reads back:

```python
bridge = AutoBridge.from_config(hf_config, dtype)        # verl-patched mbridge
tf_config = bridge.config; tf_config.bf16 = True
module, _ = make_megatron_module(wrap_config, tf_config, hf_config, bridge=bridge, provider=None)
bridge.load_weights(module, hf_path)                     # HF safetensors into the Megatron module
dist_checkpointing.save(unwrap_model(chunks[0]).sharded_state_dict(), out,
                        sharded_strategy=None, async_sharded_save=False)
```

`wrap_config` had `wrap_with_ddp=False` and `use_distributed_optimizer=False`. Converting at TP 1 /
EP 8 on one node and training at TP 2 / EP 16 worked because dist checkpointing reshards on load;
the result was about 245 GB in 8 shards. Two things had to be fixed:

1. `torchrun --standalone` binds to the container hostname, which the node cannot route to. Pass
   `--master_addr=127.0.0.1 --master_port=29500` instead.
2. A UC Volume rejects the parallel writes of a sharded save. Write to `/local_disk0` first, then
   copy to the Volume sequentially.
