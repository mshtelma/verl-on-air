#!/usr/bin/env python3
# =============================================================================
# HF safetensors  ->  Megatron dist_checkpointing tree, for Qwen3.5 VL-MoE.
#
# WHY THIS EXISTS (and not verl's stock scripts/converter_hf_to_mcore.py):
#   Qwen3.5-{35B,122B}-A?B are multimodal (`Qwen3_5MoeForConditionalGeneration`,
#   with text_config + vision_config). The stock converter dispatches by
#   architecture; this arch is not registered, so it falls into the text-only
#   `load_state_dict_to_megatron_gptmodel` branch, which cannot build the vision
#   tower and forbids CPU init. So we cannot use it.
#
#   Instead we reuse the EXACT vanilla-mbridge path verl runs at training init
#   (verl/workers/engine/megatron/transformer_impl.py::_build_tf_config/_build_model):
#       bridge = AutoBridge.from_config(hf_config, dtype)   # verl-patched mbridge
#       tf_config = bridge.config
#       module = make_megatron_module(..., bridge=bridge, provider=None)
#       bridge.load_weights(module, hf_path)                # HF -> megatron module
#   then write the Megatron dist checkpoint with
#       dist_checkpointing.save(unwrap_model(module[0]).sharded_state_dict(), out)
#   which is byte-for-byte what verl's load_mcore_dist_weights() reads back at
#   init when `*.megatron.use_dist_checkpointing=True` + `.dist_checkpointing_path`
#   is set. That flag flips BOTH init-load and checkpoint-save onto the sharded
#   dist path, eliminating the full-gather HF export that OOM'd the 122B save
#   (run 444713674103804: _save_model_as_hf_via_bridge, needed 3 GiB, 2.45 free).
#
# RESHARDING: dist_checkpointing is reshard-aware. We convert at TP=1 (like the
#   stock converter always does) and training loads at TP=2; EP here should be a
#   divisor of the training EP so expert resharding stays clean.
#
# RUN (single node, torchrun; EP shards the 256 experts across ranks so no rank
#   holds all of them):
#       torchrun --standalone --nproc_per_node=8 scripts/convert_hf_to_mcore_dist.py \
#           --hf_model_path <hf> --output_path <out> --tp 1 --pp 1 --ep 8 --trust_remote_code
# =============================================================================
import argparse
import os
import time

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

# Applies verl's mbridge patch on import, then re-exports the patched AutoBridge
# (identical to what the training worker imports).
from verl.models.mcore.mbridge import AutoBridge
from verl.utils.megatron_utils import (
    McoreModuleWrapperConfig,
    make_megatron_module,
    unwrap_model,
)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_path", required=True, help="HF model dir (safetensors)")
    p.add_argument("--output_path", required=True, help="output Megatron dist_checkpointing dir")
    p.add_argument("--tp", type=int, default=1, help="tensor model parallel size (converter default 1)")
    p.add_argument("--pp", type=int, default=1, help="pipeline model parallel size")
    p.add_argument("--ep", type=int, default=1, help="expert model parallel size (must divide num_experts)")
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def log0(rank, *a):
    if rank == 0:
        print("[convert]", *a, flush=True)


def main():
    args = _args()

    # torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT; keep a
    # single-process fallback for local smoke tests.
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Mirror scripts/converter_hf_to_mcore.py: TP is forced small, world = tp*pp*ep.
    assert args.tp * args.pp * args.ep == world, (
        f"tp*pp*ep ({args.tp}*{args.pp}*{args.ep}={args.tp * args.pp * args.ep}) must equal WORLD_SIZE ({world})"
    )
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=args.ep,
    )
    model_parallel_cuda_manual_seed(0)

    hf_config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    log0(rank, "arch:", hf_config.architectures, "| world:", world,
         "| tp/pp/ep:", args.tp, args.pp, args.ep)

    dtype = torch.bfloat16

    # ---- build the megatron module exactly like verl's vanilla-mbridge init ----
    bridge = AutoBridge.from_config(hf_config, dtype=dtype)
    tf_config = bridge.config
    tf_config.fp16 = False
    tf_config.bf16 = True

    tie = getattr(hf_config, "tie_word_embeddings", False)
    wrap_config = McoreModuleWrapperConfig(
        is_value_model=False,
        share_embeddings_and_output_weights=tie,
        wrap_with_ddp=False,          # no DDP: pure conversion, no optimizer/grad buckets
        use_distributed_optimizer=False,
        use_megatron_fsdp=False,
    )
    t0 = time.time()
    module, _ = make_megatron_module(wrap_config, tf_config, hf_config, bridge=bridge, provider=None)
    chunks = module if isinstance(module, list) else [module]
    assert len(chunks) == 1, f"expected 1 pipeline chunk at pp={args.pp}, got {len(chunks)}"
    n_local = sum(p.numel() for p in chunks[0].parameters())
    log0(rank, f"built module in {time.time() - t0:.1f}s | local params on rank0: {n_local / 1e9:.2f}B")

    # ---- HF safetensors -> megatron module (the proven training load path) ----
    t0 = time.time()
    bridge.load_weights(module, args.hf_model_path)
    dist.barrier()
    log0(rank, f"loaded HF weights in {time.time() - t0:.1f}s")

    # ---- save as a Megatron dist checkpoint (what load_mcore_dist_weights reads) ----
    ssd = unwrap_model(chunks[0]).sharded_state_dict()
    log0(rank, f"sharded_state_dict has {len(ssd)} entries; saving to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    t0 = time.time()
    dist_checkpointing.save(ssd, args.output_path, sharded_strategy=None, async_sharded_save=False)
    dist.barrier()
    log0(rank, f"dist_checkpointing.save done in {time.time() - t0:.1f}s")

    if rank == 0:
        entries = sorted(os.listdir(args.output_path))
        total = 0
        for dirpath, _, files in os.walk(args.output_path):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        log0(rank, f"OK output has {len(entries)} top-level entries, {total / 1e9:.1f} GB total")
        log0(rank, "top-level:", entries[:12])

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
