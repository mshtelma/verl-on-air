#!/usr/bin/env python3
"""Reproduce every number in docs/sizing.md.

    python3 docs/sizing.py

Parameter counts are derived from Qwen/Qwen3.5-35B-A3B's published config.json,
not from the marketing "35B". Pass --fetch to re-read the config from the Hub
and assert our hard-coded shape still matches.

UNITS: everything is computed in BYTES and converted only for display, always to
GiB (2**30) -- the unit torch and verl report memory in (verl's `*_gb` fields are
bytes / 1024**3). Each section is tagged with what kind of number it is:
  [analytic]   arithmetic on the config and the assumptions stated next to it
  [measured]   a value logged by a real job, hard-coded here with its source --
               this script does not reproduce it
  [hypothesis] what the analytic model predicts for a configuration nobody ran
"""

from __future__ import annotations

import argparse

GIB = 2**30
MIB = 2**20

# ---- Qwen3.5-35B-A3B shape (config.json, text_config) ----------------------
H = 2048          # hidden_size
L = 40            # num_hidden_layers
E = 256           # num_experts
TOPK = 8          # num_experts_per_tok
MI = 512          # moe_intermediate_size
SHARED_I = 512    # shared_expert_intermediate_size
V = 248320        # vocab_size (tie_word_embeddings=false -> counted twice)
NQ, NKV, HD = 16, 2, 256
FULL_INTERVAL = 4                       # -> L/4 full-attn, rest linear
LIN_K, LIN_KH = 16, 128                 # linear_num_key_heads / key_head_dim
LIN_V, LIN_VH = 32, 128                 # linear_num_value_heads / value_head_dim
VIS_L, VIS_H, VIS_I, PATCH = 27, 1152, 4304, 16

# ---- the GPU --------------------------------------------------------------------
# What torch.cuda.mem_get_info() reports as total on an AI Runtime H100 80GB.
USABLE_HBM = 81_559 * MIB                # = 79.65 GiB

# ---- assumptions of the analytic model (NOT measured) ---------------------------
# bytes per parameter. Adam state = fp32 m + v + master (12); precision-aware
# optimizer keeps m, v in bf16-ish storage (8). Gradients are counted bf16 (2);
# Megatron keeps main grads in fp32 when grads are accumulated/reduced in fp32,
# which would make that 4 -- run with --grad-bytes 4 for that bound.
PARAM_BYTES, GRAD_BYTES, REF_BYTES, VLLM_BYTES = 2, 2, 2, 2
# Steady-state transient allowance, a FIXED guess: CUDA ctx 2 + chunked
# logits/activations 6 + FSDP transient 4 + weight-sync bucket 6 GiB. It does not
# scale with sequence length or micro-batch, and the co-located weight-sync peak
# is larger still (see MEASURED_PEAKS).
OVERHEAD = 18 * GIB

# ---- measured (logged) ----------------------------------------------------------
# Co-located GRPO, rung 4 topology (infra/geo3k/air/rung4_35b_fsdp_16gpu.yaml):
# Megatron-FSDP, TP=1 EP=8 GEN_TP=8, ROLLOUT_GPU_MEM_UTIL=0.25, no offload, geo3k
# 64-example subset, train_batch_size=32, rollout_n=5, prompt 1024 / response 2048,
# image v5, df1, 2026-09-09. train_peak = verl's max_memory_reserved_gb (GiB);
# vllm_awake = vLLM's woken weights + buffers at on_step_end (GiB, from its log).
MEASURED_PEAKS = (
    # (GPUs, train_peak GiB, vLLM awake GiB, outcome)
    (16, 63.4, 17.0, "OOM at the on_step_end weight sync"),
    (32, 46.2, 15.2, "SUCCESS (both steps)"),
)


def params() -> dict[str, int]:
    n_full = L // FULL_INTERVAL
    n_lin = L - n_full
    kdim, vdim = LIN_K * LIN_KH, LIN_V * LIN_VH   # 2048, 4096

    return {
        # 3 matrices per expert (gate, up, down), each H x MI
        "routed experts": L * E * 3 * H * MI,
        # gate/up/down + the per-layer shared_expert_gate (H -> 1)
        "shared expert": L * (3 * H * SHARED_I + H),
        "router": L * H * E,
        # q, o, k, v, plus attn_output_gate (config: attn_output_gate=true)
        "full-attn": n_full * (H * NQ * HD * 2 + H * NKV * HD * 2 + H * NQ * HD),
        # GDN: q/k projections to kdim, v to vdim, out from vdim
        "GDN linear-attn": n_lin * (H * kdim * 2 + H * vdim + vdim * H),
        "embed + lm_head": 2 * V * H,
        "vision tower": VIS_L * (4 * VIS_H**2 + 2 * VIS_H * VIS_I) + 3 * PATCH**2 * VIS_H,
    }


def split(p: dict[str, int] | None = None) -> tuple[int, int]:
    """(expert-parallel params, everything else). Only the ROUTED experts are in the
    expert-parallel group: Megatron's shared expert is a dense TP-sharded MLP
    (`mlp.shared_experts.linear_fc1/fc2`, not under `mlp.experts`), so it is
    replicated across EP ranks like attention is."""
    p = p or params()
    experts = p["routed experts"]
    return experts, sum(p.values()) - experts


def per_gpu(n: int, mode: str, adam_bytes: int = 12, gen_tp: int = 8, tp: int = 2,
            ep: int = 8, etp: int = 1, pp: int = 1, grad_bytes: int = GRAD_BYTES) -> dict[str, int]:
    """[analytic] Persistent HBM per GPU in BYTES, no offload."""
    experts, non_expert = split()
    tot = experts + non_expert

    if mode == "fsdp":
        # ZeRO-3: params, grads and optimizer all sharded across the full world.
        par = tot * PARAM_BYTES // n
        grad = tot * grad_bytes // n
        opt = tot * adam_bytes // n
        ref = tot * REF_BYTES // n
    else:
        # ZeRO-1: optimizer sharded over DP, but params/grads REPLICATED over DP.
        dp = n // (tp * pp)
        edp = n // (ep * etp * pp)
        shard = experts // (ep * etp) + non_expert // tp
        par = shard * PARAM_BYTES
        grad = shard * grad_bytes
        opt = (experts // (ep * etp) // edp + non_expert // tp // dp) * adam_bytes
        ref = shard * REF_BYTES

    vllm = tot * VLLM_BYTES // gen_tp   # sharded by GEN_TP only, NOT by n
    return {"params": par, "grads": grad, "adam": opt, "ref": ref, "vllm": vllm,
            "sum": par + grad + opt + ref + vllm}


def verdict(total_bytes: int, budget: int = USABLE_HBM) -> str:
    return "OOM" if total_bytes > budget else "TIGHT" if total_bytes > 0.85 * budget else "OK"


def gib(b: float) -> float:
    return b / GIB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="re-read config.json from the Hub and verify the shape")
    ap.add_argument("--grad-bytes", type=int, default=GRAD_BYTES,
                    help="bytes per gradient element (2 = bf16, the default; 4 = fp32 main grads)")
    args = ap.parse_args()

    p = params()
    tot = sum(p.values())

    print("=" * 72)
    print("[analytic] Qwen3.5-35B-A3B parameter budget")
    print("=" * 72)
    for name, val in sorted(p.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {val / 1e9:7.3f} B  ({100 * val / tot:5.1f}%)")
    print(f"  {'TOTAL':22s} {tot / 1e9:7.3f} B")

    print("\n" + "=" * 72)
    print("[analytic] Unsharded GRPO state (no critic -- GRPO uses a group baseline)")
    print("=" * 72)
    rows = [("params bf16", tot * PARAM_BYTES), (f"grads ({args.grad_bytes} B)", tot * args.grad_bytes),
            ("Adam fp32 (m,v,master)", tot * 12), ("ref policy bf16", tot * REF_BYTES),
            ("vLLM weights bf16", tot * VLLM_BYTES)]
    for name, byts in rows:
        print(f"  {name:24s} {gib(byts):7.1f} GiB")
    print(f"  {'SUM':24s} {gib(sum(b for _, b in rows)):7.1f} GiB"
          f"   vs 8 x H100 usable = {gib(8 * USABLE_HBM):.0f} GiB")

    for adam in (12, 8):
        label = "Adam 12 B/param" if adam == 12 else "Adam 8 B/param (precision-aware, classic only)"
        print("\n" + "=" * 72)
        print(f"[analytic] No-offload per-GPU HBM, GiB -- {label}, GEN_TP=8")
        print(f"           budget {gib(USABLE_HBM):.2f} GiB; +ovh = sum + {gib(OVERHEAD):.0f} GiB assumed")
        print("=" * 72)
        print(f"{'N':>4} {'mode':>8} {'par':>6} {'grad':>6} {'adam':>6} "
              f"{'ref':>6} {'vllm':>6} {'sum':>7} {'+ovh':>7}  verdict")
        for n in (8, 16, 32, 64):
            for mode in ("classic", "fsdp"):
                d = per_gpu(n, mode, adam_bytes=adam, grad_bytes=args.grad_bytes)
                total = d["sum"] + OVERHEAD
                print(f"{n:>4} {mode:>8} {gib(d['params']):6.1f} {gib(d['grads']):6.1f} "
                      f"{gib(d['adam']):6.1f} {gib(d['ref']):6.1f} {gib(d['vllm']):6.1f} "
                      f"{gib(d['sum']):7.1f} {gib(total):7.1f}  {verdict(total)}")

    print("\n" + "=" * 72)
    print("[measured] Co-located rollout: weight-sync PEAK (logged, not reproduced here)")
    print("=" * 72)
    # The tables above are per-GPU STEADY STATE. Co-located GRPO peaks during the
    # on_step_end actor->vLLM weight sync, which the persistent budget misses:
    #   (a) vLLM is AWAKE holding ~15-17 GiB (weights + buffers), not the ~8 GiB
    #       weight shard the tables count -- util does NOT shrink this (util sizes
    #       only the ['kv_cache'] tag, asleep during the sync);
    #   (b) ZeRO-3 must all-gather each param to a FULL unsharded tensor to export
    #       to HF/vLLM layout -- ~1.9 GiB for the largest MoE expert tensor.
    print(f"{'N':>4} {'train_peak':>11} {'vLLM_awake':>11} {'total':>7}  vs budget  outcome")
    for n, train_peak, vllm_awake, outcome in MEASURED_PEAKS:
        total = (train_peak + vllm_awake) * GIB
        print(f"{n:>4} {train_peak:>11.1f} {vllm_awake:>11.1f} {gib(total):>7.1f}  "
              f"{verdict(total):>9}  {outcome}")
    est = per_gpu(32, "fsdp", grad_bytes=args.grad_bytes)["sum"] + OVERHEAD
    print(f"  32-fsdp analytic persistent estimate {gib(est):.1f} GiB vs measured train_peak "
          f"{MEASURED_PEAKS[1][1]:.1f} GiB: the same unit, but the overhead term is a fixed guess,")
    print("  so the agreement is a plausibility check, not a validation of the model.")
    print("  => smallest VALIDATED offload-free co-located configuration: 32 GPUs (16 measured to OOM;")
    print("     nothing between 16 and 32, and no other sequence budget, was tested).")

    print("\n" + "=" * 72)
    print("[analytic] expert-DP = world / (EP x ETP x PP)   <- decides if FSDP helps at all")
    print("=" * 72)
    for n, ep, etp in ((8, 8, 1), (8, 4, 1), (16, 8, 1), (32, 8, 1)):
        edp = n // (ep * etp)
        note = "FSDP has NO dim to shard experts -> pure overhead" if edp == 1 else "ok"
        print(f"  N={n:<3} EP={ep} ETP={etp} -> expert-DP={edp}   {note}")

    print("\n" + "=" * 72)
    print("[analytic] Host RAM if OFFLOAD=1 at 8 GPUs (classic, TP=2 EP=8, offload_fraction=1)")
    print("=" * 72)
    experts, non_expert = split(p)
    exp_adam = experts // 8 * 12 * 8           # per rank x 8 ranks on the node (expert-DP = 1)
    ne_adam = non_expert // 2 * 12 // 4 * 8    # TP=2, DP=4
    par_off = (experts // 8 + non_expert // 2) * PARAM_BYTES * 8
    print(f"  expert Adam           {gib(exp_adam):7.0f} GiB")
    print(f"  non-expert Adam       {gib(ne_adam):7.0f} GiB")
    print(f"  offloaded actor params{gib(par_off):7.0f} GiB")
    print(f"  offloaded ref params  {gib(par_off):7.0f} GiB")
    print(f"  {'TOTAL per node':22s}{gib(exp_adam + ne_adam + 2 * par_off):7.0f} GiB")

    if args.fetch:
        print("\nverifying against the Hub ...")
        from transformers import AutoConfig
        tc = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B",
                                        trust_remote_code=True).text_config
        checks = {
            "hidden_size": (H, tc.hidden_size),
            "num_hidden_layers": (L, tc.num_hidden_layers),
            "num_experts": (E, tc.num_experts),
            "num_experts_per_tok": (TOPK, tc.num_experts_per_tok),
            "moe_intermediate_size": (MI, tc.moe_intermediate_size),
            "vocab_size": (V, tc.vocab_size),
        }
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        for key, (ours, theirs) in checks.items():
            print(f"  {'ok  ' if ours == theirs else 'DIFF'} {key}: ours={ours} hub={theirs}")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
