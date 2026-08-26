#!/usr/bin/env python3
"""Reproduce every number in docs/sizing.md.

    python3 docs/sizing.py

Parameter counts are derived from Qwen/Qwen3.5-35B-A3B's published config.json,
not from the marketing "35B". Pass --fetch to re-read the config from the Hub
and assert our hard-coded shape still matches.
"""

from __future__ import annotations

import argparse

GB = 1e9
GIB = 2**30

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


def params() -> dict[str, float]:
    n_full = L // FULL_INTERVAL
    n_lin = L - n_full
    kdim, vdim = LIN_K * LIN_KH, LIN_V * LIN_VH   # 2048, 4096

    return {
        # 3 matrices per expert (gate, up, down), each H x MI
        "routed experts": L * E * 3 * H * MI,
        "shared expert": L * 3 * H * SHARED_I,
        "router": L * H * E,
        # q, o, k, v, plus attn_output_gate (config: attn_output_gate=true)
        "full-attn": n_full * (H * NQ * HD * 2 + H * NKV * HD * 2 + H * NQ * HD),
        # GDN: q/k projections to kdim, v to vdim, out from vdim
        "GDN linear-attn": n_lin * (H * kdim * 2 + H * vdim + vdim * H),
        "embed + lm_head": 2 * V * H,
        "vision tower": VIS_L * (4 * VIS_H**2 + 2 * VIS_H * VIS_I) + 3 * PATCH**2 * VIS_H,
    }


def per_gpu(n: int, mode: str, adam_bytes: int = 12, gen_tp: int = 8,
            tp: int = 2, ep: int = 8, etp: int = 1, pp: int = 1) -> dict[str, float]:
    """Persistent HBM per GPU in GB, no offload."""
    p = params()
    tot = sum(p.values())
    experts = p["routed experts"] + p["shared expert"]
    non_expert = tot - experts

    if mode == "fsdp":
        # ZeRO-3: params, grads and optimizer all sharded across the full world.
        par = grad = tot * 2 / n
        opt = tot * adam_bytes / n
        ref = tot * 2 / n
    else:
        # ZeRO-1: optimizer sharded over DP, but params/grads REPLICATED over DP.
        dp = n // (tp * pp)
        edp = n // (ep * etp * pp)
        par = (experts / (ep * etp) + non_expert / tp) * 2
        grad = par
        opt = (experts / (ep * etp) / edp + non_expert / tp / dp) * adam_bytes
        ref = par

    vllm = tot * 2 / gen_tp   # sharded by GEN_TP only, NOT by n
    return {"params": par / GB, "grads": grad / GB, "adam": opt / GB,
            "ref": ref / GB, "vllm": vllm / GB,
            "sum": (par + grad + opt + ref + vllm) / GB}


# CUDA ctx 2 + chunked logits/acts 6 + FSDP transient 4 + weight-sync bucket 6
OVERHEAD_GB = 18.0
BUDGET_GB = 79.6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="re-read config.json from the Hub and verify the shape")
    args = ap.parse_args()

    p = params()
    tot = sum(p.values())

    print("=" * 68)
    print("Qwen3.5-35B-A3B parameter budget")
    print("=" * 68)
    for name, val in sorted(p.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {val / 1e9:7.3f} B  ({100 * val / tot:5.1f}%)")
    print(f"  {'TOTAL':22s} {tot / 1e9:7.3f} B")

    print("\n" + "=" * 68)
    print("Unsharded GRPO state (no critic — GRPO uses a group baseline)")
    print("=" * 68)
    rows = [("params bf16", tot * 2), ("grads bf16", tot * 2),
            ("Adam fp32 (m,v,master)", tot * 12), ("ref policy bf16", tot * 2),
            ("vLLM weights bf16", tot * 2)]
    for name, byts in rows:
        print(f"  {name:24s} {byts / GIB:7.1f} GiB")
    print(f"  {'SUM':24s} {sum(b for _, b in rows) / GIB:7.1f} GiB"
          f"   vs 8xH100 = {8 * 79.6:.0f} GiB")

    for adam in (12, 8):
        label = "Adam 12 B/param" if adam == 12 else "Adam 8 B/param (precision-aware)"
        print("\n" + "=" * 68)
        print(f"No-offload per-GPU HBM — {label}, GEN_TP=8")
        print("=" * 68)
        print(f"{'N':>4} {'mode':>8} {'par':>6} {'grad':>6} {'adam':>6} "
              f"{'ref':>6} {'vllm':>6} {'sum':>7} {'+ovh':>7}  verdict")
        for n in (8, 16, 32, 64):
            for mode in ("classic", "fsdp"):
                d = per_gpu(n, mode, adam_bytes=adam)
                total = d["sum"] + OVERHEAD_GB
                verdict = ("OOM" if total > BUDGET_GB
                           else "TIGHT" if total > BUDGET_GB * 0.85 else "OK")
                print(f"{n:>4} {mode:>8} {d['params']:6.1f} {d['grads']:6.1f} "
                      f"{d['adam']:6.1f} {d['ref']:6.1f} {d['vllm']:6.1f} "
                      f"{d['sum']:7.1f} {total:7.1f}  {verdict}")

    print("\n" + "=" * 68)
    print("expert-DP = world / (EP x ETP x PP)   <- decides if FSDP helps at all")
    print("=" * 68)
    for n, ep, etp in ((8, 8, 1), (8, 4, 1), (16, 8, 1), (32, 8, 1)):
        edp = n // (ep * etp)
        note = "FSDP has NO dim to shard experts -> pure overhead" if edp == 1 else "ok"
        print(f"  N={n:<3} EP={ep} ETP={etp} -> expert-DP={edp}   {note}")

    print("\n" + "=" * 68)
    print("Host RAM if OFFLOAD=1 at 8 GPUs (classic, offload_fraction=1)")
    print("=" * 68)
    experts = p["routed experts"] + p["shared expert"]
    non_expert = tot - experts
    exp_adam = experts / 8 * 12 * 8            # per rank x 8 ranks on the node
    ne_adam = non_expert / 2 * 12 / 4 * 8      # TP=2, DP=4
    par_off = (experts / 8 + non_expert / 2) * 2 * 8
    print(f"  expert Adam           {exp_adam / GIB:7.0f} GiB")
    print(f"  non-expert Adam       {ne_adam / GIB:7.0f} GiB")
    print(f"  offloaded actor params{par_off / GIB:7.0f} GiB")
    print(f"  offloaded ref params  {par_off / GIB:7.0f} GiB")
    print(f"  {'TOTAL per node':22s}{(exp_adam + ne_adam + 2 * par_off) / GIB:7.0f} GiB")

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
