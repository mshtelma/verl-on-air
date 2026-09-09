#!/usr/bin/env python3
"""Isolate the Transformer Engine multi-tensor crash that kills rung 2.

rung 2 (Qwen3.5-9B, Megatron-FSDP, 8xH100) dies with:

    !!!!!!! Segfault encountered !!!!!!!
      transformer_engine::multi_tensor_scale::multi_tensor_scale_tensor_cuda(...)
      nvte_multi_tensor_scale_tensor_cuda

reached from megatron/core/optimizer/clip_grads.py:clip_grad_by_total_norm_fp32.
Because TE exports multi_tensor_scale_tensor, Megatron keeps total_norm as a GPU
tensor (to avoid a device sync), so clip_coeff is a tensor and the *_tensor
variant is used. rung 1 (2B) went through that same call successfully, so the
trigger is a property of the tensor list, not the call itself.

Rather than bisect on a 15-minute 8xH100 job, probe the primitive directly on a
1xA10. Each case runs in its OWN subprocess so a SIGSEGV is measured rather than
fatal, and both the float and tensor variants are exercised.

Hypotheses under test, in rough order of suspicion:
  mixed_dtypes   TE dispatches on the first tensor's dtype and then reads every
                 tensor as that dtype -> OOB read. Qwen3.5 has GDN layers whose
                 A_log / dt_bias are conventionally fp32 while the rest is bf16.
  zero_numel     Megatron-FSDP reported 31 of 62 buckets EMPTY. A zero-element
                 tensor can yield a degenerate launch config.
  many_tensors   9B has more param tensors than 2B; chunking limits differ.
  noncontiguous  FSDP hands out views into a flat buffer.

Exit status is deliberately NOT a pass/fail gate: this is a measurement.
"""

from __future__ import annotations

import os
import subprocess
import sys

CASES = ("uniform_fp32", "uniform_bf16", "mixed_dtypes", "zero_numel", "many_tensors", "noncontiguous")


def _run_case(name: str) -> int:
    """Child process: build one tensor list and scale it. Returns via exit code."""
    import torch

    from transformer_engine.pytorch.optimizers import (  # noqa: PLC0415
        multi_tensor_applier,
        multi_tensor_scale,
        multi_tensor_scale_tensor,
    )

    dev = "cuda"
    if name == "uniform_fp32":
        grads = [torch.ones(1024, dtype=torch.float32, device=dev) for _ in range(8)]
    elif name == "uniform_bf16":
        grads = [torch.ones(1024, dtype=torch.bfloat16, device=dev) for _ in range(8)]
    elif name == "mixed_dtypes":
        grads = [
            torch.ones(1024, dtype=torch.bfloat16 if i % 2 else torch.float32, device=dev)
            for i in range(8)
        ]
    elif name == "zero_numel":
        grads = [torch.ones(1024, dtype=torch.bfloat16, device=dev) for _ in range(4)]
        grads += [torch.empty(0, dtype=torch.bfloat16, device=dev) for _ in range(4)]
    elif name == "many_tensors":
        grads = [torch.ones(256, dtype=torch.bfloat16, device=dev) for _ in range(4096)]
    elif name == "noncontiguous":
        flat = torch.ones(8 * 2048, dtype=torch.bfloat16, device=dev)
        grads = [flat[i * 2048 : i * 2048 + 1024] for i in range(8)]  # strided views
    else:
        raise SystemExit(f"unknown case {name}")

    dtypes = sorted({str(g.dtype).replace("torch.", "") for g in grads})
    print(f"    tensors={len(grads)} dtypes={','.join(dtypes)} numels={{min:{min(g.numel() for g in grads)}}}")

    buf = torch.zeros(1, dtype=torch.int, device=dev)

    # (a) float coefficient -> multi_tensor_scale (the older, apex-compatible entry)
    multi_tensor_applier(multi_tensor_scale, buf, [grads, grads], 0.5)
    torch.cuda.synchronize()
    print("    float-coeff  multi_tensor_scale        OK")

    # (b) tensor coefficient -> multi_tensor_scale_tensor (what actually crashed)
    coeff = torch.tensor(0.5, dtype=torch.float32, device=dev)
    multi_tensor_applier(multi_tensor_scale_tensor, buf, [grads, grads], coeff)
    torch.cuda.synchronize()
    print("    tensor-coeff multi_tensor_scale_tensor OK")
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--case":
        return _run_case(sys.argv[2])

    import torch

    print("=" * 74)
    print("TE multi-tensor probe (isolating the rung-2 segfault)")
    print("=" * 74)
    try:
        import transformer_engine  # noqa: PLC0415

        te_ver = getattr(transformer_engine, "__version__", "?")
    except Exception as exc:  # pragma: no cover
        te_ver = f"import failed: {exc}"
    print(f"  transformer_engine : {te_ver}")
    print(f"  torch              : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"  device             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
    print()

    results: dict[str, str] = {}
    for case in CASES:
        print(f"  [{case}]")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--case", case],
            capture_output=True,
            text=True,
            timeout=600,
        )
        sys.stdout.write("".join(f"    {ln}\n" for ln in proc.stdout.strip().splitlines() if ln.strip()))
        rc = proc.returncode
        if rc == 0:
            results[case] = "ok"
        elif rc < 0:
            results[case] = f"CRASH signal {-rc}" + (" (SIGSEGV)" if rc == -11 else "")
            tail = proc.stderr.strip().splitlines()[-2:]
            for ln in tail:
                print(f"    stderr: {ln[:150]}")
        else:
            results[case] = f"error rc={rc}"
            tail = proc.stderr.strip().splitlines()[-3:]
            for ln in tail:
                print(f"    stderr: {ln[:150]}")
        print(f"    -> {results[case]}")
        print()

    print("=" * 74)
    print("SUMMARY")
    for case, verdict in results.items():
        mark = "ok  " if verdict == "ok" else "FAIL"
        print(f"  {mark}  {case:16s} {verdict}")
    bad = [c for c, v in results.items() if v != "ok"]
    print()
    if bad:
        print(f"  reproduced the rung-2 trigger on cheap hardware: {', '.join(bad)}")
        print("  -> fix the tensor list (or bypass TE's fused clip), then re-run rung 2")
    else:
        print("  no case reproduces here. The trigger is therefore NOT a plain")
        print("  dtype/shape property of the list -- it is specific to Megatron-FSDP")
        print("  DTensor grads or to H100. Next step: bypass TE in the clip path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
