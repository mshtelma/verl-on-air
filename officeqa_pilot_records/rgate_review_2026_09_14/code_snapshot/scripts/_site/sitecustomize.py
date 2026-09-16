#!/usr/bin/env python3
"""Trainer-side whole-group quarantine patch (Gate R), auto-imported by Python.

Activation contract:
  * This DIRECTORY (scripts/_site) is put on PYTHONPATH by run_grpo_fully_async.sh
    ONLY when OQ_REWARD_QUARANTINE=1, so stock runs are completely untouched.
  * When active, we install a meta-path import hook that wraps verl's registered
    GRPO advantage estimator (verl.trainer.ppo.core_algos,
    ADV_ESTIMATOR_REGISTRY["grpo"] + the module attribute) the moment core_algos is
    imported -- BEFORE any trainer code can capture the function reference.

What the wrapper does (see scripts/reward/quarantine.py for the policy):
  after the original estimator returns, every uid-group containing the unknown
  sentinel reward (OQ_UNKNOWN_SENTINEL, emitted by compute_score under the same env
  flag) gets advantages AND returns zeroed -- the group contributes no policy
  gradient instead of poisoning its own baseline with a fabricated 0.0.

Fail-closed: with the env flag set, any failure to locate/patch verl's estimator
raises at import time and kills the process. A silently-unpatched "quarantine" is
worse than no quarantine.

The tensor ops are duck-typed (work on torch.Tensors AND plain lists) so the entire
mechanism is unit-testable on a CPU box without torch -- see
scripts/reward/tests/test_reward_contract.py.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_ENV = "OQ_REWARD_QUARANTINE"
_TARGET = "verl.trainer.ppo.core_algos"


def _load_quarantine():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "reward", "quarantine.py"))
    spec = importlib.util.spec_from_file_location("oq_quarantine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load quarantine module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows_sum(x):
    """Per-sample scalar reward from a [B, T] reward matrix (torch or lists)."""
    if hasattr(x, "sum") and not isinstance(x, list):
        return [float(v) for v in x.sum(dim=-1).detach().float().cpu().tolist()]
    return [float(sum(row)) for row in x]


def _mask_fill_rows(t, rows):
    """Zero whole rows of a [B, T] matrix (torch or lists)."""
    if hasattr(t, "masked_fill"):
        import torch
        sel = torch.tensor(rows, dtype=torch.bool, device=t.device)
        return t.masked_fill(sel.unsqueeze(-1), 0.0)
    return [[0.0] * len(r) if m else list(r) for r, m in zip(t, rows)]


def make_quarantine_wrapper(orig, qmod):
    """Wrap a verl outcome-advantage estimator (grpo signature) with quarantine."""
    def patched(token_level_rewards, response_mask, index, *args, **kwargs):
        adv, ret = orig(token_level_rewards, response_mask, index, *args, **kwargs)
        scores = _rows_sum(token_level_rewards)
        idx = list(index)
        mask = qmod.quarantine_row_mask(scores, idx, qmod.OQ_UNKNOWN_SENTINEL)
        if any(mask):
            adv = _mask_fill_rows(adv, mask)
            ret = _mask_fill_rows(ret, mask)
            n_groups = len({i for i, m in zip(idx, mask) if m})
            print(f"[quarantine] zeroed {n_groups} group(s) / {sum(mask)} sample(s) "
                  f"carrying unknown-verdict sentinel", flush=True)
        return adv, ret
    patched.__name__ = f"quarantined_{getattr(orig, '__name__', 'adv_estimator')}"
    patched.__doc__ = ("GRPO advantage + OfficeQA unknown-verdict whole-group "
                       "quarantine (scripts/_site/sitecustomize.py).\n\n" + str(orig.__doc__))
    return patched


def _patch_core_algos(module, qmod) -> bool:
    """Replace the registered GRPO estimator with its quarantined wrapper. Returns
    True only if the GRPO entry was found and wrapped."""
    patched_any = False
    orig = getattr(module, "compute_grpo_outcome_advantage", None)
    if callable(orig) and not getattr(orig, "__name__", "").startswith("quarantined_"):
        module.compute_grpo_outcome_advantage = make_quarantine_wrapper(orig, qmod)
        patched_any = True
    registry = getattr(module, "ADV_ESTIMATOR_REGISTRY", None)
    if isinstance(registry, dict):
        for name, fn in list(registry.items()):
            if name == "grpo" and callable(fn) and not getattr(fn, "__name__", "").startswith("quarantined_"):
                # Wrap the SAME function object the registry captured (may differ from
                # the module attribute if the trainer imports by symbol).
                registry[name] = make_quarantine_wrapper(fn, qmod)
                patched_any = True
    return patched_any


class _CoreAlgosHook(importlib.abc.MetaPathFinder):
    """Intercept the verl.trainer.ppo.core_algos import and patch it post-exec."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        # Delegate to the remaining finders with ourselves temporarily out of the way
        # (avoids recursion if resolving the spec imports parent packages).
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        hook = self

        def exec_module(module):
            orig_exec(module)
            hook._patch(module)

        spec.loader.exec_module = exec_module
        return spec

    def _patch(self, module):
        qmod = _load_quarantine()
        if not _patch_core_algos(module, qmod):
            raise RuntimeError(
                "OQ_REWARD_QUARANTINE=1 but verl.trainer.ppo.core_algos exposes no "
                "GRPO estimator to wrap -- refusing to train without quarantine.")
        print("[sitecustomize] GRPO unknown-quarantine patch ACTIVE "
              f"(sentinel={qmod.OQ_UNKNOWN_SENTINEL})", flush=True)


def _maybe_install():
    if os.environ.get(_ENV) != "1":
        return
    # Patch an ALREADY-imported core_algos (e.g. sitecustomize ran late), else hook.
    mod = sys.modules.get(_TARGET)
    if mod is not None:
        _CoreAlgosHook()._patch(mod)
        return
    if not any(isinstance(f, _CoreAlgosHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _CoreAlgosHook())


_maybe_install()
