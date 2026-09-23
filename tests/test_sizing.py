"""docs/sizing.py (R29): one unit throughout, and the numbers docs/sizing.md quotes.

Before: per_gpu() returned decimal GB (1e9) and was compared with a GiB budget and with measured
GiB peaks ("46.3 GB modeled vs 46.2 GiB measured" is not a sub-GiB agreement), and the shared
expert was counted as expert-parallel although Megatron keeps it a dense TP-sharded MLP.
"""
from __future__ import annotations

import pytest

from support import REPO, load_module, run

sz = load_module(REPO / "docs" / "sizing.py")


def test_the_budget_is_the_measured_usable_hbm_in_bytes():
    assert sz.USABLE_HBM == 81_559 * 2**20
    assert sz.USABLE_HBM / sz.GIB == pytest.approx(79.647, abs=1e-3)
    assert sz.OVERHEAD == 18 * 2**30


def test_parameter_budget():
    p = sz.params()
    assert all(isinstance(v, int) for v in p.values())
    assert sum(p.values()) / 1e9 == pytest.approx(34.816, abs=1e-3)
    assert p["routed experts"] / sum(p.values()) == pytest.approx(0.925, abs=1e-3)


def test_only_the_routed_experts_are_expert_parallel():
    experts, rest = sz.split()
    p = sz.params()
    assert experts == p["routed experts"] and rest == sum(p.values()) - experts
    assert p["shared expert"] > 0     # counted -- in the TP-sharded, EP-replicated part


@pytest.mark.parametrize("mode", ["classic", "fsdp"])
@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_per_gpu_is_bytes_and_its_parts_sum(mode, n):
    d = sz.per_gpu(n, mode)
    assert all(isinstance(v, int) for v in d.values()), "bytes, not GB floats"
    assert d["sum"] == d["params"] + d["grads"] + d["adam"] + d["ref"] + d["vllm"]


def test_scaling_laws_of_the_model():
    tot = sum(sz.params().values())
    # vLLM weights shard by GEN_TP only
    assert {sz.per_gpu(n, m)["vllm"] for n in (8, 16, 32) for m in ("classic", "fsdp")} == {tot * 2 // 8}
    # ZeRO-3 shards everything by N; ZeRO-1 replicates params/grads over DP
    assert sz.per_gpu(32, "fsdp")["params"] * 2 == pytest.approx(sz.per_gpu(16, "fsdp")["params"], abs=1)
    assert sz.per_gpu(32, "classic")["params"] == sz.per_gpu(16, "classic")["params"]
    assert sz.per_gpu(16, "fsdp", grad_bytes=4)["grads"] == 2 * sz.per_gpu(16, "fsdp")["grads"]


def test_the_numbers_the_doc_quotes():
    """docs/sizing.md quotes these (GiB); keep the doc and the script in step."""
    gib = lambda b: round(b / sz.GIB, 1)  # noqa: E731
    assert gib(sz.per_gpu(16, "fsdp")["sum"] + sz.OVERHEAD) == 62.6
    assert gib(sz.per_gpu(32, "fsdp")["sum"] + sz.OVERHEAD) == 44.3
    assert gib(sz.per_gpu(16, "classic")["sum"] + sz.OVERHEAD) == 80.2
    assert gib(sz.per_gpu(16, "classic", adam_bytes=8)["sum"] + sz.OVERHEAD) == 72.1
    assert gib(sz.per_gpu(32, "classic")["sum"] + sz.OVERHEAD) == 68.0


def test_verdicts_compare_like_with_like():
    assert sz.verdict(sz.USABLE_HBM + 1) == "OOM"
    assert sz.verdict(sz.USABLE_HBM) == "TIGHT"
    assert sz.verdict(int(0.5 * sz.USABLE_HBM)) == "OK"
    for n, peak, vllm, outcome in sz.MEASURED_PEAKS:   # the logged outcome is what the budget says
        assert (sz.verdict(int((peak + vllm) * sz.GIB)) == "OOM") == outcome.startswith("OOM")


def test_the_script_runs_and_labels_its_numbers():
    r = run(["python3", str(REPO / "docs" / "sizing.py")])
    assert r.returncode == 0, r.stdout
    for tag in ("[analytic]", "[measured]", "smallest VALIDATED"):
        assert tag in r.stdout
    assert " GB" not in r.stdout.replace("GiB", ""), "a decimal-GB figure is back"
