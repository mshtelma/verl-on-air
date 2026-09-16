#!/usr/bin/env python3
"""Isolation contract for the OfficeQA `compute` sandbox (scripts/tools/py_compute.py).

The boundary is now bubblewrap (an OS user/mount/net namespace), NOT an in-process
filter -- a critical review showed every in-process design (AST+builtins, then the
audit-hook child) was escapable. So the contract changed: `open`/`os`/introspection are
NOT "blocked" (that was the theatre the review demolished); instead the kernel guarantees
there is nothing sensitive to reach (network down, data dirs masked, root read-only).

HONESTY: boundary tests require a real sandbox (bwrap + userns). Where it is absent they
are reported as SKIP and counted SEPARATELY -- never as passes (the previous suite counted
self-skipped pandas/numpy cases as passes, inflating "12/12"). Compute-functionality tests
run in either mode. `test_fail_closed_*` proves the safe default when no sandbox exists.

Run standalone (no GPU, no judge):
    PYTHONPATH=scripts python3 scripts/tools/tests/test_py_compute.py
Also importable by pytest (functions named test_*; SkipTest maps to pytest skips).
"""

from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest import SkipTest

os.environ.setdefault("OQ_COMPUTE_TIMEOUT_S", "5")            # keep the timeout test quick
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.abspath(os.path.join(_HERE, ".."))          # scripts/tools
sys.path.insert(0, _TOOLS)

import py_compute  # noqa: E402

_SANDBOXED = py_compute._sandbox_ok()
if not _SANDBOXED:
    # Let the compute-FUNCTIONALITY tests still exercise run_code (bare mode). Boundary
    # tests below explicitly SKIP -- they cannot hold without the kernel namespace.
    os.environ["OQ_COMPUTE_ALLOW_UNSANDBOXED"] = "1"

_HAVE_PANDAS = False
try:
    import pandas  # noqa: F401
    _HAVE_PANDAS = True
except Exception:  # noqa: BLE001
    pass


def _need_sandbox():
    if not _SANDBOXED:
        raise SkipTest("no real sandbox here (bwrap + userns) -- boundary not exercised")


# --- legitimate compute still works (either mode) -----------------------------
def test_sum_works():
    assert "2602" in py_compute.run_code(
        "vals=[132,129,143,159,154,153,177,200,219,287,376,473]\nprint(sum(vals))")


def test_expression_result():
    assert "146.12" in py_compute.run_code("round(100*(6404-2602)/2602, 2)")


def test_pandas_math_works():
    if not _HAVE_PANDAS:
        raise SkipTest("pandas not installed in this environment")
    assert "6" in py_compute.run_code("import pandas as pd\npd.DataFrame({'v':[1,2,3]})['v'].sum()")


# --- the boundary: no network, data dirs masked (require a real sandbox) -------
def test_network_denied():
    _need_sandbox()
    out = py_compute.run_code("import socket\nsocket.create_connection(('1.1.1.1',80),timeout=3)")
    assert out.startswith("Error:") and ("unreachable" in out.lower() or "network" in out.lower()
                                          or "denied" in out.lower() or "errno" in out.lower()), out[:200]


def test_masked_home_not_readable_via_open():
    _need_sandbox()
    sentinel = os.path.expanduser("~/.oq_compute_test_sentinel.secret")
    with open(sentinel, "w") as fh:
        fh.write("TOP-SECRET-GOLD-KEY-42")
    try:
        out = py_compute.run_code(f"print(open({sentinel!r}).read())")
        assert "TOP-SECRET-GOLD-KEY-42" not in out, f"SANDBOX ESCAPE: read a masked host file: {out[:200]}"
        assert out.startswith("Error:"), out[:200]
    finally:
        os.unlink(sentinel)


def test_masked_home_not_readable_via_pandas():
    _need_sandbox()
    if not _HAVE_PANDAS:
        raise SkipTest("pandas not installed in this environment")
    sentinel = os.path.expanduser("~/.oq_compute_test_sentinel.csv")
    with open(sentinel, "w") as fh:
        fh.write("secret\nTOP-SECRET-GOLD-KEY-42\n")
    try:
        out = py_compute.run_code(f"import pandas as pd\nprint(pd.read_csv({sentinel!r}))")
        assert "TOP-SECRET-GOLD-KEY-42" not in out, f"SANDBOX ESCAPE: pandas read a masked host file: {out[:200]}"
    finally:
        os.unlink(sentinel)


def test_introspection_allowed_but_contained():
    """We deliberately do NOT block dunder/operator introspection in-process -- pretending
    to was the escapable theatre. Prove it runs (no fake block) AND that the very escape the
    review used to neutralise the old audit hook still cannot reach the network under bwrap."""
    _need_sandbox()
    out = py_compute.run_code(
        "f = lambda: 0\n"
        "print('globals_ok', len(f.__globals__) >= 0)\n"      # introspection NOT blocked
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1',80),timeout=3); print('NET_LEAK')\n"
        "except OSError:\n"
        "    print('net_denied')\n")
    assert "globals_ok True" in out and "net_denied" in out and "NET_LEAK" not in out, out[:200]


# --- resource control: kill, bounds, isolation (either mode) ------------------
def test_infinite_loop_is_killed():
    out = py_compute.run_code("while True:\n  pass")
    assert out.startswith("Error:") and "timeout" in out.lower(), out[:200]


def test_concurrent_stdout_isolation():
    def one(i):
        return i, py_compute.run_code(f"print('MARKER_{i}_' * 3)")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, range(8)))
    for i, out in results:
        assert f"MARKER_{i}_" in out, f"missing own marker in worker {i}"
        for j in range(8):
            if j != i:
                assert f"MARKER_{j}_" not in out, f"worker {i} leaked worker {j}'s output"


def test_success_output_is_bounded():
    out = py_compute.run_code("print('x' * 200000)")
    assert len(out) <= py_compute._MAX_OUTPUT + 64


def test_error_output_is_bounded():
    # review sub-bug: the error branch bypassed the cap; a 10k-char exception message
    # returned ~10k chars. It must now be clipped like every other path.
    out = py_compute.run_code("raise ValueError('x' * 10000)")
    assert out.startswith("Error:") and len(out) <= py_compute._MAX_OUTPUT + 64, len(out)


def test_mem_mb_env_reaches_child():
    # review sub-bug: OQ_COMPUTE_MEM_MB was set in the parent but dropped from the child
    # env, so RLIMIT_AS was never applied. It must now reach the child.
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "from py_compute import run_code; "
         "print(run_code('import resource; resource.getrlimit(resource.RLIMIT_AS)'))"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "OQ_COMPUTE_MEM_MB": "2048", "PYTHONPATH": _TOOLS,
             "OQ_COMPUTE_ALLOW_UNSANDBOXED": os.environ.get("OQ_COMPUTE_ALLOW_UNSANDBOXED", "")})
    assert "2147483648" in r.stdout, f"RLIMIT_AS not applied from OQ_COMPUTE_MEM_MB: {r.stdout!r} {r.stderr[-300:]!r}"


def test_fail_closed_when_no_sandbox_and_no_optin():
    # The SAFE DEFAULT: with no real sandbox and no explicit opt-in, run NO code.
    saved_ok = py_compute._SANDBOX_OK
    saved_env = os.environ.pop("OQ_COMPUTE_ALLOW_UNSANDBOXED", None)
    try:
        py_compute._SANDBOX_OK = False
        out = py_compute.run_code("print('should not run')")
        assert out.startswith("Error: compute sandbox unavailable"), out[:200]
        assert "should not run" not in out
    finally:
        py_compute._SANDBOX_OK = saved_ok
        if saved_env is not None:
            os.environ["OQ_COMPUTE_ALLOW_UNSANDBOXED"] = saved_env


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, skipped, failed = 0, [], []
    for t in tests:
        try:
            t(); passed += 1; print(f"  PASS  {t.__name__}")
        except SkipTest as e:
            skipped.append(t.__name__); print(f"  SKIP  {t.__name__}: {e}")
        except AssertionError as e:
            failed.append(t.__name__); print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__); print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\nsandbox_available={_SANDBOXED}  ->  {passed} passed, {len(skipped)} skipped"
          + (f", {len(failed)} FAILED: {failed}" if failed else ""))
    if skipped:
        print(f"  skipped (NOT counted as pass): {skipped}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
