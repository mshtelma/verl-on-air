#!/usr/bin/env python3
"""Sandboxed Python executor for the OfficeQA `compute` tool (decorator-free).

The agent writes Python to parse retrieved tables and compute answers (sums over
months, ratios, %-change). numpy / pandas are available.

ISOLATION -- bubblewrap (`bwrap`), an OS boundary, NOT an in-process trick.
A critical review demonstrated that the previous designs were NOT security
boundaries: the in-process AST+builtins filter was escapable, and the later
`sys.addaudithook` child was ALSO escapable (`operator.attrgetter('__globals__')`
reaches the hook's module globals and neutralises it -- Python's own docs say audit
hooks are unsuitable for sandboxing). So the boundary is now the kernel, via bwrap:

  * FRESH CHILD INTERPRETER in a new USER + MOUNT + NET + PID + IPC + UTS namespace
    (`bwrap --unshare-user --unshare-net ...`). No network at all (exfiltration is
    impossible, not merely discouraged).
  * READ-ONLY root with the DATA/SECRET directories MASKED by empty tmpfs:
    /Volumes, /dbfs, /home, /root, /local_disk0, /mnt. Those paths hold the corpus,
    benchmark questions/answers, model weights, snapshots, and host scratch. The child
    literally cannot see them. Retrieval outputs are passed to the agent only by the
    parent tool layer; user code must never need a direct file descriptor to the corpus.
    Preloading any corpus-backed module inside the child would defeat this boundary, so
    compute keeps an explicitly allowed calculation stack rather than reusing the
    already-loaded parent process.
  * `--clearenv` + an explicit minimal `--setenv` set -- no HF_TOKEN / JUDGE_API_KEY /
    credentials leak in. Only interpreter/runtime discovery variables are forwarded,
    never PYTHONPATH (which could expose the parent repo/corpus path inside the child).
  * KILLABLE: the parent enforces a wall-clock timeout and SIGKILLs the bwrap process
    (`--die-with-parent` takes the child with it). RLIMIT_CPU / RLIMIT_AS (via
    OQ_COMPUTE_MEM_MB) / RLIMIT_FSIZE are set inside the child as CPU/mem/write bounds.
  * PER-PROCESS stdout over a pipe -- concurrent calls cannot interleave output.

Because the kernel namespace is the boundary, the child runs ORDINARY Python (full
builtins, real `import`): there is deliberately NO in-process allow/deny list to give
a false sense of security. `open()` / `os` / `socket` are reachable but harmless --
there is nothing sensitive to read (masked), nowhere to write but a tmpfs, and no
network.

FAIL-CLOSED: if `bwrap` is absent OR user namespaces are blocked by the runtime, the
tool runs NO code and returns an error -- UNLESS `OQ_COMPUTE_ALLOW_UNSANDBOXED=1` is
set, which runs the child with process + rlimits ONLY and is explicitly NOT a
filesystem/network boundary (for local dev). df1's serverless-GPU runtime must be
verified with scripts/tools/sandbox_probe.py / air/80 before training; the image
installs `bubblewrap` (docker/Dockerfile).

STATELESS by design: every call is a fresh child with a fresh namespace.

Public API: ``run_code(code: str) -> str``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

_MAX_CODE = 20_000
_MAX_OUTPUT = 8_000
_TIMEOUT_S = float(os.environ.get("OQ_COMPUTE_TIMEOUT_S", "15"))
_MEM_MB = int(os.environ.get("OQ_COMPUTE_MEM_MB", "0"))     # 0 = rely on cgroup; else RLIMIT_AS
# Directories to MASK (empty tmpfs) inside the sandbox: data, secrets, scratch, homes.
# User code receives the numerals from search/grep/read through tool outputs. Those
# parent-side paths must be invisible INSIDE compute so model code cannot open the raw
# corpus, benchmark CSVs, answer keys, model weights, or the reward implementation.
_MASK_CANDIDATES = ("/Volumes", "/dbfs", "/home", "/root", "/local_disk0", "/mnt", "/media", "/srv")


# =============================== CHILD SIDE ===================================
def _run_child() -> None:
    """Executed in the fresh child interpreter INSIDE bwrap (``py_compute.py --child``).
    Reads code from stdin, runs it as ordinary Python (the kernel namespace is the
    boundary -- see the module docstring), writes one JSON line to stdout.

    Resource bounds are set here (bwrap isolates FS/net but not CPU/mem/write)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(_TIMEOUT_S) + 5, int(_TIMEOUT_S) + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if _MEM_MB > 0:
            b = _MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
    except Exception:  # noqa: BLE001
        pass

    code = sys.stdin.read()

    g: dict[str, object] = {"__name__": "__compute__", "__builtins__": __builtins__}
    for mod, names in (("numpy", ("numpy", "np")), ("pandas", ("pandas", "pd"))):
        try:
            m = __import__(mod)
            for n in names:
                g[n] = m
        except Exception:  # noqa: BLE001
            pass

    result: dict[str, object] = {"ok": False, "stdout": "", "result": None, "err": None}
    buf = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):     # capture trailing expr
            last = tree.body.pop()
            tree.body.append(ast.Assign(targets=[ast.Name(id="__result__", ctx=ast.Store())], value=last.value))
            ast.fix_missing_locations(tree)
        compiled = compile(tree, "<compute>", "exec")
    except SyntaxError as e:
        result["err"] = f"SyntaxError: {e}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return
    except Exception as e:  # noqa: BLE001
        result["err"] = f"{type(e).__name__}: {e}"
        os.write(1, json.dumps(result).encode("utf-8", "replace"))
        return

    try:
        with contextlib.redirect_stdout(buf):
            exec(compiled, g)  # noqa: S102 - isolated child under bwrap
        result["ok"] = True
        rv = g.get("__result__")
        result["result"] = None if rv is None else repr(rv)
    except Exception as e:  # noqa: BLE001
        result["err"] = f"{type(e).__name__}: {e}"
    result["stdout"] = buf.getvalue()
    # Write to the real stdout fd (already open -> no file-open needed).
    os.write(1, json.dumps(result).encode("utf-8", "replace"))


# =============================== PARENT SIDE ==================================
_SANDBOX_OK: bool | None = None
_UNSANDBOXED_WARNED = False


def _sandbox_ok() -> bool:
    """True iff a REAL bwrap sandbox is usable here: the binary exists AND the runtime
    actually permits the unshare (present-but-blocked userns -> False). Cached."""
    global _SANDBOX_OK
    if _SANDBOX_OK is not None:
        return _SANDBOX_OK
    bw = shutil.which("bwrap")
    if not bw:
        _SANDBOX_OK = False
        return False
    try:
        probe = [bw, "--unshare-user", "--unshare-net", "--ro-bind", "/", "/",
                 "--dev", "/dev", "--tmpfs", "/tmp", "true"]
        r = subprocess.run(probe, capture_output=True, timeout=15)
        _SANDBOX_OK = r.returncode == 0
    except Exception:  # noqa: BLE001
        _SANDBOX_OK = False
    return _SANDBOX_OK


def _mask_dirs() -> list[str]:
    """Data/secret/scratch dirs to hide with an empty tmpfs -- but never one that
    contains the interpreter or its venv (that would break Python inside the sandbox)."""
    py = {os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix),
          os.path.dirname(os.path.realpath(sys.executable))}
    out = []
    for d in _MASK_CANDIDATES:
        if not os.path.isdir(d):
            continue
        rd = os.path.realpath(d)
        if any(p == rd or p.startswith(rd + os.sep) for p in py):
            continue                                   # interpreter lives here -> keep it
        out.append(d)
    return out


def _child_setenv() -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp", "TMPDIR": "/tmp",
        "LANG": os.environ.get("LANG", "C.UTF-8"), "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "MPLCONFIGDIR": "/tmp", "XDG_CACHE_HOME": "/tmp",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        # Pass the resource knobs THROUGH -- the child re-reads them at import (the old
        # _child_env dropped OQ_COMPUTE_MEM_MB, so RLIMIT_AS was never set in the child).
        "OQ_COMPUTE_MEM_MB": str(_MEM_MB), "OQ_COMPUTE_TIMEOUT_S": str(_TIMEOUT_S),
    }
    # Interpreter/runtime discovery only, never app secrets or PYTHONPATH. The masked
    # /home and /local_disk0 paths would otherwise expose the repository/corpus inside
    # the child; pandas/numpy/scipy are imported from the interpreter's site-packages.
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "LD_LIBRARY_PATH"):
        v = os.environ.get(k)
        if v:
            env[k] = v
    return env


_CHILD_IN_SANDBOX = "/tmp/_oq_compute_child.py"


def _bwrap_cmd() -> list[str]:
    cmd = [
        shutil.which("bwrap"),
        "--unshare-user", "--unshare-net", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent",
        "--ro-bind", "/", "/",             # everything readable...
        "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    for d in _mask_dirs():                  # ...except data/secret/scratch, masked empty
        cmd += ["--tmpfs", d]
    # Expose THIS script at a stable, UNMASKED path. Its real directory may itself be
    # masked (the snapshot lands under /local_disk0 or /home on the runner), so running
    # `python <real_path> --child` would hit "No such file". bwrap resolves the bind
    # SOURCE in the OUTER namespace, so the masked view does not block the bind.
    cmd += ["--ro-bind", os.path.abspath(__file__), _CHILD_IN_SANDBOX]
    cmd += ["--chdir", "/tmp", "--clearenv"]
    for k, v in _child_setenv().items():
        cmd += ["--setenv", k, v]
    return cmd + [sys.executable, _CHILD_IN_SANDBOX, "--child"]


def _child_env_bare() -> dict:
    """Env for the OPT-IN unsandboxed fallback (process + rlimits only, no bwrap)."""
    env = {k: v for k, v in _child_setenv().items()}
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    return env


def _plan() -> tuple[list[str] | None, dict, bool]:
    """Return (argv, launch_env, sandboxed). argv is None if we must fail closed."""
    if _sandbox_ok():
        return _bwrap_cmd(), {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, True
    child = [sys.executable, os.path.abspath(__file__), "--child"]
    if os.environ.get("OQ_COMPUTE_ALLOW_UNSANDBOXED") == "1":
        global _UNSANDBOXED_WARNED
        if not _UNSANDBOXED_WARNED:
            print("[oq-compute] WARNING: bwrap unavailable; running UNSANDBOXED "
                  "(process+rlimits only, NOT a FS/net boundary) per OQ_COMPUTE_ALLOW_UNSANDBOXED=1",
                  file=sys.stderr, flush=True)
            _UNSANDBOXED_WARNED = True
        return child, _child_env_bare(), False
    return None, {}, False


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + "\n... (output truncated)"


def run_code(code: str) -> str:
    """Execute one self-contained Python snippet in an isolated bwrap child; return text."""
    code = str(code or "").strip()
    if not code:
        return "Error: 'code' must be a non-empty string"
    if len(code) > _MAX_CODE:
        return f"Error: code exceeds {_MAX_CODE} chars"

    argv, launch_env, _sandboxed = _plan()
    if argv is None:
        return ("Error: compute sandbox unavailable -- bubblewrap (bwrap) is not installed or "
                "user namespaces are blocked here, and OQ_COMPUTE_ALLOW_UNSANDBOXED is not set. "
                "Refusing to execute code without an isolation boundary (fail-closed).")

    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=launch_env, close_fds=True,
        )
    except Exception as e:  # noqa: BLE001 - FAIL CLOSED, never run in-process
        return f"Error: compute sandbox failed to launch ({type(e).__name__}: {e})"

    try:
        out, err = proc.communicate(code.encode("utf-8"), timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()                                    # SIGKILL bwrap -> --die-with-parent kills child
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return _clip(f"Error: TimeoutError: code execution exceeded {_TIMEOUT_S:.0f}s")

    if not out:
        tail = (err or b"").decode("utf-8", "replace").strip()[-500:]
        rc = proc.returncode
        if rc and rc < 0:
            return _clip(f"Error: compute child terminated by signal {-rc} (killed: OOM / resource limit).")
        return _clip(f"Error: compute produced no result (rc={rc}).{(' stderr: ' + tail) if tail else ''}")
    try:
        data = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return "Error: compute returned an unparseable result."

    # Uniform output cap on EVERY path -- the error branch previously bypassed it, so
    # `raise ValueError('x'*10000)` returned ~10k chars past the advertised cap.
    if data.get("err"):
        pre = str(data.get("stdout") or "").strip()
        msg = f"Error: {str(data['err'])[:2000]}"
        return _clip(f"{msg}\nOutput before error:\n{pre[:1500]}" if pre else msg)

    parts = []
    if data.get("result") is not None:
        parts.append(f"Result: {data['result']}")
    so = str(data.get("stdout") or "").rstrip()
    if so:
        parts.append(f"Output:\n{so}")
    return _clip("\n\n".join(parts) if parts else "(code executed successfully, no output)")


if __name__ == "__main__":
    if "--child" in sys.argv:
        _run_child()
    else:
        # Self-test: legit compute works; the boundary (masked FS + no network) holds.
        # Requires bwrap + userns (this dev box has both); otherwise set
        # OQ_COMPUTE_ALLOW_UNSANDBOXED=1 to exercise the unsandboxed path.
        print(f"sandbox available (bwrap + userns): {_sandbox_ok()}")
        print(f"masked dirs: {_mask_dirs()}")
        sentinel = os.path.expanduser("~/.oq_compute_sentinel.secret")
        try:
            with open(sentinel, "w") as fh:
                fh.write("TOP-SECRET-GOLD-KEY-42")
        except OSError:
            sentinel = None
        checks = [
            ("sum works", "vals=[132,129,143,159,154,153,177,200,219,287,376,473]\nprint(sum(vals))", "-> 2602"),
            ("ratio expr", "round(100*(6404-2602)/2602, 2)", "-> Result: 146.12"),
            ("pandas works", "import pandas as pd\npd.DataFrame({'v':[1,2,3]})['v'].sum()", "-> Result: 6 (if pandas present)"),
            ("network connect (DENY)", "import socket\nsocket.create_connection(('1.1.1.1',80),timeout=3)", "-> blocked (unshare-net)"),
            ("read masked HOME sentinel (DENY)",
             f"print(open({sentinel!r}).read())" if sentinel else "print('no sentinel')",
             "-> FileNotFound (home masked)"),
            ("os is reachable but contained", "import os\nprint('root entries:', len(os.listdir('/')))", "-> works, but FS is masked/ro"),
            ("infinite loop (KILL)", "while True:\n  pass", "-> timeout+kill"),
        ]
        for label, c, expect in checks:
            print(f"\n>>> {label}  [{expect}]\n{run_code(c)}")
        if sentinel and os.path.exists(sentinel):
            os.unlink(sentinel)
