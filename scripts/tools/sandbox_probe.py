#!/usr/bin/env python3
"""Probe: is a bubblewrap OS sandbox VIABLE in this runtime (esp. the Databricks
serverless-GPU container on df1)?

The compute tool's real isolation plan is to wrap the child interpreter in ``bwrap``
(bubblewrap). bwrap needs unprivileged USER NAMESPACES to be permitted by the container
runtime -- a property the Dockerfile CANNOT grant: it depends on the k8s / seccomp /
AppArmor policy that applies at RUN time. runc with a default seccomp profile usually
allows it; gVisor/runsc and hardened seccomp profiles often do not. So we probe directly
instead of assuming.

Run locally, or on air via ``air/80_sandbox_probe.yaml``. ALWAYS exits 0 -- it is a
report, not a gate. The decisive line is:

    USERNS(user+mnt+net)         : OK        <- bwrap will work once installed
    USERNS(user+mnt+net)         : BLOCKED   <- bwrap cannot work here; use the honest
                                               no-FS/net-boundary fallback instead
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return "(n/a)"


def _try_unshare(flags: int) -> tuple[bool, str]:
    """Fork a child and attempt ``unshare(flags)`` there, so a failure (or a successful
    namespace switch) cannot disturb the parent. Returns (ok, detail)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    pid = os.fork()
    if pid == 0:                                   # child
        ctypes.set_errno(0)
        rc = libc.unshare(ctypes.c_int(flags))
        os._exit(0 if rc == 0 else (ctypes.get_errno() or 1))
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    if code == 0:
        return True, "OK"
    detail = os.strerror(code) if 0 < code < 256 else "signalled/other"
    return False, f"errno={code} ({detail})"


def main() -> int:
    print("== sandbox viability probe ==", flush=True)
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.platform()}")
    print(f"image tag   : {os.environ.get('VERL_ON_AIR_IMAGE_TAG', '(unset)')}")
    print(f"euid/egid   : {os.geteuid()}/{os.getegid()}")

    print("-- kernel knobs --")
    print(f"unprivileged_userns_clone       : {_read('/proc/sys/kernel/unprivileged_userns_clone')}")
    print(f"user.max_user_namespaces        : {_read('/proc/sys/user/max_user_namespaces')}")
    print(f"apparmor_restrict_unpriv_userns : {_read('/proc/sys/kernel/apparmor_restrict_unprivileged_userns')}")

    print("-- unshare() attempts (the bwrap prerequisite) --")
    userns_ok = None
    for label, flags in [
        ("USERNS(user)", CLONE_NEWUSER),
        ("USERNS(user+mnt)", CLONE_NEWUSER | CLONE_NEWNS),
        ("USERNS(user+mnt+net)", CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET),
        ("USERNS(user+mnt+net+pid)", CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET | CLONE_NEWPID),
    ]:
        try:
            ok, msg = _try_unshare(flags)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"probe error: {type(e).__name__}: {e}"
        if userns_ok is None:
            userns_ok = ok
        print(f"  {label:28s}: {'OK' if ok else 'BLOCKED'}  {msg}", flush=True)

    print("-- bwrap --")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        print("  bwrap: ABSENT -- add `bubblewrap` to the image apt line (docker/Dockerfile "
              "step 1), rebuild, then re-run this probe to confirm the full path.")
        print("== end probe ==", flush=True)
        return 0

    ver = subprocess.run([bwrap, "--version"], capture_output=True, text=True)
    print(f"  bwrap: {bwrap}  {(ver.stdout or ver.stderr).strip()}")
    if userns_ok is False:
        print("  bwrap functional: SKIPPED -- user namespaces are blocked by the runtime")
        print("== end probe ==", flush=True)
        return 0
    # Functional test: run python inside a locked-down bwrap and assert network is denied.
    # Some container runtimes reject a fresh nested proc mount, so report the full command
    # with /proc and, if it fails, the equivalent compute-tool command without it.
    prog = (
        "import socket\n"
        "print('compute', 2 + 2)\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
        "    print('NET: OPEN  <- LEAK (unshare-net not effective)')\n"
        "except OSError as e:\n"
        "    print('NET: denied (' + type(e).__name__ + ')')\n"
    )
    variants = [
        ("with /proc", ["--unshare-pid", "--proc", "/proc"]),
        ("without /proc", []),
    ]
    for label, extra in variants:
        cmd = [
            bwrap, "--unshare-user", "--unshare-net", "--unshare-ipc", "--unshare-uts",
            *extra, "--die-with-parent",
            "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/tmp",
            sys.executable, "-c", prog,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            print(f"  bwrap functional ({label}) rc={r.returncode}")
            for ln in (r.stdout or "").splitlines():
                print(f"    out: {ln}")
            if r.stderr.strip():
                print(f"    err: {r.stderr.strip()[:500]}")
        except Exception as e:  # noqa: BLE001
            print(f"  bwrap functional ({label}) FAILED: {type(e).__name__}: {e}")
    print("== end probe ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
