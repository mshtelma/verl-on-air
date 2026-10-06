#!/usr/bin/env python3
"""Trusted AIR diagnostics; no model-generated code is executed by this job."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import sandbox


def main() -> None:
    base = ["bwrap", "--unshare-user", "--unshare-net", "--unshare-pid",
            "--unshare-ipc", "--unshare-uts", "--ro-bind", "/", "/"]
    variants = {
        "fresh_proc_dev": ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"],
        "empty_proc_dev": ["--tmpfs", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"],
        "empty_proc": ["--tmpfs", "/proc", "--tmpfs", "/tmp"],
        "dev_only": ["--dev", "/dev", "--tmpfs", "/tmp"],
    }
    report = {"run_id": os.environ.get("RUN_ID"), "mount_probes": {}}
    for name, flags in variants.items():
        result = subprocess.run([*base, *flags, "true"], capture_output=True, text=True, timeout=15)
        row = {"rc": result.returncode, "stderr": result.stderr.strip()}
        report["mount_probes"][name] = row
        print(json.dumps({"mount_probe": name, **row}), flush=True)
    # The candidate hides host proc completely instead of attempting a forbidden
    # proc mount. All other Miles filesystem/environment/network controls remain.
    if report["mount_probes"]["empty_proc_dev"]["rc"] != 0:
        raise RuntimeError("empty /proc sandbox candidate is unavailable")
    command = sandbox._bwrap_cmd()
    command[command.index("--proc")] = "--tmpfs"
    checks = {
        "arithmetic": ("print(100 + 200)", "300"),
        "numpy_pandas": ("print(int(np.array([1,2,3]).sum()), int(pd.DataFrame({'x':[2,3]})['x'].sum()))", "6 5"),
        "proc_hidden": ("import os\nprint(os.listdir('/proc'))", "[]"),
        "data_hidden": ("import os\nprint(os.listdir('/Volumes'), os.listdir('/databricks'))", "[] []"),
        "env_cleared": ("import os\nprint(os.environ.get('OQ_PROBE_SECRET', 'absent'))", "absent"),
        "network_namespace": ("import socket\nprint([name for _, name in socket.if_nameindex()])", "['lo']"),
    }
    os.environ["OQ_PROBE_SECRET"] = "probe-only"
    report["compute_checks"] = {}
    for name, (code, expected) in checks.items():
        result = subprocess.run(command, input=code, capture_output=True, text=True,
                                timeout=20, env={"PATH": os.environ["PATH"]})
        parsed = json.loads(result.stdout) if result.stdout else {}
        passed = result.returncode == 0 and parsed.get("ok") and parsed.get("stdout", "").strip() == expected
        row = {"pass": bool(passed), "rc": result.returncode, "result": parsed,
               "stderr": result.stderr.strip()[-500:]}
        report["compute_checks"][name] = row
        print(json.dumps({"compute_check": name, **row}), flush=True)
    report["status"] = "PASS" if all(row["pass"] for row in report["compute_checks"].values()) else "FAIL"
    if os.environ.get("OQ_PROBE_OUT"):
        path = Path(os.environ["OQ_PROBE_OUT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("OfficeQA compute qualification failed")


if __name__ == "__main__":
    main()
