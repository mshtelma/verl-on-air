#!/usr/bin/env bash
# Runtime environment probe: where do python3 / ray / the venv actually live when
# air runs the container? (The image ENV PATH != the air runtime PATH — DCS
# overlays its own pyenv, which broke bare `ray` in ray_cluster.sh on multi-node.)
set -uo pipefail
echo "================ diag_env ================"
echo "PATH=$PATH"
echo "whoami=$(whoami)  pwd=$(pwd)"
echo "--- python3 ---"
echo "which python3      : $(command -v python3 2>&1)"
echo "sys.executable     : $(python3 -c 'import sys; print(sys.executable)' 2>&1)"
echo "sys.prefix         : $(python3 -c 'import sys; print(sys.prefix)' 2>&1)"
echo "--- ray: is it importable by THIS python3, and where is its CLI? ---"
python3 - <<'PY' 2>&1
try:
    import ray, os
    print("import ray OK  ray.__file__ =", ray.__file__)
    print("ray version   =", getattr(ray, "__version__", "?"))
    binguess = os.path.join(os.path.dirname(os.path.dirname(ray.__file__)), "..", "bin", "ray")
    print("bin guess     =", os.path.normpath(binguess), "exists=", os.path.exists(os.path.normpath(binguess)))
    from ray.scripts.scripts import main  # the console_script entry point
    print("ray.scripts.scripts.main import OK  -> python3 -m works")
except Exception as e:
    print("RAY IMPORT/CLI FAILED:", type(e).__name__, e)
PY
echo "--- bare ray on PATH? ---"
echo "which ray          : $(command -v ray 2>&1)"
echo "--- candidate ray binaries on disk ---"
find / -maxdepth 6 -name ray -type f -path '*/bin/*' 2>/dev/null | head -5
echo "--- /opt/venv present at runtime? ---"
ls -ld /opt/venv /opt/venv/bin 2>&1 | head
ls /opt/venv/bin/ray /opt/venv/bin/python3 2>&1 | head
echo "=========================================="
