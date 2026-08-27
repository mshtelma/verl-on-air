#!/usr/bin/env bash
# =============================================================================
# One-shot: fresh x86_64 Linux box -> registered AI Runtime image.
#
#   git clone https://github.com/mshtelma/verl-on-air.git
#   cd verl-on-air
#   bash scripts/bootstrap_linux.sh
#
# What it does, in order:
#   1. installs docker (+ buildx), the databricks CLI and the air CLI if absent
#   2. runs scripts/doctor.sh and stops if the box cannot build
#   3. builds linux/amd64, gates on image size, pushes, registers
#
# CLEAN=1 bash scripts/bootstrap_linux.sh   -> from-scratch build (--no-cache --pull)
#
# Deliberately NOT idempotent-hostile: every step is skipped when already
# satisfied, so re-running after a failure is safe.
#
# Requires sudo only for the package installs. Set SKIP_INSTALL=1 to do none of
# them (e.g. on a managed box where you lack sudo but the tools already exist).
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

SKIP_INSTALL="${SKIP_INSTALL:-0}"
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- guards -----
ARCH=$(uname -m)
if [ "${ARCH}" != "x86_64" ] && [ "${ARCH}" != "amd64" ]; then
  echo "This script targets x86_64 Linux; found ${ARCH}." >&2
  echo "The image must be linux/amd64. On arm64, use a remote buildx builder" >&2
  echo "or a cloud build service instead - see docs/build-linux.md." >&2
  exit 1
fi
[ "$(uname -s)" = "Linux" ] || { echo "Linux only; found $(uname -s)" >&2; exit 1; }

# ------------------------------------------------------------- installs ------
if [ "${SKIP_INSTALL}" != "1" ]; then
  step "installing prerequisites (skip with SKIP_INSTALL=1)"

  if ! command -v docker >/dev/null 2>&1; then
    echo "-- installing docker via get.docker.com"
    curl -fsSL https://get.docker.com | sudo sh
    # Let the current user talk to the daemon without sudo. Requires a new login
    # shell to take effect, so we fall back to sudo -g below in this session.
    sudo usermod -aG docker "$USER" || true
    sudo systemctl enable --now docker || true
  else
    echo "-- docker already installed"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "-- installing uv"
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  if ! command -v databricks >/dev/null 2>&1; then
    echo "-- installing databricks CLI"
    curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh
  fi

  if ! command -v air >/dev/null 2>&1; then
    echo "-- installing air CLI"
    uv tool install --force databricks-air --python 3.12
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
fi

# If the group change has not taken effect in this shell, re-exec docker via sg.
if ! docker version >/dev/null 2>&1; then
  if command -v sg >/dev/null 2>&1 && sg docker -c 'docker version' >/dev/null 2>&1; then
    echo "-- docker group not active in this shell; re-executing under 'sg docker'"
    exec sg docker -c "SKIP_INSTALL=1 bash $0 $*"
  fi
fi

# ------------------------------------------------------------- preflight -----
step "preflight (scripts/doctor.sh)"
bash scripts/doctor.sh || {
  echo
  echo "Preflight failed. Fix the FAIL lines above, then re-run." >&2
  exit 1
}

# ------------------------------------------------------------------ auth -----
PROFILE=$(grep -E '^AIR_PROFILE=' config.env | cut -d= -f2)
step "auth checks (profile=${PROFILE})"

if ! databricks current-user me -p "${PROFILE}" >/dev/null 2>&1; then
  cat >&2 <<EOF
Databricks profile '${PROFILE}' is not authenticated.

Run this (it opens a browser; on a headless box it prints a device code):

  databricks auth login --host <workspace-url> --profile ${PROFILE}

NOTE a pre-existing ~/.databrickscfg is not sufficient - recent CLI versions
reject the old token cache with "stored credentials from older CLI versions are
no longer used". Then re-run this script.
EOF
  exit 1
fi
echo "  databricks: authenticated as $(databricks current-user me -p "${PROFILE}" 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("userName","?"))' 2>/dev/null || echo '?')"

if ! grep -q 'index.docker.io' "${HOME}/.docker/config.json" 2>/dev/null; then
  echo "  docker hub: not logged in -> running 'docker login'"
  docker login
fi

# `air register image` needs registry credentials stored in a Databricks secret.
# config.env carries SECRET_SCOPE/SECRET_KEY; without them `make register` falls
# back to the interactive flow, which reads the controlling TTY and would HANG in
# CI or any non-interactive shell. Warn early rather than block at the last step.
SEC_SCOPE=$(grep -E '^SECRET_SCOPE=' config.env | cut -d= -f2)
SEC_KEY=$(grep -E '^SECRET_KEY=' config.env | cut -d= -f2)
if [ -n "${SEC_SCOPE}" ] && [ -n "${SEC_KEY}" ]; then
  echo "  registry secret: ${SEC_SCOPE}/${SEC_KEY} (non-interactive registration)"
else
  echo "  registry secret: NOT configured -> registration will prompt interactively."
  echo "                   In CI this hangs. Run 'air register image ... -p <prof>'"
  echo "                   once by hand, then put the printed scope/key in config.env."
fi

# ----------------------------------------------------------------- build -----
step "build / size gate / push / register"
# CLEAN=1 forces --no-cache --pull. Worth it after any material Dockerfile change:
# a cached build can succeed on layers produced by an earlier, buggy version of the
# file, so it proves less than it looks like it does.
if [ "${CLEAN:-0}" = "1" ]; then make rebuild; else make build; fi
make size          # hard-fails above the DCS limit before wasting a registration
make push
make register

step "done"
cat <<EOF
Image is registered. Next:

  make smoke     # 1xA10 preflight (~2 min) - read cpu ram, EFA, AutoBridge
  make rung1     # Qwen3.5-2B  FSDP  8xH100  (cheap full-path check)
  make rung4     # 35B-A3B MoE Megatron-FSDP 16xH100  (headline)

Data and model are staged independently of this image (steps 01/02 run on a
stock environment), so if you already ran 'make prep' and 'make stage' from any
machine there is nothing else to do.
EOF
