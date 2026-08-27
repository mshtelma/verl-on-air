#!/usr/bin/env bash
# =============================================================================
# Preflight: can THIS machine build and register the image?
#
#   bash scripts/doctor.sh      (or: make doctor)
#
# Checks the four things that actually stop people, in the order they bite.
# Exits non-zero if any hard requirement fails, so it is safe in CI.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
[ -f config.env ] && IMAGE_LINE=$(grep -E '^(DOCKERHUB_USER|IMAGE_NAME|IMAGE_TAG)=' config.env | tr '\n' ' ')

hard_fail=0
warn=0

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; hard_fail=$((hard_fail+1)); }
wrn()  { printf '  \033[33mwarn\033[0m  %s\n' "$1"; warn=$((warn+1)); }

echo "== host =="
ARCH=$(uname -m)
OS=$(uname -s)
case "${ARCH}" in
  x86_64|amd64) ok "arch ${ARCH} on ${OS} — native amd64 build" ;;
  arm64|aarch64)
    wrn "arch ${ARCH} on ${OS} — the image MUST be linux/amd64."
    echo "        A local build runs under QEMU emulation: expect hours, and"
    echo "        upstream documents QEMU going flaky mid-stream on the large"
    echo "        wheels (torch ~4 GB, TransformerEngine ~1.5 GB)."
    echo "        Prefer a native x86_64 Linux host, or a remote buildx builder:"
    echo "          docker buildx create --name amd --platform linux/amd64 <endpoint>"
    ;;
  *) wrn "unrecognised arch ${ARCH}" ;;
esac

echo "== docker =="
if command -v docker >/dev/null 2>&1; then
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    ok "docker $(docker version --format '{{.Server.Version}}') (daemon reachable), server-arch=$(docker version --format '{{.Server.Arch}}' 2>/dev/null || echo '?')"
  else
    bad "docker CLI present but the daemon is unreachable — start Docker and retry"
  fi
  # Heredocs in RUN need BuildKit frontend >= 1.4; the Dockerfile declares
  # `# syntax=docker/dockerfile:1.7`, which requires BuildKit to be in use.
  if docker buildx version >/dev/null 2>&1; then
    ok "buildx $(docker buildx version | awk '{print $2}') (BuildKit available — required for the RUN heredoc)"
  else
    bad "docker buildx missing; BuildKit is required by '# syntax=docker/dockerfile:1.7'"
  fi
else
  bad "docker not installed — see docs/build-linux.md"
fi

echo "== disk =="
# Budget: ~4.7 GB base + ~11 GB of wheels + layer churn. 100 GB is comfortable,
# 60 GB is the practical floor, and Docker's own data root is what matters.
ROOT=$( { docker info --format '{{.DockerRootDir}}' 2>/dev/null; } || true )
ROOT=${ROOT:-/var/lib/docker}
PROBE=${ROOT}
while [ -n "${PROBE}" ] && [ ! -d "${PROBE}" ]; do PROBE=$(dirname "${PROBE}"); done
AVAIL_GB=$(df -Pk "${PROBE:-/}" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
if [ "${AVAIL_GB:-0}" -ge 100 ]; then
  ok "${AVAIL_GB} GiB free at ${PROBE} (comfortable)"
elif [ "${AVAIL_GB:-0}" -ge 60 ]; then
  wrn "${AVAIL_GB} GiB free at ${PROBE} — tight but usually enough; prune with 'docker system prune -af'"
else
  bad "${AVAIL_GB} GiB free at ${PROBE} — need >=60 GiB (100 GiB recommended)"
fi

echo "== registries / auth =="
if [ -f "${HOME}/.docker/config.json" ] && grep -q 'index.docker.io' "${HOME}/.docker/config.json" 2>/dev/null; then
  ok "Docker Hub credentials present (~/.docker/config.json)"
else
  wrn "no Docker Hub entry in ~/.docker/config.json — run 'docker login' before 'make push'"
fi

if command -v air >/dev/null 2>&1; then
  ok "air CLI present"
else
  wrn "air CLI missing — needed for register/run: uv tool install --force databricks-air --python 3.12"
fi

if command -v databricks >/dev/null 2>&1; then
  PROFILE=$(grep -E '^AIR_PROFILE=' config.env 2>/dev/null | cut -d= -f2)
  PROFILE=${PROFILE:-df1}
  if databricks current-user me -p "${PROFILE}" >/dev/null 2>&1; then
    ok "databricks profile '${PROFILE}' authenticated"
  else
    wrn "databricks profile '${PROFILE}' not authenticated — run: databricks auth login --profile ${PROFILE}"
    echo "        (a stale ~/.databrickscfg is NOT enough; recent CLIs reject the old token cache)"
  fi
else
  wrn "databricks CLI missing"
fi

echo
if [ "${hard_fail}" -gt 0 ]; then
  echo "${hard_fail} hard failure(s), ${warn} warning(s) — cannot build here. See docs/build-linux.md"
  exit 1
fi
echo "no hard failures (${warn} warning(s)). Ready: make build size push register"
