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

echo "== build indexes + container egress =="
# A real build died here: apt succeeded, then `uv pip install pybind11` failed with
# "dns error: failed to lookup address information" for pypi.org. Cause: the box
# uses an internal PyPI proxy configured in ~/.pip/pip.conf, and the build
# container does not inherit that config, so uv fell back to unreachable pypi.org.
#
# So probe the index we will ACTUALLY pass to the build, plus the other hosts the
# Dockerfile needs. Note pypi.org itself is NOT required when a proxy is set.
DETECTED_PAIR=$(bash scripts/detect_pypi_index.sh --source 2>/dev/null || true)
DETECTED_INDEX=${DETECTED_PAIR%%$'\t'*}
DETECTED_FROM=${DETECTED_PAIR#*$'\t'}
if [ -n "${DETECTED_INDEX}" ]; then
  ok "PyPI index: ${DETECTED_INDEX}"
  echo "        (from ${DETECTED_FROM}; passed to the build as --build-arg PIP_INDEX_URL)"
else
  wrn "no PyPI index configured anywhere -> the build would use public pypi.org"
  echo "        Checked: \$PIP_INDEX_URL, \$UV_INDEX_URL, \$UV_DEFAULT_INDEX,"
  echo "                 uv.toml, pip config, pip.conf (see scripts/detect_pypi_index.sh)"
fi
INDEX_URL=${PIP_INDEX_URL:-${DETECTED_INDEX:-https://pypi.org/simple}}
INDEX_HOST=$(printf '%s' "${INDEX_URL}" | sed -E 's#^[a-z]+://([^/]+).*#\1#')

# torch comes from the same index by default: PyPI's torch 2.11.0 IS the cu13
# build (requires nvidia-cudnn-cu13 etc.), so download.pytorch.org is not needed.
# DNS is necessary but NOT sufficient: a build died with
#   invalid peer certificate: UnknownIssuer
# for github.com while DNS was perfectly fine. Corporate TLS inspection presents
# an internal-CA cert, and uv links rustls with BUNDLED roots, ignoring the system
# trust store. So probe an actual HTTPS handshake, not just resolution.
if docker version >/dev/null 2>&1; then
  TE_URL="https://github.com/verl-project/verl-wheelhouse/releases/download/transformer-engine-v2.16.1/transformer_engine-2.16.1-cp312-cp312-linux_x86_64.whl"
  probe=$(docker run --rm --entrypoint sh alpine:3 -c "
      apk add --no-cache curl >/dev/null 2>&1 || true
      for u in '${INDEX_URL}' '${TE_URL}'; do
        code=\$(curl -sS -o /dev/null -w '%{http_code}' --max-time 25 -r 0-1 -L \"\$u\" 2>&1)
        case \"\$code\" in
          200|206|302) echo \"good \$u\" ;;
          *)           echo \"BAD \$u :: \$code\" ;;
        esac
      done" 2>/dev/null)
  if [ -z "${probe}" ]; then
    wrn "egress probe container did not run (cannot pull alpine:3?)"
  else
    printf '%s\n' "${probe}" | grep '^good' | while read -r _ u; do
      ok "TLS+HTTP ok: $(printf '%s' "$u" | cut -c1-72)"
    done
    if printf '%s\n' "${probe}" | grep -q '^BAD'; then
      printf '%s\n' "${probe}" | grep '^BAD' | sed 's/^BAD /        /' | while read -r l; do
        bad "unreachable from container: ${l}"
      done
      # A cert error is a different problem from a DNS/connect error.
      if printf '%s\n' "${probe}" | grep -qiE 'certificate|SSL|60\)'; then
        echo "        This looks like TLS INTERCEPTION (internal CA). Fix:"
        echo "          make certs      # copy this host's CA bundle into ./certs"
        echo "          make build      # image trusts it; UV_NATIVE_TLS=1 makes uv use it"
        echo "        If that still fails, bypass github entirely:"
        echo "          make vendor     # pre-fetch wheels/sources on the host"
        echo "          make build"
      else
        echo "        Diagnose:"
        echo "          docker run --rm alpine:3 getent hosts ${INDEX_HOST}"
        echo "          cat /etc/resolv.conf ; env | grep -i proxy"
        echo "        Fixes: /etc/docker/daemon.json {\"dns\":[\"8.8.8.8\"]} + restart docker,"
        echo "               or  make build BUILD_ARGS='--build-arg HTTPS_PROXY=http://proxy:3128'"
        echo "               or  make vendor   # skip github entirely"
      fi
    fi
  fi
else
  wrn "skipping container egress probe (docker unavailable)"
fi

# Report what is already mitigated locally.
if ls certs/*.crt certs/*.pem >/dev/null 2>&1; then
  ok "extra CA bundle staged in ./certs (will be trusted in the image)"
fi
if ls vendor/wheels/*.whl >/dev/null 2>&1; then
  ok "$(ls vendor/wheels/*.whl | wc -l | tr -d ' ') vendored wheel(s) present — github not needed for those"
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
