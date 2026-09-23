#!/usr/bin/env bash
# =============================================================================
# Pre-fetch the non-PyPI build artefacts on the HOST, for networks where the
# container cannot complete TLS to github.com.
#
#   bash scripts/vendor_artifacts.sh      (or: make vendor)
#
# Why this exists: uv inside the build container failed with
#   invalid peer certificate: UnknownIssuer
# for https://github.com/... because corporate TLS inspection presents a cert
# signed by an internal CA. `make certs` + UV_SYSTEM_CERTS usually fixes that; this
# is the fallback that removes the need for the container to reach github at all.
#
# The host already trusts the intercepting CA (that is why curl/git work here),
# so we fetch with the host's tools; the build bind-mounts the results.
#
# What to fetch, and its identity, is docker/artifacts.lock -- the same file the
# Dockerfile reads. Nothing is accepted on trust:
#   vendor/wheels/  each wheel must match its pinned sha256 (a mismatch is deleted
#                   and fails the run)
#   vendor/src/     each source tree is checked out at its pinned COMMIT, and the
#                   commit is recorded in <tree>/.voa-commit (the .git dir is
#                   dropped); the Dockerfile refuses a tree whose record differs
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

LOCK=docker/artifacts.lock
mkdir -p vendor/wheels vendor/src

sha256() { sha256sum "$1" | cut -d' ' -f1; }

fetch() {  # fetch <name> <url> <sha256>
  local name="$1" url="$2" want="$3" out
  out="vendor/wheels/$(basename "${url}")"
  if [ -s "${out}" ] && [ "$(sha256 "${out}")" = "${want}" ]; then
    echo "  have  ${name} ($(du -h "${out}" | cut -f1), sha256 ok)"
    return
  fi
  echo "  get   ${name}"
  curl -fSL --retry 5 --retry-delay 3 --retry-all-errors -o "${out}.part" "${url}"
  local got
  got="$(sha256 "${out}.part")"
  if [ "${got}" != "${want}" ]; then
    rm -f "${out}.part"
    echo "FATAL: ${url} has sha256 ${got}; ${LOCK} pins ${want}" >&2
    exit 1
  fi
  mv "${out}.part" "${out}"
  echo "        -> $(du -h "${out}" | cut -f1), sha256 ok"
}

checkout() {  # checkout <name> <repo> <commit>
  local name="$1" repo="$2" commit="$3" dest="vendor/src/$1"
  if [ "$(cat "${dest}/.voa-commit" 2>/dev/null || true)" = "${commit}" ]; then
    echo "  have  ${name} @ ${commit:0:12}"
    return
  fi
  echo "  clone ${name} @ ${commit:0:12}"
  rm -rf "${dest}"
  git init --quiet "${dest}"
  git -C "${dest}" fetch --quiet --depth 1 "${repo}" "${commit}" \
    || git -C "${dest}" fetch --quiet "${repo}"
  git -C "${dest}" checkout --quiet "${commit}"
  local head
  head="$(git -C "${dest}" rev-parse HEAD)"
  [ "${head}" = "${commit}" ] || { echo "FATAL: ${name} checked out ${head}, wanted ${commit}" >&2; exit 1; }
  # Drop .git to keep the build context small (the tree is what pip needs), and record
  # the identity the Dockerfile checks.
  rm -rf "${dest}/.git"
  printf '%s\n' "${commit}" > "${dest}/.voa-commit"
}

echo "== wheels (verl wheelhouse, cu130/torch-2.11/cp312) =="
while read -r kind name _version src pin; do
  [ "${kind}" = "wheel" ] && fetch "${name}" "${src}" "${pin}"
done < <(grep -E '^wheel[[:space:]]' "${LOCK}")

echo "== sources =="
while read -r kind name _version src pin; do
  [ "${kind}" = "git" ] && checkout "${name}" "${src}" "${pin}"
done < <(grep -E '^git[[:space:]]' "${LOCK}")

echo
echo "vendored:"
du -sh vendor/wheels vendor/src 2>/dev/null | sed 's/^/  /'
echo
echo "The build now uses these instead of github.com (after verifying them). Re-run: make build"
