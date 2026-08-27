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
# signed by an internal CA. `make certs` + UV_NATIVE_TLS usually fixes that; this
# is the fallback that removes the need for the container to reach github at all.
#
# The host already trusts the intercepting CA (that is why curl/git work here),
# so we fetch with the host's tools and COPY the results into the image.
#
# Populates:
#   vendor/wheels/  transformer_engine, apex, flash_attn, megatron_bridge
#   vendor/src/     megatron-lm (core_v0.18.0), mbridge (pinned rev)
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

WH=https://github.com/verl-project/verl-wheelhouse/releases/download
MCORE_VERSION=${MCORE_VERSION:-core_v0.18.0}
MBRIDGE_REV=${MBRIDGE_REV:-641a5a0}

mkdir -p vendor/wheels vendor/src

fetch() {  # fetch <url>
  local url="$1" out="vendor/wheels/$(basename "$1")"
  if [ -s "${out}" ]; then
    echo "  have  $(basename "${out}") ($(du -h "${out}" | cut -f1))"
    return
  fi
  echo "  get   $(basename "${out}")"
  curl -fSL --retry 5 --retry-delay 3 --retry-all-errors -o "${out}.part" "${url}"
  mv "${out}.part" "${out}"
  echo "        -> $(du -h "${out}" | cut -f1)"
}

echo "== wheels (verl wheelhouse, cu130/torch-2.11/cp312) =="
fetch "${WH}/transformer-engine-v2.16.1/transformer_engine-2.16.1-cp312-cp312-linux_x86_64.whl"
fetch "${WH}/apex-master/apex-0.1-cp312-cp312-linux_x86_64.whl"
fetch "${WH}/flash-attention-v2.8.3/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
fetch "${WH}/megatron-bridge-r0.5.0/megatron_bridge-0.5.2-py3-none-any.whl"

clone() {  # clone <url> <ref> <dest>
  local url="$1" ref="$2" dest="vendor/src/$3"
  if [ -d "${dest}/.git" ]; then
    echo "  have  $3 ($(git -C "${dest}" describe --tags --always 2>/dev/null || echo '?'))"
    return
  fi
  echo "  clone $3 @ ${ref}"
  rm -rf "${dest}"
  git clone --quiet --depth 1 --branch "${ref}" "${url}" "${dest}" 2>/dev/null \
    || { git clone --quiet "${url}" "${dest}" && git -C "${dest}" checkout --quiet "${ref}"; }
  # Drop .git to keep the build context small (the tree is what pip needs).
  rm -rf "${dest}/.git"
}

echo "== sources =="
clone https://github.com/NVIDIA/Megatron-LM.git "${MCORE_VERSION}" megatron-lm
clone https://github.com/ISEEKYAN/mbridge.git   "${MBRIDGE_REV}"   mbridge

echo
echo "vendored:"
du -sh vendor/wheels vendor/src 2>/dev/null | sed 's/^/  /'
echo
echo "The Dockerfile now prefers these over github.com. Re-run: make build"
