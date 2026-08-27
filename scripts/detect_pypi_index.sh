#!/usr/bin/env bash
# =============================================================================
# Print the PyPI index URL this machine is configured to use, or nothing.
#
#   scripts/detect_pypi_index.sh          -> https://host/simple
#   scripts/detect_pypi_index.sh --source -> https://host/simple<TAB>~/.pip/pip.conf
#
# Single source of truth for the Makefile (--build-arg PIP_INDEX_URL) and
# doctor.sh. Both previously called `pip config get global.index-url` directly,
# which found nothing on a box that *does* use an internal proxy -- pip may be
# absent, or the setting may live in a uv config or an env var instead. Hence
# this checks every location, in precedence order.
#
# Precedence: explicit env > uv config > pip config > config files.
# =============================================================================
set -uo pipefail

WANT_SOURCE=0
[ "${1:-}" = "--source" ] && WANT_SOURCE=1

emit() {   # emit <url> <source>
  url=$(printf '%s' "$1" | tr -d '"'"'"' \t\r\n')
  [ -z "${url}" ] && return 1
  if [ "${WANT_SOURCE}" = "1" ]; then printf '%s\t%s\n' "${url}" "$2"; else printf '%s\n' "${url}"; fi
  exit 0
}

# 1. explicit environment ------------------------------------------------------
[ -n "${PIP_INDEX_URL:-}" ]     && emit "${PIP_INDEX_URL}"     "env PIP_INDEX_URL"
[ -n "${UV_INDEX_URL:-}" ]      && emit "${UV_INDEX_URL}"      "env UV_INDEX_URL"
[ -n "${UV_DEFAULT_INDEX:-}" ]  && emit "${UV_DEFAULT_INDEX}"  "env UV_DEFAULT_INDEX"

# 2. uv config -----------------------------------------------------------------
# uv.toml can express the index either as `index-url = "..."` (legacy) or as a
# [[index]] table with `default = true`. Take the first url in either form.
for f in "${UV_CONFIG_FILE:-}" "${HOME}/.config/uv/uv.toml" "${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml" /etc/uv/uv.toml; do
  [ -n "${f}" ] && [ -f "${f}" ] || continue
  url=$(sed -nE 's/^[[:space:]]*(index-url|url)[[:space:]]*=[[:space:]]*"([^"]+)".*/\2/p' "${f}" | head -1)
  [ -n "${url}" ] && emit "${url}" "${f}"
done

# 3. pip's own resolver (authoritative when pip exists) -------------------------
for pipcmd in "python3 -m pip" "python -m pip" pip3 pip; do
  # shellcheck disable=SC2086
  command -v ${pipcmd%% *} >/dev/null 2>&1 || continue
  url=$(${pipcmd} config get global.index-url 2>/dev/null | tail -1)
  [ -n "${url}" ] && emit "${url}" "${pipcmd} config"
  url=$(${pipcmd} config list 2>/dev/null | sed -nE "s/^global\\.index-url='?([^']+)'?.*/\\1/p" | head -1)
  [ -n "${url}" ] && emit "${url}" "${pipcmd} config list"
done

# 4. pip config files ----------------------------------------------------------
for f in "${PIP_CONFIG_FILE:-}" "${HOME}/.config/pip/pip.conf" "${HOME}/.pip/pip.conf" \
         "${XDG_CONFIG_HOME:-$HOME/.config}/pip/pip.conf" /etc/pip.conf /etc/xdg/pip/pip.conf; do
  [ -n "${f}" ] && [ -f "${f}" ] || continue
  url=$(sed -nE 's/^[[:space:]]*(index-url|index_url)[[:space:]]*=[[:space:]]*(.+)$/\2/p' "${f}" | head -1)
  [ -n "${url}" ] && emit "${url}" "${f}"
done

exit 1   # nothing found
