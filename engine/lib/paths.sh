#!/usr/bin/env bash
# shellcheck shell=bash
# =============================================================================
# Path resolution for values a job YAML puts in `env_variables:`.
#
# AI Runtime expands ${CODE_SOURCE_PATH} in `command:` only. Values under
# `env_variables:` reach the process LITERALLY, so
#     EVAL_SCRIPT: ${CODE_SOURCE_PATH}/usecases/agentic-search/eval.py
# arrives with the `${...}` still in it -- and running the launcher from `command:`
# does NOT expand the CONTENTS of that variable. Every engine entrypoint that takes a
# path from the environment resolves it here. The value is treated as data: only the
# two spellings of that one variable are substituted (never `eval`), and a relative
# path is anchored at the code snapshot.
#
#   source engine/lib/paths.sh
#   EVAL_SCRIPT="$(resolve_code_path "${EVAL_SCRIPT}")"
# =============================================================================

VOA_REPO_ROOT="${VOA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

resolve_code_path() {  # resolve_code_path <value>  -> absolute path on stdout
    local p="$1" base="${CODE_SOURCE_PATH:-${VOA_REPO_ROOT}}"
    p="${p//\$\{CODE_SOURCE_PATH\}/${base}}"
    p="${p//\$CODE_SOURCE_PATH/${base}}"
    if [[ "${p}" != /* ]]; then p="${base}/${p}"; fi
    printf '%s' "${p}"
}
