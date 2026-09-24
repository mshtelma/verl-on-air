#!/usr/bin/env bash
# =============================================================================
# Read the air `parameters:` block into shell variables.
#
# air materialises `parameters:` as a **YAML** file at $HYPERPARAMETERS_PATH.
# (It is not JSON — a `json.load` here fails with a confusing parse error.)
#
# Usage:
#   source engine/lib/hparams.sh
#   MODEL_PATH=$(hp model_name "Qwen/Qwen3.5-35B-A3B")
#
# MISSING vs EMPTY is a real distinction and this helper preserves it:
#
#   (key absent)      -> the default
#   image_key: ""     -> "" , NOT the default
#
# That matters because `image_key: ""` is how a YAML says "this dataset is
# text-only, do not pass data.image_key". Collapsing empty into the default
# silently re-enables the multimodal path. The helper signals "absent" with
# exit code 42 rather than by returning an empty string.
#
# A file that is not a valid YAML mapping is an ERROR, never "every key absent":
# falling back to defaults would run a different job than the one written.
# hp_check (called by hp_dump) stops the script on it; hp itself fails (exit 2).
# =============================================================================

# hp <key> [default]
hp() {
  local key="$1" default="${2-}" val rc

  if val="$(
      HP_KEY="$key" python3 - <<'PY'
import os, sys

key  = os.environ["HP_KEY"]
path = os.environ.get("HYPERPARAMETERS_PATH", "")
MISSING = 42

if not path or not os.path.exists(path):
    sys.exit(MISSING)

import yaml
try:
    with open(path) as fh:
        data = yaml.safe_load(fh)
except (OSError, yaml.YAMLError) as e:
    print(f"FATAL: {path} (the air parameters block) is not valid YAML: {e}", file=sys.stderr)
    sys.exit(2)
data = {} if data is None else data
if not isinstance(data, dict):
    print(f"FATAL: {path} (the air parameters block) is not a mapping of key: value", file=sys.stderr)
    sys.exit(2)

if key not in data:
    sys.exit(MISSING)

value = data[key]
print("" if value is None else value)
PY
  )"; then
    rc=0
  else
    rc=$?
  fi

  case "${rc}" in
    0)  printf '%s' "${val}" ;;
    42) printf '%s' "${default}" ;;
    *)  return "${rc}" ;;          # malformed parameters: an error, not a default
  esac
}

# hp_check: stop the script unless the parameters block (if any) is a valid YAML mapping.
hp_check() {
  hp __voa_hp_check__ >/dev/null || { echo "FATAL: fix the job's parameters: block." >&2; exit 1; }
}

# hp_has <key>: true if the parameters block sets <key> (to anything, even "").
hp_has() {
  [ "$(hp "$1" "__voa_hp_missing__")" != "__voa_hp_missing__" ]
}

# Validate, then echo the whole parameter block once, for the job log.
hp_dump() {
  hp_check
  [ -n "${HYPERPARAMETERS_PATH:-}" ] && [ -f "${HYPERPARAMETERS_PATH}" ] || return 0
  echo "---------- air parameters (${HYPERPARAMETERS_PATH}) ----------"
  cat "${HYPERPARAMETERS_PATH}"
  echo "--------------------------------------------------------------"
}
