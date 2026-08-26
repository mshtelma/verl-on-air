#!/usr/bin/env bash
# =============================================================================
# Read the air `parameters:` block into shell variables.
#
# air materialises `parameters:` as a **YAML** file at $HYPERPARAMETERS_PATH.
# (It is not JSON — a `json.load` here fails with a confusing parse error.)
#
# Usage:
#   source scripts/lib/hparams.sh
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

data = None
try:
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
except Exception:
    # Minimal "key: value" fallback if pyyaml is somehow unavailable.
    data = {}
    try:
        with open(path) as fh:
            for line in fh:
                k, sep, v = line.partition(":")
                if sep and not k.startswith((" ", "\t", "#")):
                    data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        sys.exit(MISSING)

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

  if [ "${rc}" -eq 0 ]; then
    printf '%s' "${val}"
  else
    printf '%s' "${default}"
  fi
}

# Echo the whole parameter block once, for the job log.
hp_dump() {
  [ -n "${HYPERPARAMETERS_PATH:-}" ] && [ -f "${HYPERPARAMETERS_PATH}" ] || return 0
  echo "---------- air parameters (${HYPERPARAMETERS_PATH}) ----------"
  cat "${HYPERPARAMETERS_PATH}"
  echo "--------------------------------------------------------------"
}
