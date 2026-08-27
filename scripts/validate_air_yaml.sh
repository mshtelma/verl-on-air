#!/usr/bin/env bash
# =============================================================================
# Validate every air/*.yaml against the REAL air CLI, without needing the custom
# image to be registered yet.
#
# Why the substitution: `air run --dry-run` checks the YAML schema first, but then
# also verifies that `environment.docker_image.url` is registered. Before the
# first `make register` that check fails for every file and masks whatever else
# might be wrong. So we swap the custom image for a stock environment, which
# leaves the schema, compute topology, code_source and parameters intact and
# exercises exactly the parts we author by hand.
#
#   bash scripts/validate_air_yaml.sh [profile]
#
# Note the probe files are written INTO air/ (as .probe_*.yaml, gitignored) so
# that relative paths like `root_path: ..` resolve the same way they will for the
# real file — air resolves them relative to the YAML's own location.
# =============================================================================
set -uo pipefail

PROFILE="${1:-${AIR_PROFILE:-df1}}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

command -v air >/dev/null || { echo "air CLI not found on PATH"; exit 1; }

fails=0
for f in air/*.yaml; do
    case "$(basename "$f")" in .probe_*) continue ;; esac
    probe="air/.probe_$(basename "$f")"

    python3 - "$f" > "$probe" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
src = re.sub(
    r"environment:\n  docker_image:\n    url: .*\n",
    "environment:\n  version: 'databricks_ai_v5'\n  dependencies:\n    - mlflow\n",
    src,
)
sys.stdout.write(src)
PY

    printf '  %-44s ' "$(basename "$f")"
    if out=$(air run --dry-run --file "$probe" -p "${PROFILE}" 2>&1); then
        echo "OK"
    else
        echo "FAIL"
        echo "${out}" | grep -viE '^\s*$' | tail -8 | sed 's/^/      /'
        fails=$((fails + 1))
    fi
    rm -f "$probe"
done

echo
if [ "${fails}" -ne 0 ]; then
    echo "${fails} file(s) failed schema validation"
    exit 1
fi
echo "all air/*.yaml pass schema validation (profile=${PROFILE})"
