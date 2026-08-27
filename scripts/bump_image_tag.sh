#!/usr/bin/env bash
# =============================================================================
# Bump IMAGE_TAG in config.env and in every air/*.yaml that references the image.
#
#   bash scripts/bump_image_tag.sh          # v1 -> v2
#   bash scripts/bump_image_tag.sh v7       # explicit
#
# WHY THIS IS NECESSARY, not just tidy:
# `air register image` caches per IMAGE TAG. Re-pushing the same tag with new
# content does NOT get picked up -- the platform keeps serving the digest it
# registered. We lost a full debug cycle to this: the nvcc fix was built, pushed
# and "registered", yet jobs kept running the previous :v1 image
# ("Using cached image: sha256:23d37a3c2..."), so the fix appeared not to work.
#
# The air YAMLs deliberately carry the image reference LITERALLY (so any one of
# them can be read and submitted by hand), which is why this rewrites them
# instead of templating.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

CUR=$(grep -E '^IMAGE_TAG=' config.env | cut -d= -f2)
[ -n "${CUR}" ] || { echo "IMAGE_TAG not found in config.env" >&2; exit 1; }

if [ $# -ge 1 ]; then
    NEW="$1"
elif [[ "${CUR}" =~ ^v([0-9]+)$ ]]; then
    NEW="v$(( ${BASH_REMATCH[1]} + 1 ))"
else
    echo "cannot auto-increment tag '${CUR}'; pass one explicitly" >&2
    exit 1
fi

USER_NAME=$(grep -E '^DOCKERHUB_USER=' config.env | cut -d= -f2)
NAME=$(grep -E '^IMAGE_NAME=' config.env | cut -d= -f2)

echo "bumping ${USER_NAME}/${NAME}: ${CUR} -> ${NEW}"

sed -i.bak -E "s|^IMAGE_TAG=${CUR}\$|IMAGE_TAG=${NEW}|" config.env && rm -f config.env.bak

changed=0
for f in air/*.yaml; do
    if grep -q "${USER_NAME}/${NAME}:${CUR}" "$f"; then
        sed -i.bak "s|${USER_NAME}/${NAME}:${CUR}|${USER_NAME}/${NAME}:${NEW}|g" "$f"
        rm -f "$f.bak"
        echo "  updated $f"
        changed=$((changed + 1))
    fi
done
echo "  ${changed} YAML file(s) updated"
echo
echo "Now: make release      # rebuild -> size gate -> push -> register the NEW tag"
