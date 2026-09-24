#!/usr/bin/env bash
# =============================================================================
# Bump IMAGE_TAG in config.env, then point every custom-image job YAML (infra/ +
# usecases/) at the new image via scripts/retarget.py.
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
# instead of templating. Changing DOCKERHUB_USER / IMAGE_NAME needs no bump:
# `make retarget` applies config.env to the job files.
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

echo "bumping IMAGE_TAG: ${CUR} -> ${NEW}"

cp config.env config.env.bak
sed -i.tmp -E "s|^IMAGE_TAG=${CUR}\$|IMAGE_TAG=${NEW}|" config.env && rm -f config.env.tmp

# Rewrite every job's environment.docker_image.url from config.env BY FIELD -- not by
# searching for "<user>/<name>:<old tag>", which silently matched nothing once
# DOCKERHUB_USER/IMAGE_NAME had been customised. --expect-change fails if no job moved.
if ! "${PYTHON:-python3}" scripts/retarget.py --expect-change; then
    mv -f config.env.bak config.env
    echo "bump FAILED -- config.env restored to IMAGE_TAG=${CUR}" >&2
    exit 1
fi
rm -f config.env.bak
echo
echo "Now: make release      # rebuild -> size gate -> push -> register the NEW tag"
