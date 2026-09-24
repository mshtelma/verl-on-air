#!/bin/sh
# uvi <uv args...>: uv with this build's package index and pins.
#
# The index comes from the BuildKit secret `pip_index` when the build was given one (`make build`
# passes the auto-detected index that way), else uv's default (PyPI). A secret, not a build ARG:
# an ARG that a RUN uses is recorded in the image's history, so an index URL carrying credentials
# would ship inside the image's metadata -- and clearing an ENV at the end does not undo that.
# Exported, not passed as --index-url, so uv's inner build-dependency resolutions (a source
# install's setuptools/wheel) use the same index.
#
# `uvi pip install` is also constrained to VOA_CONSTRAINTS (docker/requirements.lock) while the
# build sets it; `VOA_CONSTRAINTS= uvi ...` opts one install out (the judge's own Ray).
set -eu
if [ -s /run/secrets/pip_index ]; then
    UV_DEFAULT_INDEX="$(cat /run/secrets/pip_index)"
    UV_INDEX_URL="${UV_DEFAULT_INDEX}"
    export UV_DEFAULT_INDEX UV_INDEX_URL
fi
if [ -n "${VOA_CONSTRAINTS:-}" ] && [ "${1:-}" = "pip" ] && [ "${2:-}" = "install" ]; then
    shift 2
    set -- pip install --constraint "${VOA_CONSTRAINTS}" "$@"
fi
exec uv "$@"
