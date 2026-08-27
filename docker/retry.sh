#!/bin/sh
# retry <attempts> <command...>
#
# Every network step in the Dockerfile goes through this. Rationale:
#
#  * Transient DNS/TLS failures are common on corporate networks and CI, and the
#    wheels here are large (torch ~4 GB, TransformerEngine ~1.5 GB), so a build
#    has a long exposure window.
#  * The original inline `cmd && break || warn` loops exited 0 when every attempt
#    failed (the || branch succeeds), which would have shipped a silently broken
#    image. This helper exits non-zero on exhaustion, so the build fails loudly.
#  * Backoff is linear-ish (5s, 10s, 15s, ...) because DNS/proxy hiccups tend to
#    clear in tens of seconds, and we do not want to add minutes to a good build.
#
# Usage in the Dockerfile:
#   RUN retry 6 uv pip install --no-cache foo bar
set -eu

attempts="$1"
shift

i=1
while [ "$i" -le "$attempts" ]; do
    if "$@"; then
        exit 0
    fi
    if [ "$i" -lt "$attempts" ]; then
        delay=$((i * 5))
        echo "retry: attempt $i/$attempts failed, sleeping ${delay}s :: $*" >&2
        sleep "$delay"
    fi
    i=$((i + 1))
done

echo "retry: FATAL - all $attempts attempts failed :: $*" >&2
echo "retry: if this is a DNS/egress failure, run 'make doctor' - it probes" >&2
echo "retry: container DNS and reachability of every index this build needs." >&2
exit 1
