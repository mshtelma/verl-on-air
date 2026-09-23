#!/usr/bin/env bash
# =============================================================================
# The job's rendezvous files: how nodes that share nothing but a Unity Catalog
# Volume tell each other things (dispatch_agentic.sh; the dir is per RUN_ID).
#
#   rdv_put  <file> <value>      atomic: write a temp file, then rename -- a reader
#                                never sees a half-written value
#   rdv_wait <file> <timeout_s>  print the value once it exists AND was written
#                                during this job; return 1 at the deadline
#
# "During this job" = modified at or after DISPATCH_T0 - RDV_SKEW_S (default 300 s of
# clock skew between nodes). A resumed run reuses its RUN_ID's dir, so without this a
# judge endpoint or head address left by the last attempt would be taken for a live one.
# =============================================================================

rdv_put() {
    local f="$1" v="$2"
    printf '%s\n' "${v}" > "${f}.tmp.$$"
    mv -f "${f}.tmp.$$" "${f}"
}

rdv_wait() {
    local f="$1" deadline=$(( $(date +%s) + ${2:-1800} ))
    local fresh=$(( ${DISPATCH_T0:-0} - ${RDV_SKEW_S:-300} ))
    until [ -s "${f}" ] && [ "$(stat -c %Y "${f}" 2>/dev/null || echo 0)" -ge "${fresh}" ]; do
        [ "$(date +%s)" -ge "${deadline}" ] && return 1
        sleep "${RDV_POLL_S:-5}"
    done
    cat "${f}"
}
