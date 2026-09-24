#!/usr/bin/env bash
# =============================================================================
# Running the training driver -- shared by both launchers.
#
#   run_driver <log> <cmd...>     runs <cmd> to completion; its exit code lands in DRIVER_RC
#
#  * The driver runs as its OWN process group (job control on for this one job, off
#    inside it so the tee pipeline stays in the group), so one signal stops python AND tee.
#  * The abort channel (engine/lib/run_control.py): with ABORT_FILE set, a watchdog stops
#    the group once any component writes that file -- verl swallows a raise on the reward
#    and rollout paths, so a component that must stop the run writes a file instead.
#  * Signals: the launcher WAITS for the driver rather than running it in the foreground,
#    because bash defers a trap until a foreground child exits -- a cancelled launcher would
#    sit on its TERM until training ended. On TERM/INT/HUP the driver group is stopped and
#    the launcher exits 128+n. On a multi-node head, ray_install_cleanup_trap owns those
#    traps (and records the signal for the workers); here they are installed only if unset.
# =============================================================================

abort_watchdog() {  # abort_watchdog <pgid> <abort_file>: stop the training job on request
    { set +x; } 2>/dev/null
    local pgid="$1" f="$2"
    while kill -0 "${pgid}" 2>/dev/null; do
        if [ -s "${f}" ]; then
            echo "[head] ABORT requested -> stopping training: $(head -c 600 "${f}")" >&2
            kill -TERM -- "-${pgid}" 2>/dev/null || true
            sleep "${ABORT_GRACE_S:-60}"
            kill -KILL -- "-${pgid}" 2>/dev/null || true
            return 0
        fi
        sleep "${ABORT_POLL_S:-30}"
    done
}

_driver_on_signal() {  # _driver_on_signal <name> <exit code>
    echo "[head] ${1} received: stopping the training driver" >&2
    if [ -n "${VOA_DRIVER_PGID:-}" ]; then kill -TERM -- "-${VOA_DRIVER_PGID}" 2>/dev/null || true; fi
    exit "${2}"
}

# shellcheck disable=SC2034  # DRIVER_RC is this function's result, read by the launchers
run_driver() {
    local log="$1" watchdog=""
    shift
    set -m
    (
        set +m
        "$@" 2>&1 | tee "${log}"
        exit "${PIPESTATUS[0]}"   # the driver's code, not tee's
    ) &
    VOA_DRIVER_PGID=$!
    set +m
    if [ -z "$(trap -p TERM)" ]; then
        trap '_driver_on_signal TERM 143' TERM
        trap '_driver_on_signal INT 130' INT
        trap '_driver_on_signal HUP 129' HUP
    fi
    if [ -n "${ABORT_FILE:-}" ]; then
        abort_watchdog "${VOA_DRIVER_PGID}" "${ABORT_FILE}" &
        watchdog=$!
    fi
    DRIVER_RC=0
    wait "${VOA_DRIVER_PGID}" || DRIVER_RC=$?
    if [ -n "${watchdog}" ]; then kill "${watchdog}" 2>/dev/null; wait "${watchdog}" 2>/dev/null; fi
    VOA_DRIVER_PGID=""
    return 0
}
