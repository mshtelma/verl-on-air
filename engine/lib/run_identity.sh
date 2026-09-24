#!/usr/bin/env bash
# shellcheck shell=bash
# =============================================================================
# Run identity + resume intent, shared by both training launchers.
#
# Every run writes to <output_dir>/<RUN_ID>/, so re-running a job file starts a NEW
# run -- instead of verl's default trainer.resume_mode=auto silently resuming whatever
# checkpoint sits in output_dir (the reason an eval once pointed at a step a fresh run
# never produces). Continuing a run is an explicit choice:
#   RESUME=never  (default)     start fresh; refuse if <output_dir>/<RUN_ID>/ has checkpoints
#   RESUME=auto                 continue <output_dir>/<RUN_ID>/ from its latest checkpoint
#   RESUME=<.../global_step_N>  start from that checkpoint
#   MAX_CKPT_TO_KEEP=<n>        optional: verl deletes checkpoints beyond the n newest.
#                               Default: keep all, so several can be evaluated.
#
#   source engine/lib/run_identity.sh
#   resolve_run_identity "${OUTPUT_DIR}" || exit 1   # sets RUN_ID, CKPT_DIR, IDENTITY_ARGS
# =============================================================================

resolve_run_identity() {
    local root="$1"
    if [ -z "${RUN_ID:-}" ]; then
        if [ "${DRY_RUN:-0}" = "1" ]; then
            RUN_ID="dryrun"
        else
            echo "FATAL: RUN_ID is not set. Every training run writes to <output_dir>/<RUN_ID>/;" \
                 "submit with make (it sets one), or add" \
                 "--override env_variables.RUN_ID=\$(date -u +%Y%m%dT%H%M%SZ)" >&2
            return 1
        fi
    fi
    if ! [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "FATAL: RUN_ID=${RUN_ID} must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2
        return 1
    fi
    CKPT_DIR="${root%/}/${RUN_ID}"
    RESUME="${RESUME:-never}"
    case "${RESUME}" in
        never) IDENTITY_ARGS=(trainer.resume_mode=disable) ;;
        auto)  IDENTITY_ARGS=(trainer.resume_mode=auto) ;;
        */global_step_*)
            IDENTITY_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME}") ;;
        *) echo "FATAL: RESUME=${RESUME}: expected never | auto | <path>/global_step_N" >&2
           return 1 ;;
    esac
    if [ "${RESUME}" = "never" ] && [ "${DRY_RUN:-0}" != "1" ] \
       && compgen -G "${CKPT_DIR}/global_step_*" >/dev/null; then
        echo "FATAL: ${CKPT_DIR} already holds checkpoints of run ${RUN_ID}. Continue it with" \
             "RESUME=auto, or start a new run under a new RUN_ID." >&2
        return 1
    fi
    if [ -n "${MAX_CKPT_TO_KEEP:-}" ]; then
        if ! [[ "${MAX_CKPT_TO_KEEP}" =~ ^[1-9][0-9]*$ ]]; then
            echo "FATAL: MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP}: expected a positive integer" >&2
            return 1
        fi
        IDENTITY_ARGS+=(trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}")
    fi
    export RUN_ID RESUME
}
