#!/usr/bin/env bash
# Stage/probe OfficeQA on every training/rollout node before the shared launcher.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}/../.."
export OQ_TRACE_DIR="${OQ_ARTIFACT_ROOT:?}/${RUN_ID:?}/reward-traces"
export OQ_CALIBRATION_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/calibration.json"
export OQ_STAGE_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/stage-${POD_RANK:-0}.json"
if [ "${DRY_RUN:-0}" != "1" ]; then
  python3 "${HERE}/stage.py"
fi
exec bash "${ROOT}/engine/train/dispatch_agentic.sh"
