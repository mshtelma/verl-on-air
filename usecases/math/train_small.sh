#!/usr/bin/env bash
# Keep per-rollout learning evidence and export it after a certified training run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}/../.."
# shellcheck source=../../engine/lib/hparams.sh
source "${ROOT}/engine/lib/hparams.sh"
export MATH_TRACE_DIR="${MATH_ARTIFACT_ROOT:?}/${RUN_ID:?}/reward-traces"
bash "${ROOT}/engine/train/dispatch_agentic.sh"
if [ "${DRY_RUN:-0}" != "1" ]; then
  checkpoint_dir="$(hp output_dir)/${RUN_ID}"
  python3 "${HERE}/export_run.py" --checkpoint-dir "${checkpoint_dir}" --reward-dir "${MATH_TRACE_DIR}"
fi
