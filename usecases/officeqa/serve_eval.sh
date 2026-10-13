#!/usr/bin/env bash
# A self-hosted judge shares the AIR job; a workspace judge needs only the actor.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE}/../.."
source "${ROOT}/engine/lib/rendezvous.sh"
RDV="${RENDEZVOUS_ROOT:?}/${RUN_ID:?}"
export VOA_RDV_DIR="${RDV}"
export PYTHONPATH="${HERE}:${ROOT}/engine/train:${ROOT}/engine/lib:${ROOT}/engine/serve${PYTHONPATH:+:${PYTHONPATH}}"
export OQ_TRACE_DIR="${OQ_ARTIFACT_ROOT:?}/${RUN_ID}/reward-traces"
export OQ_CALIBRATION_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/calibration.json"
export OQ_STAGE_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/stage-${POD_RANK:-0}.json"
export EVAL_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/eval.json"
export EVAL_TRACE_OUT="${OQ_ARTIFACT_ROOT}/${RUN_ID}/episodes.jsonl"
mkdir -p "${RDV}"
JUDGE_NODES="${JUDGE_NODES:-1}"
if [ "${NUM_NODES:?}" -ne "$(( 1 + JUDGE_NODES ))" ] || [ "${JUDGE_NODES}" -gt 1 ]; then
  echo "OfficeQA eval requires one actor node and zero or one judge node." >&2
  exit 2
fi
if [ "${JUDGE_NODES}" = "1" ] && [ "${POD_RANK:?}" = "1" ]; then
  export JUDGE_NNODES=1 JUDGE_NODE_RANK=0
  export JUDGE_RENDEZVOUS="${RDV}/judge_endpoint"
  export JUDGE_EXIT_SENTINEL="${RDV}/training_done"
  exec bash "${ROOT}/engine/serve/serve_judge.sh"
fi
rm -f "${RDV}/training_done" "${RDV}/ABORT.json"
trap 'rdv_put "${RDV}/training_done" "done"' EXIT
python3 "${HERE}/stage.py"
if [ "${JUDGE_NODES}" = "1" ]; then
  JUDGE_BASE_URL="$(rdv_wait "${RDV}/judge_endpoint" "$(( JUDGE_STAGE_TIMEOUT + JUDGE_HEALTH_TIMEOUT + 300 ))")"
  export JUDGE_BASE_URL
  export JUDGE_ENDPOINT_FILE="${RDV}/judge_endpoint"
elif [ "${JUDGE_PROVIDER:-}" != "databricks" ]; then
  echo "OfficeQA eval without judge GPUs requires JUDGE_PROVIDER=databricks." >&2
  exit 2
fi
python3 "${HERE}/judge_selfcheck.py"
bash "${ROOT}/engine/serve/serve_and_eval.sh"
