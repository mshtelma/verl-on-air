#!/usr/bin/env bash
# Actor and judge share ONE AIR job: df1 has no cross-job networking.
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
if [ "${NUM_NODES:?}" != "2" ]; then
  echo "OfficeQA eval requires one actor node and one judge node." >&2
  exit 2
fi
if [ "${POD_RANK:?}" = "1" ]; then
  export JUDGE_NNODES=1 JUDGE_NODE_RANK=0
  export JUDGE_RENDEZVOUS="${RDV}/judge_endpoint"
  export JUDGE_EXIT_SENTINEL="${RDV}/training_done"
  exec bash "${ROOT}/engine/serve/serve_judge.sh"
fi
rm -f "${RDV}/training_done" "${RDV}/ABORT.json"
trap 'rdv_put "${RDV}/training_done" "done"' EXIT
python3 "${HERE}/stage.py"
JUDGE_BASE_URL="$(rdv_wait "${RDV}/judge_endpoint" "$(( JUDGE_STAGE_TIMEOUT + JUDGE_HEALTH_TIMEOUT + 300 ))")"
export JUDGE_BASE_URL
export JUDGE_ENDPOINT_FILE="${RDV}/judge_endpoint"
python3 "${HERE}/judge_selfcheck.py"
bash "${ROOT}/engine/serve/serve_and_eval.sh"
