#!/usr/bin/env bash
# Stage the OfficeQA corpus + build the local BM25 chunk index on the CURRENT node.
#
# Used by eval (via serve_and_eval.sh) and by the training Rollouter. It is
# idempotent: existing corpus/chunks are reused unless OFFICEQA_FORCE_STAGE=1.
set -euo pipefail

OQ_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OQ_ZIP="${OFFICEQA_CORPUS_ZIP:-/Volumes/main/mshtelma/verl/data/officeqa/treasury_bulletins_clean.zip}"
OQ_UNZIP="${OFFICEQA_UNZIP_DIR:-/local_disk0/officeqa_corpus}"
OQ_CHUNKS_DST="${OFFICEQA_CHUNKS:-/local_disk0/officeqa/chunks.jsonl}"

if [ "${OFFICEQA_FORCE_STAGE:-0}" != "1" ] \
   && [ -n "$(find "${OQ_UNZIP}" -name 'treasury_bulletin_*.txt' -print -quit 2>/dev/null)" ] \
   && [ -s "${OQ_CHUNKS_DST}" ]; then
  DOC="$(find "${OQ_UNZIP}" -name 'treasury_bulletin_*.txt' -print -quit)"
  export OFFICEQA_CORPUS_DIR="$(dirname "${DOC}")"
  export OFFICEQA_CHUNKS="${OQ_CHUNKS_DST}"
  echo "[officeqa-stage] reuse corpus=${OFFICEQA_CORPUS_DIR} chunks=${OFFICEQA_CHUNKS}"
  return 0 2>/dev/null || exit 0
fi

echo "[officeqa-stage] staging OfficeQA data (deps + clean corpus + chunks) ..."
if python3 -m pip install --no-input bm25s pandas scipy >/tmp/pip_oqa.log 2>&1; then
  echo "[officeqa-stage] deps installed (bm25s=$(python3 -c 'import bm25s;print(bm25s.__version__)' 2>/dev/null||echo ?) pandas=$(python3 -c 'import pandas;print(pandas.__version__)' 2>/dev/null||echo ?))"
else
  echo "[officeqa-stage] WARN dep install issues; tail /tmp/pip_oqa.log:" >&2
  tail -20 /tmp/pip_oqa.log >&2 || true
fi

mkdir -p "${OQ_UNZIP}" "$(dirname "${OQ_CHUNKS_DST}")"
[ -f "${OQ_ZIP}" ] || { echo "[officeqa-stage] FATAL: corpus zip not found at ${OQ_ZIP}" >&2; exit 1; }
echo "[officeqa-stage] extracting ${OQ_ZIP} -> ${OQ_UNZIP}"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${OQ_ZIP}" "${OQ_UNZIP}"
DOC="$(find "${OQ_UNZIP}" -name 'treasury_bulletin_*.txt' -print -quit)"
[ -n "${DOC}" ] || { echo "[officeqa-stage] FATAL: no treasury_bulletin_*.txt under ${OQ_UNZIP}" >&2; exit 1; }
export OFFICEQA_CORPUS_DIR="$(dirname "${DOC}")"
export OFFICEQA_CHUNKS="${OQ_CHUNKS_DST}"
echo "[officeqa-stage] corpus dir: ${OFFICEQA_CORPUS_DIR} ($(ls "${OFFICEQA_CORPUS_DIR}"/*.txt 2>/dev/null | wc -l) docs)"

echo "[officeqa-stage] building chunks -> ${OFFICEQA_CHUNKS}"
OFFICEQA_CORPUS_DIR="${OFFICEQA_CORPUS_DIR}" PYTHONPATH="${OQ_SCRIPTS}/officeqa" \
  python3 "${OQ_SCRIPTS}/officeqa/build_chunks.py" --out "${OFFICEQA_CHUNKS}"
echo "[officeqa-stage] chunks: $(wc -l < "${OFFICEQA_CHUNKS}") lines"
