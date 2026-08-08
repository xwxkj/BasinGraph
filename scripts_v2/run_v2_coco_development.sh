#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606/release/BasinGraph"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_ID="${1:-v2rc1_coco_development_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="results_v2/formal_development/coco_rc1"
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
LOG_ROOT="logs_v2/${RUN_ID}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

echo "$RUN_ID" > "${RESULT_ROOT}/LAST_RUN_ID.txt"

PYTHONPATH="$PWD" \
python experiments_v2/run_coco_v2_development.py \
  --run-id "$RUN_ID" \
  2>&1 | tee "${LOG_ROOT}/run.log"

PYTHONPATH="$PWD" \
python analysis_v2/validate_coco_v2_development.py \
  --run-id "$RUN_ID" \
  2>&1 | tee "${LOG_ROOT}/validation.log"

EXDATA_ROOT="exdata/routeb_v2_development/${RUN_ID}"

python -m cocopp \
  "${EXDATA_ROOT}/BasinGraph_v2" \
  "${EXDATA_ROOT}/CMA_ES" \
  "${EXDATA_ROOT}/BIPOP_CMA_ES" \
  "${EXDATA_ROOT}/DE" \
  "${EXDATA_ROOT}/MS_LBFGSB" \
  "${EXDATA_ROOT}/LHS" \
  "${EXDATA_ROOT}/Random" \
  2>&1 | tee "${LOG_ROOT}/cocopp.log"

grep -q "ALL done" "${LOG_ROOT}/cocopp.log"

COCOPP_OUTPUT="$(
  grep "Output data written to folder" "${LOG_ROOT}/cocopp.log" \
  | tail -1 \
  | sed 's/^.*folder //'
)"
printf '%s\n' "$COCOPP_OUTPUT" > "${RUN_ROOT}/cocopp_output_path.txt"

# Archive large development artifacts locally. These archives remain ignored
# by Git and are not manuscript evidence.
tar -czf "${RUN_ROOT}/BasinGraph_v2_details.tar.gz" \
  -C "${RUN_ROOT}" details

tar -czf "${RUN_ROOT}/COCO_observer_logs.tar.gz" \
  -C "exdata/routeb_v2_development" "${RUN_ID}"

cat > "${RUN_ROOT}/DEVELOPMENT_ONLY_NOTICE.md" <<EOF
# Development-only COCO result

Run ID: ${RUN_ID}

This run used COCO instances 1-3 only. It may be inspected for algorithm
development. It is not prospective holdout evidence and must not be reported
as the final v2.0.0 benchmark.
EOF

echo "V2_COCO_DEVELOPMENT_ALL_DONE"
echo "RUN_ID=${RUN_ID}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "COCOPP_OUTPUT=${COCOPP_OUTPUT}"
