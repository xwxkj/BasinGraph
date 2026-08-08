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

WORKERS="${1:-4}"
RUN_ID="${2:-v2rc1_coco_holdout_$(date -u +%Y%m%dT%H%M%SZ)}"

PYTHONPATH="$PWD" \
python scripts_v2/run_rc1_coco_holdout_all.py \
  --workers "$WORKERS" \
  --run-id "$RUN_ID"

PYTHONPATH="$PWD" \
python analysis_v2/validate_coco_rc1_holdout.py \
  --run-id "$RUN_ID"

EXDATA_ROOT="exdata/routeb_v2_holdout/${RUN_ID}"
COCOPP_OUTPUT="$ROOT/ppdata/rc1_holdout_${RUN_ID}"
LOG_ROOT="logs_v2/${RUN_ID}"
RUN_ROOT="results_v2/formal_holdout/coco_rc1/${RUN_ID}"

mkdir -p "$LOG_ROOT"
rm -rf "$COCOPP_OUTPUT"

python -m cocopp \
  -o "$COCOPP_OUTPUT" \
  "${EXDATA_ROOT}/BasinGraph_v2" \
  "${EXDATA_ROOT}/CMA_ES" \
  "${EXDATA_ROOT}/BIPOP_CMA_ES" \
  "${EXDATA_ROOT}/DE" \
  "${EXDATA_ROOT}/MS_LBFGSB" \
  "${EXDATA_ROOT}/LHS" \
  "${EXDATA_ROOT}/Random" \
  > "${LOG_ROOT}/cocopp.log" 2>&1

grep -q "ALL done" "${LOG_ROOT}/cocopp.log"
printf '%s\n' "$COCOPP_OUTPUT" > "${RUN_ROOT}/cocopp_output_path.txt"

tar -czf "${RUN_ROOT}/BasinGraph_holdout_details.tar.gz" \
  -C "${RUN_ROOT}/BasinGraph_v2" details

tar -czf "${RUN_ROOT}/COCO_holdout_observer_logs.tar.gz" \
  -C "exdata/routeb_v2_holdout" "${RUN_ID}"

cat > "${RUN_ROOT}/HOLDOUT_RECORD.md" <<EOF
# Prospective COCO holdout record

Run ID: ${RUN_ID}

- functions: 1-24
- dimensions: 2, 3, 5, 10, 20
- instances: 4-15
- budget: 1000d
- algorithms: 7
- selected implementation: BasinGraph 2.0.0-rc1
- options hash:
  031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69

The holdout was opened only after the final-candidate and runner-freeze tags
were created.
EOF

echo "V2_RC1_COCO_HOLDOUT_ALL_DONE"
echo "RUN_ID=${RUN_ID}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "COCOPP_OUTPUT=${COCOPP_OUTPUT}"
