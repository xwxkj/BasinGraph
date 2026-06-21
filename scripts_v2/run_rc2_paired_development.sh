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

RC1_ROOT="results_v2/formal_development/coco_rc1"
RC1_RUN_ID="$(cat "${RC1_ROOT}/LAST_RUN_ID.txt")"

RUN_ID="${1:-v2rc2_coco_development_$(date -u +%Y%m%dT%H%M%SZ)}"
RC2_ROOT="results_v2/formal_development/coco_rc2"
RUN_ROOT="${RC2_ROOT}/${RUN_ID}"
LOG_ROOT="logs_v2/${RUN_ID}"

mkdir -p "$RC2_ROOT" "$LOG_ROOT"
echo "$RUN_ID" > "${RC2_ROOT}/LAST_RUN_ID.txt"

PYTHONPATH="$PWD" \
python experiments_v2/run_coco_rc2_development.py \
  --run-id "$RUN_ID" \
  2>&1 | tee "${LOG_ROOT}/run.log"

PYTHONPATH="$PWD" \
python analysis_v2/validate_coco_rc2_development.py \
  --run-id "$RUN_ID" \
  2>&1 | tee "${LOG_ROOT}/validation.log"

PYTHONPATH="$PWD" \
python analysis_v2/compare_rc1_rc2_development.py \
  --rc2-run-id "$RUN_ID" \
  2>&1 | tee "${LOG_ROOT}/comparison.log"

RC1_EXDATA="exdata/routeb_v2_development/${RC1_RUN_ID}"
RC2_EXDATA="exdata/routeb_v2_rc2_development/${RUN_ID}"

for folder in \
  "${RC1_EXDATA}/BasinGraph_v2" \
  "${RC1_EXDATA}/CMA_ES" \
  "${RC1_EXDATA}/BIPOP_CMA_ES" \
  "${RC1_EXDATA}/DE" \
  "${RC1_EXDATA}/MS_LBFGSB" \
  "${RC1_EXDATA}/LHS" \
  "${RC1_EXDATA}/Random" \
  "${RC2_EXDATA}/BasinGraph_v2_rc2"
do
  test -d "$folder"
done

COCOPP_OUTPUT="$ROOT/ppdata/rc1_vs_rc2_${RUN_ID}"
rm -rf "$COCOPP_OUTPUT"

python -m cocopp \
  -o "$COCOPP_OUTPUT" \
  "${RC1_EXDATA}/BasinGraph_v2" \
  "${RC2_EXDATA}/BasinGraph_v2_rc2" \
  "${RC1_EXDATA}/CMA_ES" \
  "${RC1_EXDATA}/BIPOP_CMA_ES" \
  "${RC1_EXDATA}/DE" \
  "${RC1_EXDATA}/MS_LBFGSB" \
  "${RC1_EXDATA}/LHS" \
  "${RC1_EXDATA}/Random" \
  2>&1 | tee "${LOG_ROOT}/cocopp.log"

grep -q "ALL done" "${LOG_ROOT}/cocopp.log"
printf '%s\n' "$COCOPP_OUTPUT" > "${RUN_ROOT}/cocopp_output_path.txt"

PYTHONPATH="$PWD" \
python scripts_v2/package_rc1_rc2_cocopp_review.py \
  --rc2-run-id "$RUN_ID" \
  --cocopp-output "$COCOPP_OUTPUT" \
  2>&1 | tee "${LOG_ROOT}/package.log"

tar -czf "${RUN_ROOT}/BasinGraph_rc2_details.tar.gz" \
  -C "${RUN_ROOT}" details

cat > "${RUN_ROOT}/DEVELOPMENT_ONLY_NOTICE.md" <<EOF
# Development-only rc1 versus rc2 comparison

rc1 run: ${RC1_RUN_ID}
rc2 run: ${RUN_ID}

Only COCO instances 1-3 were used. The prospective holdout remains unopened.
The final acceptance decision is pending review of the official combined
cocopp output under the pre-frozen gate.
EOF

echo "V2_RC2_PAIRED_DEVELOPMENT_ALL_DONE"
echo "RC1_RUN_ID=${RC1_RUN_ID}"
echo "RC2_RUN_ID=${RUN_ID}"
echo "COCOPP_OUTPUT=${COCOPP_OUTPUT}"
echo "REVIEW_ARCHIVE=results_v2/BasinGraph_rc1_vs_rc2_COCO_development_review_${RUN_ID}.tar.gz"
