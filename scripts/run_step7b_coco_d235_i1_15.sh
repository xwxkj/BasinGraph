#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/BasinGraph202606

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-official

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs official_results processed_results

RUN_ID=step7b_coco_d235_f1_24_i1_15_$(date +%Y%m%d_%H%M%S)
echo "${RUN_ID}" | tee official_results/STEP7B_LAST_RUN_ID.txt

echo "============================================================"
echo "Step 7B official COCO/BBOB subset"
echo "RUN_ID: ${RUN_ID}"
echo "Dimensions: 2,3,5"
echo "Functions: 1-24"
echo "Instances: 1-15"
echo "Budget multiplier: 1000"
echo "Algorithms: all-core"
echo "============================================================"

python experiments/run_coco.py \
    --algorithm all-core \
    --dimensions 2,3,5 \
    --functions 1-24 \
    --instances 1-15 \
    --budget-multiplier 1000 \
    --result-folder "${RUN_ID}" \
    --summary-csv "processed_results/${RUN_ID}.csv" \
    2>&1 | tee "logs/${RUN_ID}_run.log"

echo "============================================================"
echo "Checking official COCO output files"
echo "============================================================"

find exdata -maxdepth 1 -type d -name "${RUN_ID}*" | sort | tee "logs/${RUN_ID}_folders.txt"

INFO_DAT_COUNT=$(find exdata -path "*/${RUN_ID}*" -type f \( -name "*.info" -o -name "*.dat" \) | wc -l | tr -d ' ')
echo "Number of .info/.dat files: ${INFO_DAT_COUNT}" | tee "logs/${RUN_ID}_file_count.txt"

echo "============================================================"
echo "Running cocopp post-processing"
echo "============================================================"

python -m cocopp exdata/${RUN_ID}_* 2>&1 | tee "logs/${RUN_ID}_cocopp.log"

echo "============================================================"
echo "STEP_7B_OK"
echo "RUN_ID=${RUN_ID}"
echo "Summary CSV: processed_results/${RUN_ID}.csv"
echo "COCO folders: exdata/${RUN_ID}_*"
echo "============================================================"
