#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/BasinGraph202606

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-official

# Prevent numerical-library thread oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs official_results processed_results protocols

RUN_ID=step12c_corrected_coco_d2351020_f1_24_i1_15_$(date +%Y%m%d_%H%M%S)
export RUN_ID

echo "${RUN_ID}" > official_results/STEP12C_LAST_RUN_ID.txt

echo "============================================================"
echo "Step 12C corrected official COCO/BBOB validation"
echo "RUN_ID: ${RUN_ID}"
echo "Dimensions: 2,3,5,10,20"
echo "Functions: 1-24"
echo "Instances: 1-15"
echo "Budget multiplier: 1000"
echo "Algorithms:"
echo "  BasinGraph"
echo "  CMA-ES via pycma"
echo "  true BIPOP-CMA-ES via pycma bipop=True"
echo "  Differential Evolution"
echo "  Multi-start L-BFGS-B"
echo "  Latin Hypercube Sampling"
echo "  Random Search"
echo "Expected result rows: 12600 + header"
echo "============================================================"

# ------------------------------------------------------------
# Freeze the software environment used for this final run.
# ------------------------------------------------------------
python - <<'PY' > "protocols/${RUN_ID}_environment.txt"
import platform
import sys

import cma
import cocoex
import cocopp
import matplotlib
import numpy
import pandas
import scipy

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("pycma:", getattr(cma, "__version__", "unknown"))
print("cocoex:", getattr(cocoex, "__version__", "unknown"))
print("cocopp:", getattr(cocopp, "__version__", "unknown"))
PY

python -m pip freeze > "protocols/${RUN_ID}_requirements-lock.txt"

# ------------------------------------------------------------
# Run all seven algorithms.
# ------------------------------------------------------------
python experiments/run_coco.py \
    --algorithm all-core \
    --dimensions 2,3,5,10,20 \
    --functions 1-24 \
    --instances 1-15 \
    --budget-multiplier 1000 \
    --result-folder "${RUN_ID}" \
    --summary-csv "processed_results/${RUN_ID}.csv" \
    2>&1 | tee "logs/${RUN_ID}_run.log"

# ------------------------------------------------------------
# Validate the summary CSV before cocopp post-processing.
# ------------------------------------------------------------
python - <<'PY' 2>&1 | tee "logs/${RUN_ID}_validation.log"
import os
from pathlib import Path

import pandas as pd

run_id = os.environ["RUN_ID"]
csv_path = Path("processed_results") / f"{run_id}.csv"
df = pd.read_csv(csv_path)

expected_algorithms = {
    "BasinGraph",
    "CMA-ES",
    "BIPOP-CMA-ES",
    "Differential Evolution",
    "Multi-start L-BFGS-B",
    "Latin Hypercube Sampling",
    "Random Search",
}

assert len(df) == 12600, f"Expected 12600 rows, found {len(df)}"
assert set(df["algorithm"].unique()) == expected_algorithms
assert sorted(df["dimension"].unique().tolist()) == [2, 3, 5, 10, 20]
assert df["problem_id"].nunique() == 1800

messages = df["message"].astype(str)
assert not messages.str.contains(
    "exception", case=False, na=False
).any(), "At least one run reported an exception."

for algorithm in ["BasinGraph", "BIPOP-CMA-ES"]:
    sub = df[df["algorithm"] == algorithm]
    ratio = sub["nfe"] / sub["budget"]

    print(
        algorithm,
        "rows=", len(sub),
        "min_budget_ratio=", ratio.min(),
        "median_budget_ratio=", ratio.median(),
        "max_budget_ratio=", ratio.max(),
    )

    assert ratio.min() >= 0.99, (
        f"{algorithm} contains runs using less than 99% of the budget."
    )

print("Rows:", len(df))
print("Algorithms:", sorted(df["algorithm"].unique()))
print("Dimensions:", sorted(df["dimension"].unique()))
print("Unique problem IDs:", df["problem_id"].nunique())
print("STEP_12C_CSV_VALIDATION_OK")
PY

# ------------------------------------------------------------
# Inspect official COCO observer output.
# ------------------------------------------------------------
find exdata -maxdepth 1 -type d -name "${RUN_ID}*" \
    | sort \
    | tee "logs/${RUN_ID}_folders.txt"

INFO_COUNT=$(find exdata -path "*/${RUN_ID}*" \
    -type f -name "*.info" | wc -l | tr -d ' ')

DAT_COUNT=$(find exdata -path "*/${RUN_ID}*" \
    -type f -name "*.dat" | wc -l | tr -d ' ')

TOTAL_COUNT=$((INFO_COUNT + DAT_COUNT))

{
    echo "info_files=${INFO_COUNT}"
    echo "dat_files=${DAT_COUNT}"
    echo "info_plus_dat=${TOTAL_COUNT}"
} | tee "logs/${RUN_ID}_file_count.txt"

# Record all formal algId labels from the .info files.
find exdata -path "*/${RUN_ID}*" \
    -type f -name "*.info" \
    -exec grep -H "algId" {} \; \
    > "logs/${RUN_ID}_algorithm_labels.txt"

# ------------------------------------------------------------
# Official cocopp post-processing.
# ------------------------------------------------------------
python -m cocopp exdata/${RUN_ID}_* \
    2>&1 | tee "logs/${RUN_ID}_cocopp.log"

grep -q "ALL done" "logs/${RUN_ID}_cocopp.log"

# The run becomes the official final core run only after all checks pass.
echo "${RUN_ID}" > official_results/OFFICIAL_FINAL_CORE_RUN_ID.txt

echo "============================================================"
echo "STEP_12C_OK"
echo "RUN_ID=${RUN_ID}"
echo "Summary CSV: processed_results/${RUN_ID}.csv"
echo "COCO folders: exdata/${RUN_ID}_*"
echo "Environment: protocols/${RUN_ID}_environment.txt"
echo "Requirements: protocols/${RUN_ID}_requirements-lock.txt"
echo "============================================================"
