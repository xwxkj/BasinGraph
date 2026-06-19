#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest
source protocols/cutest_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

python scripts/step13e_analyze_cutest.py \
  2>&1 | tee logs/step13e_analyze_cutest.log
