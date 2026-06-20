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

PYTHONPATH="$PWD" python experiments_v2/run_v2_ablation_mini.py \
  --workers "${1:-4}" \
  --seed-count 5 \
  --force

PYTHONPATH="$PWD" python analysis_v2/validate_v2_ablation_mini.py
