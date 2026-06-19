#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Documents/BasinGraph202606"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest
source protocols/cutest_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs protocols processed_results

python scripts/step13c_v2b_freeze_cutest_protocol.py \
  2> logs/step13c_v2b_compile_warnings.log \
  | tee logs/step13c_v2b_freeze_protocol.log
