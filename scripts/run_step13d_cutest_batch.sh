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

SEED_START="${1:-0}"
SEED_COUNT="${2:-10}"
WORKERS="${3:-4}"

mkdir -p logs cutest_results/protocol_v2

LOG="logs/step13d_cutest_seed$((SEED_START + 1))_to_$((SEED_START + SEED_COUNT)).log"

caffeinate -dimsu nice -n 10 \
  python scripts/step13d_run_cutest_benchmark.py \
    --seed-start "$SEED_START" \
    --seed-count "$SEED_COUNT" \
    --workers "$WORKERS" \
    --budget-multiplier 50 \
    --min-budget 1000 \
    --max-budget 20000 \
    2>&1 | tee "$LOG"
