#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606/release/BasinGraph"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest
CUTEST_ENV_FILE="${CUTEST_ENV_FILE:-protocols/cutest_env.sh}"
source "$CUTEST_ENV_FILE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SEED_START="${1:?Usage: $0 SEED_START SEED_COUNT [WORKERS] [RUN_ID]}"
SEED_COUNT="${2:?Usage: $0 SEED_START SEED_COUNT [WORKERS] [RUN_ID]}"
WORKERS="${3:-4}"
RUN_ID="${4:-$(cat results_v2/formal_holdout/cutest_rc1/LAST_RUN_ID.txt 2>/dev/null || true)}"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="v2rc1_cutest_holdout_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p results_v2/formal_holdout/cutest_rc1
  printf '%s\n' "$RUN_ID" \
    > results_v2/formal_holdout/cutest_rc1/LAST_RUN_ID.txt
fi

PYTHONPATH="$PWD" \
python experiments_v2/run_cutest_rc1_holdout_batch.py \
  --seed-start "$SEED_START" \
  --seed-count "$SEED_COUNT" \
  --workers "$WORKERS" \
  --run-id "$RUN_ID"

echo "CUTEST_HOLDOUT_BATCH_ALL_DONE"
echo "RUN_ID=${RUN_ID}"
echo "SEED_START=${SEED_START}"
echo "SEED_COUNT=${SEED_COUNT}"
