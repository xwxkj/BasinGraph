#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606/release/BasinGraph"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest
CUTEST_ENV_FILE="${CUTEST_ENV_FILE:-protocols/cutest_env.sh}"
source "$CUTEST_ENV_FILE"

RUN_ID="${1:-$(cat results_v2/formal_holdout/cutest_rc1/LAST_RUN_ID.txt)}"

PYTHONPATH="$PWD" \
python analysis_v2/validate_cutest_rc1_holdout.py \
  --run-id "$RUN_ID"

echo "V2_RC1_CUTEST_HOLDOUT_ALL_DONE"
echo "RUN_ID=${RUN_ID}"
