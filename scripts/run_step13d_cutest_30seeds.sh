#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606"
cd "$ROOT"

bash scripts/run_step13d_cutest_batch.sh 0 10 "${1:-4}"
bash scripts/run_step13d_cutest_batch.sh 10 10 "${1:-4}"
bash scripts/run_step13d_cutest_batch.sh 20 10 "${1:-4}"

echo "STEP_13D_30SEEDS_OK"
