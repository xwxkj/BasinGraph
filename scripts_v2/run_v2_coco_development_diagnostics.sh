#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/BasinGraph202606/release/BasinGraph"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest

RUN_ID="$(
  cat results_v2/formal_development/coco_rc1/LAST_RUN_ID.txt
)"

PYTHONPATH="$PWD" \
python analysis_v2/analyze_coco_v2_development_diagnostics.py \
  --run-id "$RUN_ID"

echo "V2_COCO_DEVELOPMENT_DIAGNOSTICS_ALL_DONE"
echo "RUN_ID=$RUN_ID"
