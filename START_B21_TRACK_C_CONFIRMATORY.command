#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
bash evidence_extension_v1/track_c/run_track_c_mac.sh confirmatory
