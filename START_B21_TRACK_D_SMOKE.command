#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
exec bash evidence_extension_v1/track_d/run_track_d_mac.sh smoke
