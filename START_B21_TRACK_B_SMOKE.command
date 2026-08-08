#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"
bash evidence_extension_v1/track_b/run_track_b_mac.sh smoke
