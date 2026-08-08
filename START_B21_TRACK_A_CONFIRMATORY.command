#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"
bash evidence_extension_v1/track_a/run_track_a_mac.sh confirmatory
