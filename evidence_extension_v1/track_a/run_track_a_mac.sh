#!/bin/bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke|confirmatory) ;;
  *)
    echo "Usage: bash evidence_extension_v1/track_a/run_track_a_mac.sh [smoke|confirmatory]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${B21_TRACK_A_VENV:-${PROJECT_ROOT}/.venv-track-a}"
RESULT_ROOT="results_b21/track_a"
DELIVERY_ROOT="${B21_DELIVERY_DIR:-${HOME}/Downloads/B21_RESULTS_READY}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r \
  protocols/evidence_extension_v1/track_a/TRACK_A_REQUIREMENTS_LOCK.txt
python -m pip install -e . --no-deps

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -z "${B21_WORKERS:-}" ]; then
  if command -v sysctl >/dev/null 2>&1; then
    LOGICAL_CPUS="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 2)"
  else
    LOGICAL_CPUS="$(python - <<'PY'
import os
print(os.cpu_count() or 2)
PY
)"
  fi
  if [ "${LOGICAL_CPUS}" -le 2 ]; then
    B21_WORKERS=1
  elif [ "${LOGICAL_CPUS}" -le 5 ]; then
    B21_WORKERS=$((LOGICAL_CPUS - 1))
  else
    B21_WORKERS=4
  fi
fi
export B21_WORKERS

mkdir -p "${RESULT_ROOT}"

if [ "${MODE}" = "confirmatory" ]; then
  GATE="${RESULT_ROOT}/TRACK_A_SMOKE_GATE.json"
  if [ ! -f "${GATE}" ]; then
    echo "Track A confirmatory run is locked until the smoke gate exists:" >&2
    echo "  ${PROJECT_ROOT}/${GATE}" >&2
    echo "Run the smoke command first." >&2
    exit 1
  fi
  python - <<PY
import json
import subprocess
from pathlib import Path
gate = json.loads(Path("${GATE}").read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if gate.get("status") != "TRACK_A_SMOKE_GATE_OK":
    raise SystemExit("Track A smoke gate status is invalid.")
if gate.get("source_commit") != head:
    raise SystemExit(
        "Track A smoke gate belongs to a different source commit. "
        "Rerun smoke on the current frozen runner."
    )
PY
  RUN_ID_FILE="${RESULT_ROOT}/TRACK_A_CONFIRMATORY_RUN_ID.txt"
  if [ -f "${RUN_ID_FILE}" ]; then
    RUN_ID="$(tr -d '[:space:]' < "${RUN_ID_FILE}")"
  else
    RUN_ID="b21_track_a_confirmatory_$(date -u +%Y%m%dT%H%M%SZ)"
    printf '%s\n' "${RUN_ID}" > "${RUN_ID_FILE}"
  fi
  AUTH_FLAG="--authorize-confirmatory"
else
  RUN_ID="b21_track_a_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
  AUTH_FLAG=""
fi

printf '\n=== Repository manifest ===\n'
python scripts_v2/generate_repository_manifest.py

printf '\n=== Step 0 identity audit ===\n'
python scripts_v2/audit_step0_identity.py \
  --require-manifest \
  --output "${RESULT_ROOT}/identity_audit_${MODE}.json"

printf '\n=== Track A zero-evaluation preflight ===\n'
python evidence_extension_v1/track_a/preflight_track_a.py \
  --mode "${MODE}" \
  --base-seed 20260808 \
  ${AUTH_FLAG}

printf '\n=== Track A shard execution ===\n'
python evidence_extension_v1/track_a/run_track_a_all.py \
  --mode "${MODE}" \
  --workers "${B21_WORKERS}" \
  --run-id "${RUN_ID}" \
  --base-seed 20260808 \
  ${AUTH_FLAG}

printf '\n=== Track A validation and official cocopp ===\n'
python evidence_extension_v1/track_a/finalize_track_a.py \
  --mode "${MODE}" \
  --run-id "${RUN_ID}" \
  --base-seed 20260808 \
  ${AUTH_FLAG}

RUN_ROOT="${PROJECT_ROOT}/${RESULT_ROOT}/${RUN_ID}"
if [ "${MODE}" = "smoke" ]; then
  python - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
report = json.loads((root / "validation_report.json").read_text())
if report["status"] != "TRACK_A_FINAL_VALIDATION_OK":
    raise SystemExit("Smoke gate not satisfied.")
gate = {
    "status": "TRACK_A_SMOKE_GATE_OK",
    "run_id": "${RUN_ID}",
    "source_commit": report["identity"]["head_commit"],
    "source_identity_sha256": report["identity"]["source_identity_sha256"],
}
Path("${PROJECT_ROOT}/${RESULT_ROOT}/TRACK_A_SMOKE_GATE.json").write_text(
    json.dumps(gate, indent=2)
)
PY
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DELIVERY_DIR="${DELIVERY_ROOT}/trackA_${MODE}_${TIMESTAMP}"
mkdir -p "${DELIVERY_DIR}"

LIGHT_ZIP="B21_TRACK_A_${MODE}_${TIMESTAMP}.zip"
LIGHT_SHA="${LIGHT_ZIP}.sha256"
OBSERVER_TAR="B21_TRACK_A_${MODE}_observer_${TIMESTAMP}.tar.gz"
OBSERVER_SHA="${OBSERVER_TAR}.sha256"

if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --sequesterRsrc --keepParent \
    "${RUN_ROOT}" "${DELIVERY_DIR}/${LIGHT_ZIP}"
else
  python - <<PY
import shutil
from pathlib import Path
run_root = Path("${RUN_ROOT}")
out = Path("${DELIVERY_DIR}/${LIGHT_ZIP}")
shutil.make_archive(
    str(out.with_suffix("")),
    "zip",
    root_dir=str(run_root.parent),
    base_dir=run_root.name,
)
PY
fi
shasum -a 256 "${DELIVERY_DIR}/${LIGHT_ZIP}" \
  > "${DELIVERY_DIR}/${LIGHT_SHA}"

(
  cd "${PROJECT_ROOT}"
  tar -czf "${DELIVERY_DIR}/${OBSERVER_TAR}" \
    -T "${RUN_ROOT}/complete_observer_paths.txt"
)
shasum -a 256 "${DELIVERY_DIR}/${OBSERVER_TAR}" \
  > "${DELIVERY_DIR}/${OBSERVER_SHA}"

for NAME in \
  validation_report.json \
  variant_summary.csv \
  function_group_dimension_summary.csv \
  pairwise_final_values.csv \
  checkpoint_rank_summary.csv \
  phase_summary.csv \
  friedman_final_values.json \
  launch_metadata.json; do
  if [ -f "${RUN_ROOT}/${NAME}" ]; then
    cp -f "${RUN_ROOT}/${NAME}" "${DELIVERY_DIR}/${NAME}"
  fi
done

cat > "${DELIVERY_DIR}/OPEN_THIS_FOLDER.txt" <<EOF
B21 Track A ${MODE} completed successfully.

Upload these four files:
1. ${DELIVERY_DIR}/${LIGHT_ZIP}
2. ${DELIVERY_DIR}/${LIGHT_SHA}
3. ${DELIVERY_DIR}/${OBSERVER_TAR}
4. ${DELIVERY_DIR}/${OBSERVER_SHA}

Quick review:
- ${DELIVERY_DIR}/validation_report.json
- ${DELIVERY_DIR}/variant_summary.csv
- ${DELIVERY_DIR}/function_group_dimension_summary.csv
EOF

printf '\nTRACK_A_MAC_RUN_OK\n'
printf 'Mode: %s\n' "${MODE}"
printf 'Run ID: %s\n' "${RUN_ID}"
printf 'Delivery folder: %s\n' "${DELIVERY_DIR}"
printf 'Upload ZIP: %s\n' "${DELIVERY_DIR}/${LIGHT_ZIP}"
printf 'Upload observer archive: %s\n' "${DELIVERY_DIR}/${OBSERVER_TAR}"

if [ "$(uname -s)" = "Darwin" ] \
  && [ "${B21_NO_OPEN:-0}" != "1" ] \
  && command -v open >/dev/null 2>&1; then
  open "${DELIVERY_DIR}"
  open -R "${DELIVERY_DIR}/${LIGHT_ZIP}"
fi

printf '\nFinder has been opened at the exact result folder.\n'
