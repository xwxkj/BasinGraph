#!/bin/bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke|pilot|full-mini) ;;
  *)
    echo "Usage: bash evidence_extension_v1/run_b21_mac.sh [smoke|pilot|full-mini]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${B21_VENV_DIR:-${PROJECT_ROOT}/.venv-b21}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Creating B21 virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[test]'

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

OUTPUT="results_b21/mac_${MODE}_ablation"
AUDIT_OUTPUT="results_b21/step0/identity_audit_${MODE}.json"

printf '\n=== B21 Step 0 repository manifest ===\n'
python scripts_v2/generate_repository_manifest.py

printf '\n=== B21 Step 0 identity audit ===\n'
python scripts_v2/audit_step0_identity.py \
  --require-manifest \
  --output "${AUDIT_OUTPUT}"

printf '\n=== B21 %s run ===\n' "${MODE}"
python evidence_extension_v1/run_b21_mac_smoke_ablation.py \
  --mode "${MODE}" \
  --workers "${B21_WORKERS}" \
  --output "${OUTPUT}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RETURN_ZIP="B21_RETURN_${MODE}_${TIMESTAMP}.zip"
RETURN_SHA="${RETURN_ZIP}.sha256"

if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --sequesterRsrc --keepParent "${OUTPUT}" "${RETURN_ZIP}"
else
  python - <<PY
import shutil
shutil.make_archive('${RETURN_ZIP%.zip}', 'zip', root_dir='${OUTPUT%/*}', base_dir='${OUTPUT##*/}')
PY
fi

shasum -a 256 "${RETURN_ZIP}" > "${RETURN_SHA}"

printf '\nB21_MAC_RUN_OK\n'
printf 'Mode: %s\n' "${MODE}"
printf 'Workers: %s\n' "${B21_WORKERS}"
printf 'Results: %s\n' "${OUTPUT}"
printf 'Return archive: %s\n' "${RETURN_ZIP}"
printf 'Checksum: %s\n' "${RETURN_SHA}"
printf '\nUpload both the ZIP and SHA-256 file for audit.\n'
