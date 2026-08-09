#!/bin/bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke|confirmatory) ;;
  *) echo "Usage: bash evidence_extension_v1/track_c/run_track_c_mac.sh [smoke|confirmatory]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1 \
   && [ "${B21_TRACK_C_CAFFEINATED:-0}" != "1" ]; then
  export B21_TRACK_C_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0" "${MODE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${B21_TRACK_C_VENV_DIR:-${PROJECT_ROOT}/.venv-track-c}"
DELIVERY_ROOT="${B21_DELIVERY_DIR:-${HOME}/Downloads/B21_RESULTS_READY}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r protocols/evidence_extension_v1/track_c/TRACK_C_REQUIREMENTS_LOCK.txt
python -m pip install -e . --no-deps

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IDENTITY_AUDIT="results_b21/track_c/identity_audit_${MODE}_${TIMESTAMP}.json"
PREFLIGHT="results_b21/track_c/preflight_${MODE}_${TIMESTAMP}.json"

python evidence_extension_v1/track_c/generate_protocol_manifest.py
python scripts_v2/generate_repository_manifest.py
python scripts_v2/audit_step0_identity.py --require-manifest --output "${IDENTITY_AUDIT}"

if [ "${MODE}" = "smoke" ]; then
  python evidence_extension_v1/track_c/preflight_track_c.py \
    --mode smoke --output "${PREFLIGHT}"
  RUN_ID="b21_track_c_smoke_${TIMESTAMP}"
  python evidence_extension_v1/track_c/run_track_c_all.py \
    --mode smoke --run-id "${RUN_ID}" --workers "${B21_TRACK_C_SMOKE_WORKERS:-2}"
  python evidence_extension_v1/track_c/task_specific_references.py \
    --mode smoke \
    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"
  python evidence_extension_v1/track_c/finalize_track_c.py \
    --mode smoke --run-id "${RUN_ID}"
  RUN_ROOT="results_b21/track_c/${RUN_ID}"
  cp -f "${IDENTITY_AUDIT}" "${RUN_ROOT}/identity_audit.json"
  cp -f "${PREFLIGHT}" "${RUN_ROOT}/preflight.json"
  cp -f results_b21/track_c/LOCAL_SMOKE_GATE.json "${RUN_ROOT}/LOCAL_SMOKE_GATE.json"
  DELIVERY_RUN_DIR="${DELIVERY_ROOT}/trackC_smoke_${TIMESTAMP}"
  RETURN_ZIP="B21_TRACK_C_smoke_${TIMESTAMP}.zip"
  RETURN_DETAILS="B21_TRACK_C_smoke_details_${TIMESTAMP}.tar.gz"
else
  python - <<'PY'
from pathlib import Path
import hashlib, json, subprocess
root = Path('.').resolve()
gate_path = root / 'results_b21/track_c/LOCAL_SMOKE_GATE.json'
identity_path = root / 'protocols/evidence_extension_v1/track_c/TRACK_C_SOURCE_IDENTITY.json'
if not gate_path.is_file():
    raise SystemExit('Missing Track C local smoke gate. Run smoke first.')
gate = json.loads(gate_path.read_text())
if gate.get('status') != 'TRACK_C_LOCAL_SMOKE_GATE_OK':
    raise SystemExit('Track C smoke gate is invalid.')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if gate.get('source_commit') != head:
    raise SystemExit(f"Smoke gate commit mismatch: {gate.get('source_commit')} != {head}")
if gate.get('source_identity_sha256') != hashlib.sha256(identity_path.read_bytes()).hexdigest():
    raise SystemExit('Smoke gate source-identity mismatch.')
if int(gate.get('confirmatory_objective_evaluations', -1)) != 0:
    raise SystemExit('Smoke gate reports confirmatory evaluations.')
print('TRACK_C_LOCAL_SMOKE_GATE_OK')
PY
  python evidence_extension_v1/track_c/preflight_track_c.py \
    --mode confirmatory --authorize-confirmatory --output "${PREFLIGHT}"
  RUN_ID="b21_track_c_confirmatory_${TIMESTAMP}"
  python evidence_extension_v1/track_c/run_track_c_all.py \
    --mode confirmatory --run-id "${RUN_ID}" \
    --workers "${B21_TRACK_C_CONFIRMATORY_WORKERS:-4}" --authorize-confirmatory
  python evidence_extension_v1/track_c/task_specific_references.py \
    --mode confirmatory --authorize-confirmatory \
    --output "results_b21/track_c/${RUN_ID}/task_specific_reference_results.csv"
  python evidence_extension_v1/track_c/finalize_track_c.py \
    --mode confirmatory --run-id "${RUN_ID}" --authorize-confirmatory
  RUN_ROOT="results_b21/track_c/${RUN_ID}"
  cp -f "${IDENTITY_AUDIT}" "${RUN_ROOT}/identity_audit.json"
  cp -f "${PREFLIGHT}" "${RUN_ROOT}/preflight.json"
  cp -f results_b21/track_c/LOCAL_SMOKE_GATE.json "${RUN_ROOT}/LOCAL_SMOKE_GATE.json"
  DELIVERY_RUN_DIR="${DELIVERY_ROOT}/trackC_confirmatory_${TIMESTAMP}"
  RETURN_ZIP="B21_TRACK_C_confirmatory_${TIMESTAMP}.zip"
  RETURN_DETAILS="B21_TRACK_C_confirmatory_details_${TIMESTAMP}.tar.gz"
fi

mkdir -p "${DELIVERY_RUN_DIR}"
if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --sequesterRsrc --keepParent "${RUN_ROOT}" "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
else
  python - <<PY
import shutil
shutil.make_archive('${DELIVERY_RUN_DIR}/${RETURN_ZIP%.zip}', 'zip', root_dir='${RUN_ROOT%/*}', base_dir='${RUN_ROOT##*/}')
PY
fi

tar -czf "${DELIVERY_RUN_DIR}/${RETURN_DETAILS}" \
  "${RUN_ROOT}/shards" "logs_b21/track_c/${RUN_ID}" \
  data/nist protocols/evidence_extension_v1/track_c
shasum -a 256 "${DELIVERY_RUN_DIR}/${RETURN_ZIP}" > "${DELIVERY_RUN_DIR}/${RETURN_ZIP}.sha256"
shasum -a 256 "${DELIVERY_RUN_DIR}/${RETURN_DETAILS}" > "${DELIVERY_RUN_DIR}/${RETURN_DETAILS}.sha256"

cat > "${DELIVERY_RUN_DIR}/OPEN_THIS_FOLDER.txt" <<EOF
B21 Track C completed successfully.

Mode: ${MODE}
Repository: ${PROJECT_ROOT}

Upload these four files:
1. ${DELIVERY_RUN_DIR}/${RETURN_ZIP}
2. ${DELIVERY_RUN_DIR}/${RETURN_ZIP}.sha256
3. ${DELIVERY_RUN_DIR}/${RETURN_DETAILS}
4. ${DELIVERY_RUN_DIR}/${RETURN_DETAILS}.sha256
EOF

printf '\nTRACK_C_MAC_RUN_OK\n'
printf 'Mode: %s\n' "${MODE}"
printf 'Delivery folder: %s\n' "${DELIVERY_RUN_DIR}"
if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
  open "${DELIVERY_RUN_DIR}"
  open -R "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
fi
