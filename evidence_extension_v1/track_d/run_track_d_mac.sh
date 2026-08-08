#!/bin/bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke|confirmatory) ;;
  *)
    echo "Usage: bash evidence_extension_v1/track_d/run_track_d_mac.sh [smoke|confirmatory]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1 \
   && [ "${B21_TRACK_D_CAFFEINATED:-0}" != "1" ]; then
  export B21_TRACK_D_CAFFEINATED=1
  exec caffeinate -dimsu bash "$0" "${MODE}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${B21_TRACK_D_VENV_DIR:-${PROJECT_ROOT}/.venv-track-d}"
DELIVERY_ROOT="${B21_DELIVERY_DIR:-${HOME}/Downloads/B21_RESULTS_READY}"
BASE_SEED="${B21_TRACK_D_BASE_SEED:-20260808}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Creating Track D virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r \
  protocols/evidence_extension_v1/track_d/TRACK_D_REQUIREMENTS_LOCK.txt
python -m pip install -e . --no-deps

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IDENTITY_AUDIT="results_b21/track_d/identity_audit_${MODE}_${TIMESTAMP}.json"
PREFLIGHT="results_b21/track_d/preflight_${MODE}_${TIMESTAMP}.json"

printf '\n=== Track D protocol and repository manifests ===\n'
python evidence_extension_v1/track_d/generate_protocol_manifest.py
python scripts_v2/generate_repository_manifest.py

printf '\n=== Step 0 identity audit ===\n'
python scripts_v2/audit_step0_identity.py \
  --require-manifest \
  --output "${IDENTITY_AUDIT}"

if [ "${MODE}" = "smoke" ]; then
  printf '\n=== Track D zero-evaluation smoke preflight ===\n'
  python evidence_extension_v1/track_d/preflight_track_d.py \
    --mode smoke \
    --output "${PREFLIGHT}"

  RUN_ID="b21_track_d_smoke_${TIMESTAMP}"
  printf '\n=== Track D 30-run development smoke ===\n'
  python evidence_extension_v1/track_d/run_track_d_all.py \
    --mode smoke \
    --run-id "${RUN_ID}" \
    --workers "${B21_TRACK_D_SMOKE_WORKERS:-2}" \
    --base-seed "${BASE_SEED}"
  python evidence_extension_v1/track_d/finalize_track_d.py \
    --mode smoke \
    --run-id "${RUN_ID}" \
    --base-seed "${BASE_SEED}"

  RUN_ROOT="results_b21/track_d/${RUN_ID}"
  cp -f "${IDENTITY_AUDIT}" "${RUN_ROOT}/identity_audit.json"
  cp -f "${PREFLIGHT}" "${RUN_ROOT}/preflight.json"
  cp -f results_b21/track_d/LOCAL_SMOKE_GATE.json \
    "${RUN_ROOT}/LOCAL_SMOKE_GATE.json"

  DELIVERY_RUN_DIR="${DELIVERY_ROOT}/trackD_smoke_${TIMESTAMP}"
  mkdir -p "${DELIVERY_RUN_DIR}"
  RETURN_ZIP="B21_TRACK_D_smoke_${TIMESTAMP}.zip"
  RETURN_OBSERVER="B21_TRACK_D_smoke_observer_${TIMESTAMP}.tar.gz"

  if command -v ditto >/dev/null 2>&1; then
    ditto -c -k --sequesterRsrc --keepParent \
      "${RUN_ROOT}" "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
  else
    python - <<PY
import shutil
shutil.make_archive(
    '${DELIVERY_RUN_DIR}/${RETURN_ZIP%.zip}',
    'zip',
    root_dir='results_b21/track_d',
    base_dir='${RUN_ID}',
)
PY
  fi
  tar -czf "${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}" \
    "exdata/b21_track_d/${RUN_ID}"

else
  printf '\n=== Verify exact local smoke gate ===\n'
  python - <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess

root = Path('.').resolve()
gate_path = root / 'results_b21/track_d/LOCAL_SMOKE_GATE.json'
identity_path = root / 'protocols/evidence_extension_v1/track_d/TRACK_D_SOURCE_IDENTITY.json'
if not gate_path.is_file():
    raise SystemExit('Missing Track D local smoke gate. Run smoke first.')
gate = json.loads(gate_path.read_text())
if gate.get('status') != 'TRACK_D_LOCAL_SMOKE_GATE_OK':
    raise SystemExit('Track D smoke gate is not valid.')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if gate.get('source_commit') != head:
    raise SystemExit(
        'Smoke-gate source commit mismatch:\n'
        f"gate={gate.get('source_commit')}\nhead={head}"
    )
digest = hashlib.sha256(identity_path.read_bytes()).hexdigest()
if gate.get('source_identity_sha256') != digest:
    raise SystemExit('Smoke-gate source identity mismatch.')
if int(gate.get('confirmatory_objective_evaluations', -1)) != 0:
    raise SystemExit('Smoke gate reports confirmatory evaluations.')
print('TRACK_D_LOCAL_SMOKE_GATE_OK')
print(json.dumps(gate, indent=2))
PY

  OVERHEAD_PREFLIGHT="results_b21/track_d/preflight_overhead_${TIMESTAMP}.json"
  CONFIRMATORY_PREFLIGHT="results_b21/track_d/preflight_confirmatory_${TIMESTAMP}.json"

  printf '\n=== Track D zero-evaluation overhead preflight ===\n'
  python evidence_extension_v1/track_d/preflight_track_d.py \
    --mode overhead \
    --output "${OVERHEAD_PREFLIGHT}"

  printf '\n=== Track D zero-evaluation confirmatory preflight ===\n'
  python evidence_extension_v1/track_d/preflight_track_d.py \
    --mode confirmatory \
    --authorize-confirmatory \
    --output "${CONFIRMATORY_PREFLIGHT}"

  OVERHEAD_RUN_ID="b21_track_d_overhead_${TIMESTAMP}"
  CONFIRMATORY_RUN_ID="b21_track_d_confirmatory_${TIMESTAMP}"

  printf '\n=== Track D registered sequential overhead probe ===\n'
  python evidence_extension_v1/track_d/run_track_d_all.py \
    --mode overhead \
    --run-id "${OVERHEAD_RUN_ID}" \
    --workers 1 \
    --base-seed "${BASE_SEED}"
  python evidence_extension_v1/track_d/finalize_track_d.py \
    --mode overhead \
    --run-id "${OVERHEAD_RUN_ID}" \
    --base-seed "${BASE_SEED}"

  printf '\n=== Track D 1,440-run large-scale confirmatory matrix ===\n'
  python evidence_extension_v1/track_d/run_track_d_all.py \
    --mode confirmatory \
    --run-id "${CONFIRMATORY_RUN_ID}" \
    --workers "${B21_TRACK_D_CONFIRMATORY_WORKERS:-4}" \
    --base-seed "${BASE_SEED}" \
    --authorize-confirmatory
  python evidence_extension_v1/track_d/finalize_track_d.py \
    --mode confirmatory \
    --run-id "${CONFIRMATORY_RUN_ID}" \
    --base-seed "${BASE_SEED}" \
    --authorize-confirmatory

  DELIVERY_RUN_DIR="${DELIVERY_ROOT}/trackD_confirmatory_${TIMESTAMP}"
  mkdir -p "${DELIVERY_RUN_DIR}"
  STAGING="${DELIVERY_RUN_DIR}/track_d_evidence_${TIMESTAMP}"
  mkdir -p "${STAGING}"
  cp -R "results_b21/track_d/${OVERHEAD_RUN_ID}" \
    "${STAGING}/overhead"
  cp -R "results_b21/track_d/${CONFIRMATORY_RUN_ID}" \
    "${STAGING}/confirmatory"
  cp -f "${IDENTITY_AUDIT}" "${STAGING}/identity_audit.json"
  cp -f "${OVERHEAD_PREFLIGHT}" "${STAGING}/preflight_overhead.json"
  cp -f "${CONFIRMATORY_PREFLIGHT}" \
    "${STAGING}/preflight_confirmatory.json"
  cp -f results_b21/track_d/LOCAL_SMOKE_GATE.json \
    "${STAGING}/LOCAL_SMOKE_GATE.json"

  RETURN_ZIP="B21_TRACK_D_confirmatory_${TIMESTAMP}.zip"
  RETURN_OBSERVER="B21_TRACK_D_confirmatory_observer_${TIMESTAMP}.tar.gz"
  if command -v ditto >/dev/null 2>&1; then
    ditto -c -k --sequesterRsrc --keepParent \
      "${STAGING}" "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
  else
    python - <<PY
import shutil
shutil.make_archive(
    '${DELIVERY_RUN_DIR}/${RETURN_ZIP%.zip}',
    'zip',
    root_dir='${DELIVERY_RUN_DIR}',
    base_dir='track_d_evidence_${TIMESTAMP}',
)
PY
  fi
  tar -czf "${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}" \
    "exdata/b21_track_d/${OVERHEAD_RUN_ID}" \
    "exdata/b21_track_d/${CONFIRMATORY_RUN_ID}"
fi

shasum -a 256 "${DELIVERY_RUN_DIR}/${RETURN_ZIP}" \
  > "${DELIVERY_RUN_DIR}/${RETURN_ZIP}.sha256"
shasum -a 256 "${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}" \
  > "${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}.sha256"

cat > "${DELIVERY_RUN_DIR}/OPEN_THIS_FOLDER.txt" <<EOF
B21 Track D completed successfully.

Mode: ${MODE}
Repository: ${PROJECT_ROOT}

Upload these four files:
1. ${DELIVERY_RUN_DIR}/${RETURN_ZIP}
2. ${DELIVERY_RUN_DIR}/${RETURN_ZIP}.sha256
3. ${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}
4. ${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}.sha256
EOF

printf '\nTRACK_D_MAC_RUN_OK\n'
printf 'Mode: %s\n' "${MODE}"
printf 'Delivery folder: %s\n' "${DELIVERY_RUN_DIR}"
printf 'Upload ZIP: %s\n' "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
printf 'Upload observer archive: %s\n' "${DELIVERY_RUN_DIR}/${RETURN_OBSERVER}"

if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
  open "${DELIVERY_RUN_DIR}"
  open -R "${DELIVERY_RUN_DIR}/${RETURN_ZIP}"
fi

printf '\nFinder has been opened at the exact result folder.\n'
