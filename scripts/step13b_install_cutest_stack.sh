#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 13B
# Build and install SIFDecode + CUTEst locally on macOS arm64,
# then install and smoke-test PyCUTEst.
#
# No sudo is used. All files remain inside BasinGraph202606.
# ============================================================

ROOT="$HOME/Documents/BasinGraph202606"
STACK="$ROOT/cutest_stack"
PREFIX="$STACK/install"
SIFDECODE_SRC="$STACK/sifdecode"
CUTEST_SRC="$STACK/cutest"
MASTSIF_SRC="$STACK/mastsif"
CACHE_DIR="$ROOT/pycutest_cache"

cd "$ROOT"

# ------------------------------------------------------------
# 1. Confirm the dedicated Conda environment
# ------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "basingraph-cutest" ]]; then
    echo "ERROR: activate the basingraph-cutest environment first:"
    echo "  conda activate basingraph-cutest"
    exit 1
fi

mkdir -p \
    "$STACK" \
    "$PREFIX" \
    "$CACHE_DIR" \
    "$ROOT/logs" \
    "$ROOT/protocols"

# ------------------------------------------------------------
# 2. Select Homebrew GCC / gfortran
# ------------------------------------------------------------
BREW_PREFIX="$(brew --prefix)"
GCC_PREFIX="$(brew --prefix gcc)"

GFORTRAN="$(command -v gfortran || true)"
GCC_C="$(find "$GCC_PREFIX/bin" -maxdepth 1 \
    -type f -name 'gcc-[0-9]*' | sort | tail -1)"

if [[ -z "$GFORTRAN" || ! -x "$GFORTRAN" ]]; then
    echo "ERROR: gfortran was not found."
    exit 1
fi

if [[ -z "$GCC_C" || ! -x "$GCC_C" ]]; then
    echo "ERROR: versioned Homebrew gcc was not found."
    exit 1
fi

export CC="$GCC_C"
export FC="$GFORTRAN"
export F77="$GFORTRAN"
export F90="$GFORTRAN"

# Match object-file deployment targets on macOS.
export MACOSX_DEPLOYMENT_TARGET="$(
    sw_vers -productVersion | awk -F. '{print $1"."$2}'
)"

GCC_LIB="$GCC_PREFIX/lib/gcc/current"

# ------------------------------------------------------------
# 3. Runtime/build environment for the local prefix
# ------------------------------------------------------------
export PATH="$PREFIX/bin:$BREW_PREFIX/bin:$PATH"
export SIFDECODE="$PREFIX"
export CUTEST="$PREFIX"
export MASTSIF="$MASTSIF_SRC"
export PYCUTEST_CACHE="$CACHE_DIR"

export CPATH="$PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:${DYLD_LIBRARY_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# ------------------------------------------------------------
# 4. Clone repositories once
# ------------------------------------------------------------
clone_if_missing() {
    local url="$1"
    local destination="$2"

    if [[ -d "$destination/.git" ]]; then
        echo "Using existing repository: $destination"
    elif [[ -e "$destination" ]]; then
        echo "ERROR: $destination exists but is not a Git repository."
        exit 1
    else
        echo "Cloning: $url"
        git clone --depth 1 "$url" "$destination"
    fi
}

clone_if_missing \
    "https://github.com/ralna/SIFDecode.git" \
    "$SIFDECODE_SRC"

clone_if_missing \
    "https://github.com/ralna/CUTEst.git" \
    "$CUTEST_SRC"

clone_if_missing \
    "https://bitbucket.org/optrove/sif.git" \
    "$MASTSIF_SRC"

# ------------------------------------------------------------
# 5. Build and install SIFDecode
# ------------------------------------------------------------
echo "============================================================"
echo "Building SIFDecode"
echo "============================================================"

rm -rf "$SIFDECODE_SRC/builddir"

meson setup \
    "$SIFDECODE_SRC/builddir" \
    "$SIFDECODE_SRC" \
    --prefix="$PREFIX" \
    --buildtype=release

meson compile -C "$SIFDECODE_SRC/builddir"
meson install -C "$SIFDECODE_SRC/builddir"
meson test -C "$SIFDECODE_SRC/builddir" --print-errorlogs

echo "SIFDECODE_TESTS_OK"

# ------------------------------------------------------------
# 6. Build and install CUTEst in double precision
# ------------------------------------------------------------
echo "============================================================"
echo "Building CUTEst"
echo "============================================================"

rm -rf "$CUTEST_SRC/builddir"

meson setup \
    "$CUTEST_SRC/builddir" \
    "$CUTEST_SRC" \
    --prefix="$PREFIX" \
    --buildtype=release \
    -Dmodules=false \
    -Dtests=true

meson compile -C "$CUTEST_SRC/builddir"
meson install -C "$CUTEST_SRC/builddir"
meson test -C "$CUTEST_SRC/builddir" --print-errorlogs

echo "CUTEST_TESTS_OK"

# ------------------------------------------------------------
# 7. Verify installed executables and libraries
# ------------------------------------------------------------
echo "============================================================"
echo "Installed prefix contents"
echo "============================================================"

find "$PREFIX/bin" -maxdepth 1 -type f -print 2>/dev/null | sort || true
find "$PREFIX/lib" -maxdepth 2 -type f -print 2>/dev/null | sort | head -50 || true

if command -v sifdecoder >/dev/null 2>&1; then
    echo "sifdecoder: $(command -v sifdecoder)"
elif command -v sifdecode >/dev/null 2>&1; then
    echo "sifdecode: $(command -v sifdecode)"
else
    echo "ERROR: no SIFDecode executable was found under PATH."
    exit 1
fi

# ------------------------------------------------------------
# 8. Install PyCUTEst 1.8.2
# ------------------------------------------------------------
python -m pip install --upgrade pip
python -m pip install --upgrade "pycutest==1.8.2"

# ------------------------------------------------------------
# 9. Save a reusable environment file
# ------------------------------------------------------------
ENV_FILE="$ROOT/protocols/cutest_env.sh"

cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash

export CC="$CC"
export FC="$FC"
export F77="$F77"
export F90="$F90"

export MACOSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"

export SIFDECODE="$SIFDECODE"
export CUTEST="$CUTEST"
export MASTSIF="$MASTSIF"
export PYCUTEST_CACHE="$PYCUTEST_CACHE"

export PATH="$PREFIX/bin:\$PATH"
export CPATH="$PREFIX/include:\${CPATH:-}"
export LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:\${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:\${DYLD_LIBRARY_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="$PREFIX/lib:$GCC_LIB:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
EOF

chmod +x "$ENV_FILE"

# ------------------------------------------------------------
# 10. Record versions and Git commits
# ------------------------------------------------------------
VERSION_FILE="$ROOT/protocols/step13b_cutest_stack_versions.txt"

{
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Architecture: $(uname -m)"
    echo "macOS: $(sw_vers -productVersion)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV}"
    echo "Python: $(python --version 2>&1)"
    echo "Meson: $(meson --version)"
    echo "Ninja: $(ninja --version)"
    echo "CC: $CC"
    echo "CC version: $($CC --version | head -1)"
    echo "FC: $FC"
    echo "FC version: $($FC --version | head -1)"
    echo "MACOSX_DEPLOYMENT_TARGET: $MACOSX_DEPLOYMENT_TARGET"
    echo "SIFDECODE prefix: $SIFDECODE"
    echo "CUTEST prefix: $CUTEST"
    echo "MASTSIF: $MASTSIF"
    echo "PYCUTEST_CACHE: $PYCUTEST_CACHE"
    echo "SIFDecode commit: $(git -C "$SIFDECODE_SRC" rev-parse HEAD)"
    echo "CUTEst commit: $(git -C "$CUTEST_SRC" rev-parse HEAD)"
    echo "MASTSIF commit: $(git -C "$MASTSIF_SRC" rev-parse HEAD)"
} | tee "$VERSION_FILE"

# ------------------------------------------------------------
# 11. PyCUTEst smoke test using ROSENBR
# ------------------------------------------------------------
source "$ENV_FILE"

python - <<'PY'
from pathlib import Path
import numpy as np
import pycutest

print("PyCUTEst version:", getattr(pycutest, "__version__", "unknown"))
print("SIFDECODE:", Path(__import__("os").environ["SIFDECODE"]))
print("CUTEST:", Path(__import__("os").environ["CUTEST"]))
print("MASTSIF:", Path(__import__("os").environ["MASTSIF"]))
print("PYCUTEST_CACHE:", Path(__import__("os").environ["PYCUTEST_CACHE"]))

problem = pycutest.import_problem(
    "ROSENBR",
    quiet=False,
)

x0 = np.asarray(problem.x0, dtype=float)
f0 = float(problem.obj(x0))

print("Problem:", problem.name)
print("Variables:", problem.n)
print("Constraints:", problem.m)
print("Initial point:", x0)
print("Objective at initial point:", f0)

assert problem.name == "ROSENBR"
assert problem.n >= 2
assert np.isfinite(f0)

if hasattr(problem, "terminate"):
    problem.terminate()

print("PYCUTEST_IMPORT_OK")
print("ROSENBR_SMOKE_OK")
PY

# ------------------------------------------------------------
# 12. Freeze the Python dependency state
# ------------------------------------------------------------
python -m pip freeze \
    > "$ROOT/protocols/step13b_cutest_requirements-lock.txt"

echo "============================================================"
echo "STEP_13B_OK"
echo "Environment file: $ENV_FILE"
echo "Version record: $VERSION_FILE"
echo "Cache directory: $CACHE_DIR"
echo "============================================================"
