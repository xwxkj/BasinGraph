#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/BasinGraph202606

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate basingraph-cutest

echo "============================================================"
echo "Step 13A: CUTEst/PyCUTEst macOS toolchain preparation"
echo "============================================================"

# ------------------------------------------------------------
# 1. Check Xcode command-line tools
# ------------------------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Xcode command-line tools are missing."
    echo "Launching installer..."
    xcode-select --install
    echo "Complete the graphical installation, then rerun this script."
    exit 1
fi

# ------------------------------------------------------------
# 2. Check Homebrew
# ------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew is not installed."
    echo "Install it from https://brew.sh and rerun this script."
    exit 1
fi

# ------------------------------------------------------------
# 3. Install required build tools
# ------------------------------------------------------------
brew install gcc meson ninja pkg-config git

BREW_PREFIX="$(brew --prefix)"
export PATH="${BREW_PREFIX}/bin:${PATH}"

# ------------------------------------------------------------
# 4. Locate Homebrew gfortran
# ------------------------------------------------------------
if command -v gfortran >/dev/null 2>&1; then
    GFORTRAN="$(command -v gfortran)"
else
    GFORTRAN="$(find "${BREW_PREFIX}/bin" -maxdepth 1 \
        -type l -o -type f 2>/dev/null \
        | grep '/gfortran-[0-9][0-9]*$' \
        | sort -V \
        | tail -1 || true)"
fi

if [[ -z "${GFORTRAN}" || ! -x "${GFORTRAN}" ]]; then
    echo "ERROR: Homebrew GCC was installed, but gfortran was not located."
    exit 1
fi

export FC="${GFORTRAN}"
export F77="${GFORTRAN}"
export F90="${GFORTRAN}"

# ------------------------------------------------------------
# 5. Save reproducible toolchain information
# ------------------------------------------------------------
OUT="protocols/step13a_cutest_toolchain.txt"

{
    echo "Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Architecture: $(uname -m)"
    echo "macOS: $(sw_vers -productVersion)"
    echo "Python: $(python --version 2>&1)"
    echo "Python executable: $(which python)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
    echo "Homebrew prefix: ${BREW_PREFIX}"
    echo "Xcode tools: $(xcode-select -p)"
    echo "Clang: $(clang --version | head -1)"
    echo "gfortran executable: ${GFORTRAN}"
    echo "gfortran: $("${GFORTRAN}" --version | head -1)"
    echo "Meson: $(meson --version)"
    echo "Ninja: $(ninja --version)"
    echo "Git: $(git --version)"
    echo "CMake: $(cmake --version 2>/dev/null | head -1 || echo 'not installed')"
} | tee "${OUT}"

# ------------------------------------------------------------
# 6. Basic compiler smoke tests
# ------------------------------------------------------------
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_LOCAL}"' EXIT

cat > "${TMPDIR_LOCAL}/hello.f90" <<'F90'
program hello_cutest
    implicit none
    print *, "GFORTRAN_SMOKE_OK"
end program hello_cutest
F90

"${GFORTRAN}" "${TMPDIR_LOCAL}/hello.f90" \
    -o "${TMPDIR_LOCAL}/hello_cutest"

"${TMPDIR_LOCAL}/hello_cutest"

cat > "${TMPDIR_LOCAL}/hello.c" <<'C'
#include <stdio.h>
int main(void) {
    printf("CLANG_SMOKE_OK\n");
    return 0;
}
C

clang "${TMPDIR_LOCAL}/hello.c" \
    -o "${TMPDIR_LOCAL}/hello_c"

"${TMPDIR_LOCAL}/hello_c"

echo "STEP_13A_OK"
