#!/usr/bin/env bash

# Portable CUTEst environment template.
BASINGRAPH_ROOT="${BASINGRAPH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CUTEST_PREFIX="${CUTEST_PREFIX:-${BASINGRAPH_ROOT}/cutest_stack/install}"
MASTSIF="${MASTSIF:-${BASINGRAPH_ROOT}/cutest_stack/mastsif}"
PYCUTEST_CACHE="${PYCUTEST_CACHE:-${BASINGRAPH_ROOT}/pycutest_cache}"

GCC_PREFIX="${GCC_PREFIX:-$(brew --prefix gcc 2>/dev/null || true)}"
GCC_LIB="${GCC_LIB:-${GCC_PREFIX}/lib/gcc/current}"

export SIFDECODE="${CUTEST_PREFIX}"
export CUTEST="${CUTEST_PREFIX}"
export MASTSIF
export PYCUTEST_CACHE

export PATH="${CUTEST_PREFIX}/bin:${PATH}"
export CPATH="${CUTEST_PREFIX}/include:${CPATH:-}"
export LIBRARY_PATH="${CUTEST_PREFIX}/lib:${GCC_LIB}:${LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${CUTEST_PREFIX}/lib:${GCC_LIB}:${DYLD_LIBRARY_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="${CUTEST_PREFIX}/lib:${GCC_LIB}:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${CUTEST_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
