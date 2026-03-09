#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORMER_MODULES="${STORMER_MODULES:-cray-python/3.11.7 gcc/13.3.1 rocm/6.3.1}"
export STORMER_PIP_EXTRA_INDEX_URL="${STORMER_PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

exec "${ROOT_DIR}/install.sh" "$@"
