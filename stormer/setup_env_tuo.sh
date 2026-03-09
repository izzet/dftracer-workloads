#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORMER_MODULES="${STORMER_MODULES:-cray-python/3.11.7 gcc/13.3.1 rocm/6.3.1}"

# shellcheck disable=SC1091
source "${ROOT_DIR}/setup_env.sh"
