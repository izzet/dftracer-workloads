#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORMER_MODULES="${STORMER_MODULES:-cray-python/3.11.7 gcc/13.3.1 rocm/6.3.1}"
export LAUNCHER="${LAUNCHER:-flux}"
export NODES="${NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
export PRECISION="${PRECISION:-32}"

exec "${ROOT_DIR}/run.sh" "$@"
