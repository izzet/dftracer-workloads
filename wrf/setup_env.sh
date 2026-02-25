#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

source ~/spack/share/spack/setup-env.sh

if [[ ! -f "${ROOT_DIR}/spack.yaml" ]]; then
  echo "Missing ${ROOT_DIR}/spack.yaml." >&2
  return 1 2>/dev/null || exit 1
fi
spack env activate "${ROOT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./install.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export WORKLOAD_ROOT="${ROOT_DIR}"
export WORKLOAD_LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${WORKLOAD_LOG_DIR}"
