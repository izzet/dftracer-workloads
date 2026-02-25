#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
INSTALL_DIR="${ROOT_DIR}/install"
MONTAGE_BIN="${ROOT_DIR}/montage/bin"

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

# Path for Montage and Pegasus (Condor) - per pegasus_montage.rst
export PATH="${INSTALL_DIR}/bin:${INSTALL_DIR}/sbin:${MONTAGE_BIN}:${PATH}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
# DFTracer from venv
DFTRACER_SITE="${VENV_DIR}/lib/python3.12/site-packages/dftracer"
for pyver in 3.12 3.11 3.10 3.9; do
  if [[ -d "${VENV_DIR}/lib/python${pyver}/site-packages/dftracer" ]]; then
    DFTRACER_SITE="${VENV_DIR}/lib/python${pyver}/site-packages/dftracer"
    break
  fi
done
export DFTRACER_INSTALLED="${DFTRACER_SITE}"
export LD_LIBRARY_PATH="${DFTRACER_SITE}/lib64:${DFTRACER_SITE}/lib:${LD_LIBRARY_PATH:-}"
